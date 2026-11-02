import os
import json
import numpy as np
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from semantic_modules import (
    DSM5KnowledgeBase,
    KnowledgeGraphManager,
    ExplanationGenerator,
)


def parse_explanation_vectors(metrics_list):
    """Lee el vector de explicación (JSON string) de cada cliente."""
    exp_vectors = []
    for m in metrics_list:
        vec = m.get("explanation_vector")
        if vec is None:
            raw_v = m.get("explanation_vector_json")
            if raw_v:
                try:
                    vec = json.loads(raw_v)
                except Exception:
                    vec = []
        exp_vectors.append(vec or [])
    return exp_vectors


def compute_semantic_diversity(exp_vectors):
    """
    Diversidad Semántica SD_i = 1 - cos(e_i, centroide de los demás clientes), en [0, 1].
    Devuelve (sd_vals, has_vectors). Si no hay vectores comparables, sd_vals queda en 1.0.
    """
    K = len(exp_vectors)
    sd_vals = np.ones(K, dtype=np.float32)
    has_vectors = K > 0 and all(len(v) > 0 for v in exp_vectors) and len({len(v) for v in exp_vectors}) == 1
    if has_vectors and K > 1:
        exp_mat = np.array(exp_vectors, dtype=np.float32)
        for i in range(K):
            other_indices = [j for j in range(K) if j != i]
            centroid = np.mean(exp_mat[other_indices], axis=0)
            norm_i = np.linalg.norm(exp_mat[i])
            norm_c = np.linalg.norm(centroid)
            if norm_i > 1e-6 and norm_c > 1e-6:
                sim = float(np.dot(exp_mat[i], centroid) / (norm_i * norm_c))
                sd_vals[i] = float(np.clip(1.0 - sim, 0.0, 1.0))
            else:
                sd_vals[i] = 0.5
    elif K == 1:
        sd_vals[0] = 1.0
    return sd_vals, has_vectors


def macro_f1_from_confusions(confusions):
    """
    Macro-F1 global a partir de las matrices de confusión de todos los clientes (se suman,
    filas = clase real, columnas = predicha). Se promedia solo sobre clases con soporte.
    """
    mats = [np.array(c, dtype=np.float64) for c in confusions if c]
    if not mats:
        return None
    cm = np.sum(mats, axis=0)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = cm.sum(axis=1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


class CustomStrategy(fl.server.strategy.FedAvg):
    """
    Estrategia personalizada basada en FedAvg estándar:
    - Filtra clientes que reportaron fallo.
    - Rastrea el mejor modelo y su precisión.
    - Calcula la métrica de tráfico de red real para los parámetros transmitidos.
    - Registra por ronda las métricas de explicabilidad (EQ, SR, DC, SD) y el Macro-F1, de forma
      idéntica para FedAvg y SemFedXAI para que la comparación sea justa.
    """
    def __init__(self, client_fn=None, **kwargs):
        super().__init__(evaluate_metrics_aggregation_fn=self.aggregate_metrics, **kwargs)
        self.client_fn = client_fn
        self.best_accuracy = 0.0
        self.best_model = None
        self.history_accuracy = []
        self.history_macro_f1 = []
        # Métricas de explicabilidad por ronda (promedio entre clientes)
        self.history_eq = []          # Explanation Quality (EQ), independiente de la clase predicha
        self.history_eq_chance = []   # EQ esperada por puro azar (referencia)
        self.history_eq_pred = []     # EQ condicionada a la clase predicha (control)
        self.history_sr = []          # Relevancia Semántica
        self.history_dc = []          # Consistencia de Dominio
        self.history_sd = []          # Diversidad Semántica
        self.latest_eq = 0.0
        self.latest_sr = 0.0
        self.latest_dc = 0.0
        self.latest_sd = 0.0
        self.latest_macro_f1 = 0.0
        self.latest_fit_accuracy = 0.0
        self.latest_num_clients = 0
        self.history_network_mb = []
        # Último modelo global agregado (lista de ndarrays) para poder guardarlo al terminar
        self.final_ndarrays = None
        # Peso real (MB) de los parámetros que envía un cliente, medido en aggregate_fit
        self.payload_mb_per_client = 0.0
        self.payload_params_per_client = 0

    @staticmethod
    def _mean_metric(metrics_list, key):
        vals = [float(m[key]) for m in metrics_list if key in m]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _append_or_carry(history, value):
        """Agrega el valor; si no hay dato esta ronda, arrastra el anterior para mantener el largo alineado."""
        if value is None:
            value = history[-1] if history else 0.0
        history.append(float(value))
        return float(value)

    def record_round_metrics(self, metrics_list, sd_vals=None):
        """
        Registra las métricas de explicabilidad de la ronda a partir de los metrics de fit de los
        clientes (lista de dicts). Se usa tanto desde aggregate_fit (Ray) como desde run_sequential.
        """
        self.latest_eq = self._append_or_carry(self.history_eq, self._mean_metric(metrics_list, "explanation_quality"))
        self._append_or_carry(self.history_eq_chance, self._mean_metric(metrics_list, "explanation_chance"))
        self._append_or_carry(self.history_eq_pred, self._mean_metric(metrics_list, "explanation_quality_pred"))
        self.latest_sr = self._append_or_carry(self.history_sr, self._mean_metric(metrics_list, "semantic_relevance"))
        self.latest_dc = self._append_or_carry(self.history_dc, self._mean_metric(metrics_list, "domain_consistency"))

        if sd_vals is None:
            sd_vals, has_vectors = compute_semantic_diversity(parse_explanation_vectors(metrics_list))
            sd_mean = float(np.mean(sd_vals)) if has_vectors else None
        else:
            sd_mean = float(np.mean(sd_vals))
        self.latest_sd = self._append_or_carry(self.history_sd, sd_mean)

    def aggregate_fit(self, server_round, results, failures):
        # Filtrar clientes fallidos
        valid_results = [r for r in results if not r[1].metrics.get("failed", False)]
        accuracies = [res.metrics["accuracy"] for _, res in valid_results if "accuracy" in res.metrics]

        # Medir el payload real que viaja por la red (bytes de los tensores serializados)
        if valid_results:
            first_params = valid_results[0][1].parameters
            payload_bytes = sum(len(t) for t in first_params.tensors)
            self.payload_mb_per_client = payload_bytes / (1024 ** 2)
            self.payload_params_per_client = payload_bytes // 4  # float32
            self.latest_num_clients = len(valid_results)

        self.record_round_metrics([res.metrics for _, res in valid_results])

        if accuracies:
            self.latest_fit_accuracy = float(np.mean(accuracies))
            best_idx = np.argmax(accuracies)
            best_accuracy = accuracies[best_idx]

            if best_accuracy > self.best_accuracy:
                self.best_accuracy = best_accuracy
                self.best_model = valid_results[best_idx][1].parameters
                print(f"[INFO] Nuevo mejor modelo (Accuracy: {best_accuracy:.2%})")

        aggregated, metrics = super().aggregate_fit(server_round, valid_results, failures)
        if aggregated is not None:
            self.final_ndarrays = parameters_to_ndarrays(aggregated)
        return aggregated, metrics

    def aggregate_metrics(self, metrics):
        """Registra el promedio de precisión de los clientes."""
        accuracies = [m["accuracy"] for _, m in metrics if "accuracy" in m]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        return {"accuracy": avg_accuracy}

    def record_eval_metrics(self, accuracy, confusions):
        """Registra Macro-F1 global de la ronda (si no hay datos de evaluación, arrastra el anterior)."""
        macro_f1 = macro_f1_from_confusions(confusions)
        self.latest_macro_f1 = self._append_or_carry(self.history_macro_f1, macro_f1)

    def print_round_summary(self, server_round, avg_accuracy, total_mb_round=0.0, loss=None):
        print(f"--- Ronda {server_round} ---")
        extra = f"  (loss media: {loss:.4f})" if loss is not None else ""
        print(f"Accuracy: {avg_accuracy:.2%}{extra} | Macro-F1: {self.latest_macro_f1:.4f}")
        if self.history_eq:
            chance = self.history_eq_chance[-1] if self.history_eq_chance else 0.0
            print(
                f"Explicabilidad -> EQ: {self.latest_eq:.4f} (azar: {chance:.4f}) | "
                f"SR: {self.latest_sr:.4f} | DC: {self.latest_dc:.4f} | SD: {self.latest_sd:.4f}"
            )
        if total_mb_round > 0:
            print(f"Trafico de Red: {total_mb_round:.2f} MB")
        print("-" * 20)

    def aggregate_evaluate(self, server_round, results, failures):
        """Calcula Accuracy, Macro-F1 y Tráfico de Red Real."""
        # 1. Accuracy promedio (si evaluate no fue muestreado por Flower, usa el promedio de fit de la ronda)
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        if accuracies:
            avg_accuracy = float(np.mean(accuracies))
        elif getattr(self, "latest_fit_accuracy", 0.0) > 0.0:
            avg_accuracy = self.latest_fit_accuracy
        else:
            avg_accuracy = 0.0
        self.history_accuracy.append(avg_accuracy)

        # Macro-F1 global (suma de matrices de confusión de todos los clientes)
        confusions = []
        for _, r in results:
            raw = r.metrics.get("confusion_json")
            if raw:
                try:
                    confusions.append(json.loads(raw))
                except Exception:
                    pass
        self.record_eval_metrics(avg_accuracy, confusions)

        # Calculo de trafico de red
        total_mb_round = 0.0
        if self.payload_mb_per_client > 0:
            n_clients = len(results) if len(results) > 0 else getattr(self, "latest_num_clients", 12)
            total_mb_round = n_clients * self.payload_mb_per_client * 2
            self.history_network_mb.append(total_mb_round)

        # Registro en consola
        self.print_round_summary(server_round, avg_accuracy, total_mb_round)

        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        eval_metrics = {"accuracy": avg_accuracy, "network_mb": total_mb_round, "macro_f1": self.latest_macro_f1}
        if self.latest_eq > 0.0:
            eval_metrics["explanation_quality"] = self.latest_eq
        return loss_aggregated, eval_metrics


class SemFedXAIStrategy(CustomStrategy):
    """
    Estrategia de Agregación Semántica SemFedXAI (Amato & Branco, 2025):
    Sustituye el promedio numérico ciego tradicional por la ponderación semántica:
      w_i^sem = n_i * (1 + alpha * SR_i) * (1 + beta * DC_i) * (gamma ^ SD_i)
    donde:
      - SR_i: Relevancia Semántica de los criterios DSM-5 activados.
      - DC_i: Consistencia con las relaciones de la Ontología del DSM-5.
      - SD_i: Diversidad Semántica (complementariedad con respecto al resto de clientes).
    """
    def __init__(
        self,
        client_fn=None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.2,
        knowledge_base: DSM5KnowledgeBase = None,
        class_names: list = None,
        data_dir: str = None,
        **kwargs
    ):
        super().__init__(client_fn=client_fn, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.class_names = class_names or ["Depresion", "Ansiedad", "BPD", "PTSD"]
        self.data_dir = data_dir

        self.kb = knowledge_base or DSM5KnowledgeBase(base_dir=data_dir)
        self.kg_manager = KnowledgeGraphManager(self.kb, self.class_names)
        self.expl_generator = ExplanationGenerator(self.kb, self.class_names)

    def aggregate_fit(self, server_round, results, failures):
        valid_results = [r for r in results if not r[1].metrics.get("failed", False)]
        if not valid_results:
            return None, {}

        # Medir carga de red comunicada
        first_params = valid_results[0][1].parameters
        payload_bytes = sum(len(t) for t in first_params.tensors)
        self.payload_mb_per_client = payload_bytes / (1024 ** 2)
        self.payload_params_per_client = payload_bytes // 4

        # Metricas semanticas de clientes
        n_samples = np.array([r[1].num_examples for r in valid_results], dtype=np.float32)
        accuracies = [r[1].metrics.get("accuracy", 0.0) for r in valid_results]
        sr_vals = np.array([float(r[1].metrics.get("semantic_relevance", 0.0)) for r in valid_results], dtype=np.float32)
        dc_vals = np.array([float(r[1].metrics.get("domain_consistency", 0.0)) for r in valid_results], dtype=np.float32)
        eq_vals = np.array([float(r[1].metrics.get("explanation_quality", 0.0)) for r in valid_results], dtype=np.float32)
        exp_vectors = parse_explanation_vectors([r[1].metrics for r in valid_results])

        # Calculo de Diversidad Semantica (SD_i)
        K = len(valid_results)
        sd_vals, has_vectors = compute_semantic_diversity(exp_vectors)

        # Ponderacion Semantica SemFedXAI
        # w_i^sem = n_i * (1 + alpha * SR_i) * (1 + beta * DC_i) * (gamma ^ SD_i)
        raw_sem_weights = (
            n_samples *
            (1.0 + self.alpha * sr_vals) *
            (1.0 + self.beta * dc_vals) *
            np.power(self.gamma, sd_vals)
        )
        total_w = float(np.sum(raw_sem_weights))
        normalized_weights = raw_sem_weights / (total_w if total_w > 0 else 1.0)

        # Agregacion ponderada con los pesos semanticos calculados
        weights_results = [
            (parameters_to_ndarrays(r[1].parameters), float(w))
            for r, w in zip(valid_results, normalized_weights)
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.final_ndarrays = aggregated_ndarrays

        # Actualizacion del Knowledge Graph Manager global
        client_class_exps = []
        client_class_counts = []
        for r in valid_results:
            raw_ce = r[1].metrics.get("class_explanations_json")
            if raw_ce:
                try:
                    client_class_exps.append(json.loads(raw_ce))
                except Exception:
                    client_class_exps.append([])
            else:
                client_class_exps.append([])

            cnt = r[1].metrics.get("class_counts")
            if cnt is None:
                raw_cnt = r[1].metrics.get("class_counts_json")
                if raw_cnt:
                    try:
                        cnt = json.loads(raw_cnt)
                    except Exception:
                        cnt = []
            client_class_counts.append(cnt or [])

        if has_vectors and any(len(e) > 0 for e in client_class_exps):
            self.kg_manager.update_graph(
                normalized_weights.tolist(),
                client_class_exps,
                client_class_counts,
                server_round
            )
            kg_path = os.path.join(self.data_dir or ".", "global_knowledge_graph_latest.ttl")
            try:
                self.kg_manager.export_turtle(kg_path)
            except Exception as e:
                print(f"[WARN] Error al exportar grafo Turtle: {e}")

        # Seguimiento del mejor modelo
        self.latest_num_clients = len(valid_results)
        if accuracies:
            self.latest_fit_accuracy = float(np.mean(accuracies))
            best_idx = np.argmax(accuracies)
            best_accuracy = accuracies[best_idx]
            if best_accuracy > self.best_accuracy:
                self.best_accuracy = best_accuracy
                self.best_model = parameters_aggregated
                print(f"[INFO] Nuevo mejor modelo global SemFedXAI (Accuracy: {best_accuracy:.2%})")

        # Registro de metricas historicas (EQ, EQ azar, EQ_pred, SR, DC, SD): mismo calculo que FedAvg
        self.record_round_metrics([r[1].metrics for r in valid_results], sd_vals=sd_vals)

        # Reporte de ponderacion semantica en consola
        print(f"\n" + "=" * 70)
        print(f"AGREGACION SEMANTICA SemFedXAI — RONDA {server_round}".center(70))
        print("=" * 70)
        print(f"{'Cliente':<8} | {'Muestras':<8} | {'SR_i':<7} | {'DC_i':<7} | {'EQ_i':<7} | {'SD_i':<7} | {'Peso Sem':<10}")
        print("-" * 70)
        for i in range(K):
            print(f"C_{i:<6} | {int(n_samples[i]):<8} | {sr_vals[i]:<7.3f} | {dc_vals[i]:<7.3f} | {eq_vals[i]:<7.3f} | {sd_vals[i]:<7.3f} | {normalized_weights[i]:<9.2%}")
        print("-" * 70)
        print(f"Promedios: SR={np.mean(sr_vals):.3f} | DC={np.mean(dc_vals):.3f} | EQ={self.latest_eq:.3f} | SD={np.mean(sd_vals):.3f}")

        # Generacion de explicaciones clinicas
        if has_vectors and self.kg_manager.last_round > 0:
            print("\nNARRATIVAS CLINICAS EXPLICABLES (Explanation Generator):")
            for c_idx in range(min(2, len(self.class_names))):
                narrative = self.expl_generator.global_class_narrative(self.kg_manager, c_idx, top_k=2)
                print(narrative)
                print()
        print("=" * 70 + "\n")

        return parameters_aggregated, {
            "avg_sr": float(np.mean(sr_vals)),
            "avg_dc": float(np.mean(dc_vals)),
            "avg_sd": float(np.mean(sd_vals)),
            "avg_eq": self.latest_eq,
        }
