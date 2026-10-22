import numpy as np
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from semantic_modules import (
    DSM5KnowledgeBase,
    KnowledgeGraphManager,
    ExplanationGenerator,
)


class CustomStrategy(fl.server.strategy.FedAvg):
    """
    Estrategia personalizada basada en FedAvg estándar:
    - Filtra clientes que reportaron fallo.
    - Rastrea el mejor modelo y su precisión.
    - Calcula la métrica de tráfico de red real para los parámetros transmitidos.
    """
    def __init__(self, client_fn=None, **kwargs):
        super().__init__(evaluate_metrics_aggregation_fn=self.aggregate_metrics, **kwargs)
        self.client_fn = client_fn
        self.best_accuracy = 0.0
        self.best_model = None
        self.history_accuracy = []
        self.history_network_mb = []
        # Peso real (MB) de los parámetros que envía un cliente, medido en aggregate_fit
        self.payload_mb_per_client = 0.0
        self.payload_params_per_client = 0

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

        if accuracies:
            best_idx = np.argmax(accuracies)
            best_accuracy = accuracies[best_idx]

            if best_accuracy > self.best_accuracy:
                self.best_accuracy = best_accuracy
                self.best_model = valid_results[best_idx][1].parameters
                print(f"✅ Nuevo mejor modelo (Accuracy: {best_accuracy:.2%})")

        return super().aggregate_fit(server_round, valid_results, failures)

    def aggregate_metrics(self, metrics):
        """Registra el promedio de precisión de los clientes."""
        accuracies = [m["accuracy"] for _, m in metrics if "accuracy" in m]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        return {"accuracy": avg_accuracy}

    def aggregate_evaluate(self, server_round, results, failures):
        """Calcula Accuracy y Tráfico de Red Real."""
        # 1. Accuracy promedio
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        self.history_accuracy.append(avg_accuracy)

        # 2. CÁLCULO DE RED REAL
        total_mb_round = 0.0
        if self.payload_mb_per_client > 0:
            # Tráfico: Clientes que respondieron * peso comunicado * 2 (ida y vuelta)
            total_mb_round = len(results) * self.payload_mb_per_client * 2
            self.history_network_mb.append(total_mb_round)

        # 3. Log para consola
        print(f"--- Ronda {server_round} ---")
        print(f"📈 Accuracy: {avg_accuracy:.2%}")
        if total_mb_round > 0:
            print(f"📡 Tráfico Red: {total_mb_round:.2f} MB")
        print("-" * 20)

        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        return loss_aggregated, {"accuracy": avg_accuracy, "network_mb": total_mb_round}


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
        self.kg_manager = KnowledgeGraphManager(self.kb)
        self.expl_generator = ExplanationGenerator(self.kb)

        self.history_sr = []
        self.history_dc = []
        self.history_sd = []

    def aggregate_fit(self, server_round, results, failures):
        valid_results = [r for r in results if not r[1].metrics.get("failed", False)]
        if not valid_results:
            return None, {}

        # 1. Medir carga de red comunicada
        first_params = valid_results[0][1].parameters
        payload_bytes = sum(len(t) for t in first_params.tensors)
        self.payload_mb_per_client = payload_bytes / (1024 ** 2)
        self.payload_params_per_client = payload_bytes // 4

        # 2. Extraer métricas semánticas enviadas por los clientes
        n_samples = np.array([r[1].num_examples for r in valid_results], dtype=np.float32)
        accuracies = [r[1].metrics.get("accuracy", 0.0) for r in valid_results]
        sr_vals = np.array([float(r[1].metrics.get("semantic_relevance", 0.0)) for r in valid_results], dtype=np.float32)
        dc_vals = np.array([float(r[1].metrics.get("domain_consistency", 0.0)) for r in valid_results], dtype=np.float32)
        exp_vectors = [r[1].metrics.get("explanation_vector", []) for r in valid_results]

        # 3. Calcular Diversidad Semántica (SD_i)
        K = len(valid_results)
        sd_vals = np.ones(K, dtype=np.float32)
        has_vectors = len(exp_vectors) == K and len(exp_vectors[0]) > 0

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

        # 4. Fórmula de Ponderación Semántica SemFedXAI
        # w_i^sem = n_i * (1 + alpha * SR_i) * (1 + beta * DC_i) * (gamma ^ SD_i)
        raw_sem_weights = (
            n_samples *
            (1.0 + self.alpha * sr_vals) *
            (1.0 + self.beta * dc_vals) *
            np.power(self.gamma, sd_vals)
        )
        total_w = float(np.sum(raw_sem_weights))
        normalized_weights = raw_sem_weights / (total_w if total_w > 0 else 1.0)

        # 5. Agregación ponderada con los pesos semánticos calculados
        weights_results = [
            (parameters_to_ndarrays(r[1].parameters), float(w))
            for r, w in zip(valid_results, normalized_weights)
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # 6. Actualización del Knowledge Graph Manager global
        if has_vectors:
            self.kg_manager.update_graph(normalized_weights.tolist(), exp_vectors, server_round)

        # 7. Seguimiento del mejor modelo
        if accuracies:
            best_idx = np.argmax(accuracies)
            best_accuracy = accuracies[best_idx]
            if best_accuracy > self.best_accuracy:
                self.best_accuracy = best_accuracy
                self.best_model = parameters_aggregated
                print(f"✅ Nuevo mejor modelo global SemFedXAI (Accuracy: {best_accuracy:.2%})")

        # 8. Registro de métricas históricas
        self.history_sr.append(float(np.mean(sr_vals)))
        self.history_dc.append(float(np.mean(dc_vals)))
        self.history_sd.append(float(np.mean(sd_vals)))

        # 9. Reporte detallado de Ponderación Semántica en consola
        print(f"\n" + "=" * 62)
        print(f"🧠 AGREGACIÓN SEMÁNTICA SemFedXAI — RONDA {server_round}".center(62))
        print("=" * 62)
        print(f"{'Cliente':<8} | {'Muestras':<8} | {'SR_i':<8} | {'DC_i':<8} | {'SD_i':<8} | {'Peso Sem':<10}")
        print("-" * 62)
        for i in range(K):
            print(f"C_{i:<6} | {int(n_samples[i]):<8} | {sr_vals[i]:<8.3f} | {dc_vals[i]:<8.3f} | {sd_vals[i]:<8.3f} | {normalized_weights[i]:<9.2%}")
        print("-" * 62)
        print(f"Promedios: SR={np.mean(sr_vals):.3f} | DC={np.mean(dc_vals):.3f} | SD={np.mean(sd_vals):.3f}")

        # 10. Generación de Explicaciones Clínicas para Médicos/Psicólogos
        if has_vectors:
            mean_exp = np.mean(exp_mat, axis=0).tolist()
            print("\n📋 NARRATIVAS CLÍNICAS EXPLICABLES (Explanation Generator):")
            for c_name in self.class_names[:2]:  # Mostrar las 2 primeras clases activas
                narrative = self.expl_generator.generate_clinical_narrative(mean_exp, c_name, top_k=2)
                print(narrative)
                print()
        print("=" * 62 + "\n")

        return parameters_aggregated, {
            "avg_sr": float(np.mean(sr_vals)),
            "avg_dc": float(np.mean(dc_vals)),
            "avg_sd": float(np.mean(sd_vals)),
        }
