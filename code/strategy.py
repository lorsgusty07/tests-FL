import numpy as np
import flwr as fl


class CustomStrategy(fl.server.strategy.FedAvg):
    """
    Estrategia personalizada basada en la celda 12 del notebook Aprendizaje_Federado.ipynb.
    - Filtra clientes que reportaron fallo.
    - Rastrea el mejor modelo y su precisión.
    - Calcula la métrica de tráfico de red real para el modelo denso (baseline).
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

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_metrics(self, metrics):
        """Registra el promedio de precisión de los clientes."""
        accuracies = [m["accuracy"] for _, m in metrics if "accuracy" in m]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        return {"accuracy": avg_accuracy}

    def aggregate_evaluate(self, server_round, results, failures):
        """Calcula Accuracy y Tráfico de Red Real para el modelo denso."""
        # 1. Accuracy promedio
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        self.history_accuracy.append(avg_accuracy)

        # 2. CÁLCULO DE RED REAL
        # Se usa el peso real medido en aggregate_fit (no se instancia ningún modelo en el driver)
        total_mb_round = 0.0
        if self.payload_mb_per_client > 0:
            # Tráfico: Clientes que respondieron * peso comunicado * 2 (ida y vuelta)
            total_mb_round = len(results) * self.payload_mb_per_client * 2
            self.history_network_mb.append(total_mb_round)

        # 3. Log para consola
        print(f"--- Ronda {server_round} ---")
        print(f"📈 Accuracy: {avg_accuracy:.2%}")
        if total_mb_round > 0:
            print(f"📡 Tráfico Red: {total_mb_round:.2f} MB (Modelo DENSO)")
        print("-" * 20)

        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        return loss_aggregated, {"accuracy": avg_accuracy, "network_mb": total_mb_round}
