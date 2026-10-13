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

    def aggregate_fit(self, server_round, results, failures):
        # Filtrar clientes fallidos
        valid_results = [r for r in results if not r[1].metrics.get("failed", False)]
        accuracies = [res.metrics["accuracy"] for _, res in valid_results if "accuracy" in res.metrics]

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

        # 2. CÁLCULO DE RED REAL (DENSO)
        # Obtenemos un cliente de prueba para medir el peso total del modelo
        total_mb_round = 0.0
        if self.client_fn is not None:
            test_client = self.client_fn("0")
            # En caso de que client_fn devuelva un wrapper de Flower o el NumPyClient
            underlying_client = getattr(test_client, "numpy_client", test_client)
            if hasattr(underlying_client, "get_dense_stats"):
                _, _, mb_per_client = underlying_client.get_dense_stats()
                # Tráfico: Clientes que respondieron * peso total * 2 (ida y vuelta)
                total_mb_round = len(results) * mb_per_client * 2
                self.history_network_mb.append(total_mb_round)

        # 3. Log para consola
        print(f"--- Ronda {server_round} ---")
        print(f"📈 Accuracy: {avg_accuracy:.2%}")
        if total_mb_round > 0:
            print(f"📡 Tráfico Red: {total_mb_round:.2f} MB (Modelo DENSO)")
        print("-" * 20)

        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        return loss_aggregated, {"accuracy": avg_accuracy, "network_mb": total_mb_round}
