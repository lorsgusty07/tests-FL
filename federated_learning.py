#!/usr/bin/env python3
"""
Script de Aprendizaje Federado extraído y depurado desde Aprendizaje_Federado.ipynb.
Contiene exclusivamente las celdas funcionales para el entrenamiento federado (celdas 9 a 14),
omitiendo celdas experimentales descartadas, comandos de Colab y código obsoleto.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import flwr as fl


# =====================================================================
# 1. Dataset y Arquitectura del Modelo (Celda 10)
# =====================================================================
class TextDataset(Dataset):
    """Clase Dataset compartida por todos los clientes federados."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate_fn(pad_idx: int):
    """Función de collation con padding dinámico."""
    def collate_fn(batch):
        texts, labs = zip(*batch)
        padded = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        return padded.to(torch.long), torch.tensor(labs, dtype=torch.long)
    return collate_fn


class CNNTextClassifier(nn.Module):
    """
    Arquitectura CNN de clasificación de texto:
    Embedding -> Capas Convolucionales paralelas (kernel sizes 3, 4, 5) -> Max Pooling -> FC
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        num_classes: int = 2,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        dropout: float = 0.5,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, emb_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)
        conved = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))


# =====================================================================
# 2. Cliente Federado Flower (Celda 11)
# =====================================================================
class FlowerClient(fl.client.NumPyClient):
    """Cliente federado Flower para entrenamiento local y cálculo de tráfico de red."""
    def __init__(self, cid: int, vocab: dict, classes: list, data_dir: str = None, simulate_failures: bool = False, vocab_size: int = None):
        self.cid = int(cid)
        self.vocab = vocab
        self.classes = classes
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = vocab['<pad>'] if '<pad>' in vocab else 0

        self.collate_fn = make_collate_fn(self.pad_idx)
        self.trainloader, self.valloader = self.load_data()

        # Determinar el tamaño de vocabulario necesario para evitar IndexError
        if vocab_size is not None and vocab_size > 0:
            self.vocab_size = vocab_size
        else:
            max_tok = 0
            for seq, _ in self.trainloader.dataset:
                if len(seq) > 0:
                    m = int(seq.max())
                    if m > max_tok:
                        max_tok = m
            self.vocab_size = max(len(self.vocab), max_tok + 1)

        self.model = CNNTextClassifier(
            vocab_size=self.vocab_size,
            num_classes=len(self.classes),
            pad_idx=self.pad_idx
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.simulate_failures = simulate_failures

    def _resolve_data_path(self):
        candidates = []
        if self.data_dir:
            candidates.append(os.path.join(self.data_dir, f"client_{self.cid}.npz"))
        candidates.extend([
            f"./dataset/federated_text_data/client_{self.cid}.npz",
            f"../dataset/federated_text_data/client_{self.cid}.npz",
            f"/content/drive/My Drive/federated_text_data/client_{self.cid}.npz",
            f"client_{self.cid}.npz"
        ])
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    def load_data(self):
        file_path = self._resolve_data_path()
        data = np.load(file_path, allow_pickle=True)

        # Conversión robusta de tensores
        texts = [torch.tensor(seq.astype(np.int64)) for seq in data['texts']]
        labels = torch.tensor(data['labels'].astype(np.int64))

        # División 80-20
        split_idx = int(0.8 * len(texts))
        return (
            DataLoader(
                TextDataset(texts[:split_idx], labels[:split_idx]),
                batch_size=32,
                shuffle=True,
                collate_fn=self.collate_fn
            ),
            DataLoader(
                TextDataset(texts[split_idx:], labels[split_idx:]),
                batch_size=64,
                shuffle=False,
                collate_fn=self.collate_fn
            )
        )

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        if self.simulate_failure():
            return self.get_parameters({}), 0, {"accuracy": 0.0, "failed": True}

        self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

        total_samples = 0
        correct = 0
        self.model.train()
        for _ in range(3):
            for texts, labels in self.trainloader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return self.get_parameters({}), len(self.trainloader.dataset), {"accuracy": accuracy, "failed": False}

    def evaluate(self, parameters, config):
        self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.eval()

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for texts, labels in self.valloader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                loss += self.criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return float(loss / total), total, {"accuracy": accuracy}

    def get_dense_stats(self):
        """Calcula el peso real de los parámetros para el modelo denso baseline."""
        total_params = sum(p.numel() for p in self.model.parameters())
        active_params = total_params
        mb_size = (active_params * 4) / (1024**2)
        return total_params, active_params, mb_size

    def simulate_failure(self):
        if not self.simulate_failures:
            return False
        return np.random.random() < 0.2


# =====================================================================
# 3. Estrategia del Servidor Personalizada (Celda 12)
# =====================================================================
class CustomStrategy(fl.server.strategy.FedAvg):
    """Estrategia personalizada que calcula métricas y sobrecarga de comunicación."""
    def __init__(self, client_fn=None, **kwargs):
        super().__init__(evaluate_metrics_aggregation_fn=self.aggregate_metrics, **kwargs)
        self.client_fn = client_fn
        self.best_accuracy = 0.0
        self.best_model = None
        self.history_accuracy = []
        self.history_network_mb = []

    def aggregate_fit(self, server_round, results, failures):
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
        accuracies = [m["accuracy"] for _, m in metrics if "accuracy" in m]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        return {"accuracy": avg_accuracy}

    def aggregate_evaluate(self, server_round, results, failures):
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        self.history_accuracy.append(avg_accuracy)

        total_mb_round = 0.0
        if self.client_fn is not None:
            test_client = self.client_fn("0")
            underlying_client = getattr(test_client, "numpy_client", test_client)
            if hasattr(underlying_client, "get_dense_stats"):
                _, _, mb_per_client = underlying_client.get_dense_stats()
                total_mb_round = len(results) * mb_per_client * 2
                self.history_network_mb.append(total_mb_round)

        print(f"--- Ronda {server_round} ---")
        print(f"📈 Accuracy: {avg_accuracy:.2%}")
        if total_mb_round > 0:
            print(f"📡 Tráfico Red: {total_mb_round:.2f} MB (Modelo DENSO)")
        print("-" * 20)

        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        return loss_aggregated, {"accuracy": avg_accuracy, "network_mb": total_mb_round}


class LegacyVocab:
    """Estructura de vocabulario retrocompatible para deserializar vocabularios de torchtext legacy."""
    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        inst.stoi = {}
        inst.itos = []
        return inst

    def __init__(self, *args, **kwargs):
        self.stoi = {}
        self.itos = []
        if args:
            self._extract(args)

    def _extract(self, obj):
        if isinstance(obj, dict):
            if "stoi" in obj:
                self.stoi.update(obj["stoi"])
            elif "_stoi" in obj:
                self.stoi.update(obj["_stoi"])
            else:
                self.stoi.update(obj)
            if "itos" in obj:
                self.itos = list(obj["itos"])
            elif "_itos" in obj:
                self.itos = list(obj["_itos"])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._extract(item)
        elif hasattr(obj, "__dict__"):
            self._extract(obj.__dict__)

    def __setstate__(self, state):
        self.stoi = {}
        self.itos = []
        if isinstance(state, dict):
            self.__dict__.update(state)
            self._extract(state)
        elif isinstance(state, (list, tuple)):
            self._extract(state)
        if not self.stoi and self.itos:
            self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>", 0))

    def __len__(self):
        return len(self.stoi) if self.stoi else len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos


class RobustUnpickler(pickle.Unpickler):
    """Unpickler tolerante que intercepta referencias legacy a torchtext.vocab.vocab."""
    def find_class(self, module, name):
        if "torchtext" in module or "vocab" in module.lower():
            return LegacyVocab
        return super().find_class(module, name)


def load_vocab(vocab_path: str):
    """Carga el vocabulario con soporte nativo y fallback robusto para cualquier versión de torchtext."""
    with open(vocab_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return RobustUnpickler(f).load()


# =====================================================================
# 4. Orquestación y Ejecución de la Simulación (Celdas 9, 12, 13, 14)
# =====================================================================
def run_federated_learning(
    data_dir: str = None,
    num_clients: int = None,
    num_rounds: int = None,
    simulate_failures: bool = False,
    output_csv: str = None,
    output_plot: str = None,
    mock_data: bool = False
):
    default_data_dir = (
        "/content/drive/My Drive/federated_text_data"
        if os.path.exists("/content/drive/My Drive/federated_text_data")
        else "./dataset/federated_text_data"
    )
    parser = argparse.ArgumentParser(description="Simulación de Aprendizaje Federado")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Ruta a los datos federados")
    parser.add_argument("--num-clients", type=int, default=5, help="Número de clientes")
    parser.add_argument("--num-rounds", type=int, default=10, help="Rondas de federación")
    parser.add_argument("--simulate-failures", action="store_true", help="Simular clientes con fallo")
    parser.add_argument("--output-csv", type=str, default="tabla_precision_por_ronda.csv", help="Archivo CSV de salida")
    parser.add_argument("--output-plot", type=str, default="evolucion_rendimiento.png", help="Archivo de gráfica PNG")
    parser.add_argument("--mock-data", action="store_true", help="Generar datos sintéticos para probar la ejecución sin archivos reales")
    args, _ = parser.parse_known_args()

    # Si se invocó como función Python con parámetros, sobrescribir:
    if data_dir is not None:
        args.data_dir = data_dir
    if num_clients is not None:
        args.num_clients = num_clients
    if num_rounds is not None:
        args.num_rounds = num_rounds
    if simulate_failures:
        args.simulate_failures = True
    if output_csv is not None:
        args.output_csv = output_csv
    if output_plot is not None:
        args.output_plot = output_plot
    if mock_data:
        args.mock_data = True

    print("==================================================")
    print(" INICIANDO APRENDIZAJE FEDERADO (BASELINE DENSO) ")
    print("==================================================")

    # Si se solicita --mock-data, generar datos sintéticos temporales
    if args.mock_data:
        os.makedirs(args.data_dir, exist_ok=True)
        metadata_path = os.path.join(args.data_dir, "label_metadata.npz")
        vocab_path = os.path.join(args.data_dir, "vocab.pkl")
        print(f"🔧 Generando datos sintéticos de prueba en: {args.data_dir}...")
        
        vocab = {"<pad>": 0, "<unk>": 1}
        for w in range(2, 500):
            vocab[f"token_{w}"] = w
        classes = ["clase_0", "clase_1"]
        
        np.savez(metadata_path, classes=np.array(classes))
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
            
        for cid in range(args.num_clients):
            n_samples = 40
            dummy_texts = [np.random.randint(1, 500, size=np.random.randint(10, 30)) for _ in range(n_samples)]
            dummy_labels = np.random.randint(0, 2, size=n_samples)
            np.savez(os.path.join(args.data_dir, f"client_{cid}.npz"), texts=np.array(dummy_texts, dtype=object), labels=dummy_labels)
        print("✅ Datos sintéticos de prueba generados exitosamente.\n")

    # Resolución de archivos de metadatos y vocabulario
    def find_file(name):
        candidates = [
            os.path.join(args.data_dir, name),
            os.path.join(".", "dataset", "federated_text_data", name),
            os.path.join(".", "dataset", name),
            os.path.join("/content/drive/My Drive/federated_text_data", name),
            name
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    metadata_path = find_file("label_metadata.npz")
    vocab_path = find_file("vocab.pkl")

    if not os.path.exists(metadata_path) or not os.path.exists(vocab_path):
        print(f"❌ Error: Archivos no encontrados.")
        print(f"   Metadata buscado en: {metadata_path}")
        print(f"   Vocabulario buscado en: {vocab_path}")
        print("Asegúrate de indicar la ruta correcta con --data-dir.")
        sys.exit(1)

    print(f"📂 Cargando metadatos: {metadata_path}")
    label_metadata = np.load(metadata_path, allow_pickle=True)
    vocab = load_vocab(vocab_path)

    classes = label_metadata["classes"].tolist()

    # Calcular vocab_size global seguro verificando tokens en los clientes
    global_vocab_size = len(vocab)
    for cid in range(args.num_clients):
        c_path = find_file(f"client_{cid}.npz")
        if os.path.exists(c_path):
            data_c = np.load(c_path, allow_pickle=True)
            if "texts" in data_c:
                for seq in data_c["texts"]:
                    if len(seq) > 0:
                        m = int(np.max(seq))
                        if m >= global_vocab_size:
                            global_vocab_size = m + 1

    print(f"✅ Vocabulario cargado ({len(vocab)} tokens). Clases detectadas: {len(classes)}")
    print(f"📊 Tamaño efectivo del vocabulario para la CNN: {global_vocab_size}")

    # Función generadora de cliente
    def client_fn(cid: str):
        client = FlowerClient(
            cid=int(cid),
            vocab=vocab,
            classes=classes,
            data_dir=args.data_dir,
            simulate_failures=args.simulate_failures,
            vocab_size=global_vocab_size
        )
        return client.to_client() if hasattr(client, "to_client") else client

    strategy = CustomStrategy(client_fn=client_fn)
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.5 if torch.cuda.is_available() else 0.0
    }

    # Iniciar simulación Flower
    print(f"\n🚀 Lanzando simulación ({args.num_clients} clientes, {args.num_rounds} rondas)...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources
    )

    # Reporte técnico (Celda 12)
    print("\n" + "=" * 50)
    print("📡 MÉTRICAS DE RED: MODELO DENSO (BASELINE)".center(50))
    print("=" * 50)

    test_client = FlowerClient(0, vocab=vocab, classes=classes, data_dir=args.data_dir, vocab_size=global_vocab_size)
    total_p, active_p, mb_round = test_client.get_dense_stats()
    total_transmision_mb = sum(strategy.history_network_mb)

    print("1. Estructura del Modelo:")
    print(f"   - Parámetros Totales: {total_p:,}")
    print(f"   - Parámetros Activos: {active_p:,} (100.0%)")

    print("\n2. Carga de Comunicación (Overhead):")
    print(f"   - Payload por cliente (Subida): {mb_round:.2f} MB")
    print(f"   - Tráfico total en la simulación: {total_transmision_mb:.2f} MB")

    print("\n3. Estado:")
    print("   - Este es el baseline denso (Sparsity: 0%)")
    print("   - Cada comunicación transmite el modelo completo.")
    print("=" * 50)

    # Generar y guardar gráfica (Celda 13)
    if strategy.history_accuracy:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(strategy.history_accuracy) + 1), strategy.history_accuracy, 'b-o', label='Precisión Promedio')
        plt.xlabel("Ronda de Comunicación")
        plt.ylabel("Precisión")
        plt.title("Evolución del Rendimiento")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"\n📊 Gráfica guardada en: {args.output_plot}")

    # Guardar CSV de precisión por ronda (Celda 14)
    if strategy.history_accuracy:
        df_precision = pd.DataFrame({
            "Ronda de Comunicación": list(range(1, len(strategy.history_accuracy) + 1)),
            "Precisión": strategy.history_accuracy
        })
        df_precision.to_csv(args.output_csv, index=False)
        print(f"📄 Archivo CSV guardado en: {args.output_csv}")
        print("\nResultados por ronda:")
        print(df_precision.to_string(index=False))


if __name__ == "__main__":
    run_federated_learning()
