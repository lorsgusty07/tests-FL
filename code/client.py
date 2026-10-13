import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from model import TextDataset, CNNTextClassifier, make_collate_fn


class FlowerClient(fl.client.NumPyClient):
    """
    Cliente federado basado en la celda 11 del notebook Aprendizaje_Federado.ipynb.
    Maneja el entrenamiento y evaluación local con CNNTextClassifier.
    """
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

    def _resolve_client_data_path(self):
        """Busca el archivo client_{cid}.npz en la ruta configurada o rutas de fallback."""
        possible_paths = []
        if self.data_dir:
            possible_paths.append(os.path.join(self.data_dir, f"client_{self.cid}.npz"))
        possible_paths.extend([
            f"./dataset/federated_text_data/client_{self.cid}.npz",
            f"../dataset/federated_text_data/client_{self.cid}.npz",
            f"/content/drive/My Drive/federated_text_data/client_{self.cid}.npz",
            f"client_{self.cid}.npz"
        ])
        for p in possible_paths:
            if os.path.exists(p):
                return p
        # Si no se encuentra, devuelve la primera ruta esperada para lanzar el FileNotFoundError
        return possible_paths[0]

    def load_data(self):
        file_path = self._resolve_client_data_path()
        data = np.load(file_path, allow_pickle=True)

        # Conversión robusta
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
        if self.simulate_failure():  # Simulación de fallo
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
        """Calcula el peso real de los parámetros para el modelo denso."""
        # Contamos todos los parámetros del modelo
        total_params = sum(p.numel() for p in self.model.parameters())

        # En el modelo denso, el 100% de los parámetros son "activos"
        active_params = total_params

        # Peso en MB (4 bytes por parámetro float32)
        mb_size = (active_params * 4) / (1024**2)
        return total_params, active_params, mb_size

    def simulate_failure(self):
        if not self.simulate_failures:
            return False
        # Simulación simple de fallo con 20% de probabilidad
        return np.random.random() < 0.2
