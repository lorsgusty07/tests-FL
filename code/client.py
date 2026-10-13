import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from model import TextDataset, CNNTextClassifier, HybridTransformerClassifier, make_collate_fn


class FlowerClient(fl.client.NumPyClient):
    """
    Cliente federado para modelos de clasificación de texto.
    Soporta modelo híbrido Transformer (MentalBERT + MelBERT + CNN-1D) y baseline CNN.
    """
    def __init__(
        self,
        cid: int,
        vocab: dict,
        classes: list,
        data_dir: str = None,
        simulate_failures: bool = False,
        vocab_size: int = None,
        model_type: str = "hybrid",
        freeze_transformers: bool = True,
        shared_models: dict = None,
        max_seq_len: int = 128,
        hf_token: str = None
    ):
        self.cid = int(cid)
        self.vocab = vocab
        self.classes = classes
        self.data_dir = data_dir
        self.model_type = model_type
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

        if model_type == "hybrid":
            vocab_itos = None
            if hasattr(self.vocab, "get_itos"):
                vocab_itos = self.vocab.get_itos()
            elif hasattr(self.vocab, "itos"):
                vocab_itos = self.vocab.itos

            self.model = HybridTransformerClassifier(
                num_classes=len(self.classes),
                freeze_transformers=freeze_transformers,
                max_seq_len=max_seq_len,
                vocab_itos=vocab_itos,
                pad_idx=self.pad_idx,
                mental_model=shared_models.get("mental_model") if shared_models else None,
                melbert_model=shared_models.get("melbert_model") if shared_models else None,
                mental_tokenizer=shared_models.get("mental_tokenizer") if shared_models else None,
                mel_tokenizer=shared_models.get("mel_tokenizer") if shared_models else None,
                hf_token=hf_token
            ).to(self.device)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=1e-4)
        else:
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
        if hasattr(self.model, "freeze_transformers") and self.model.freeze_transformers:
            return [p.detach().cpu().numpy() for p in self.model.parameters() if p.requires_grad]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        if hasattr(self.model, "freeze_transformers") and self.model.freeze_transformers:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            for p, val in zip(trainable_params, parameters):
                p.data.copy_(torch.from_numpy(val).to(p.device))
        else:
            self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

    def fit(self, parameters, config):
        if self.simulate_failure():  # Simulación de fallo
            return self.get_parameters({}), 0, {"accuracy": 0.0, "failed": True}

        self.set_parameters(parameters)

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
        self.set_parameters(parameters)
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
        """Calcula el peso real de los parámetros comunicados en la red federada."""
        total_params = sum(p.numel() for p in self.model.parameters())
        if hasattr(self.model, "freeze_transformers") and self.model.freeze_transformers:
            active_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            active_params = total_params

        # Peso en MB (4 bytes por parámetro float32)
        mb_size = (active_params * 4) / (1024**2)
        return total_params, active_params, mb_size

    def simulate_failure(self):
        if not self.simulate_failures:
            return False
        # Simulación simple de fallo con 20% de probabilidad
        return np.random.random() < 0.2
