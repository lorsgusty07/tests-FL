import os
import sys
import json
import faulthandler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from model import (
    TextDataset,
    CNNTextClassifier,
    HybridTransformerClassifier,
    make_collate_fn,
    extract_itos,
)
from semantic_modules import (
    DSM5KnowledgeBase,
    LocalKnowledgeEnhancer,
    OntologyAlignmentLoss,
    LocalExplainer,
)

# Si el proceso muere por un fallo nativo (SIGSEGV/SIGABRT) imprime el traceback de Python
try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass


def _diag(msg: str):
    """Log de diagnóstico con RAM del proceso y memoria GPU; siempre con flush."""
    parts = [f"[diag pid={os.getpid()}] {msg}"]
    try:
        import psutil
        parts.append(f"RAM_proc={psutil.Process().memory_info().rss / 1024**3:.2f}GB")
        parts.append(f"RAM_libre={psutil.virtual_memory().available / 1024**3:.2f}GB")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            parts.append(f"GPU_alloc={torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        except Exception:
            pass
    print(" | ".join(parts), flush=True)


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
        hf_token: str = None,
        batch_size: int = None,
        use_semfedxai: bool = True,
        knowledge_base: DSM5KnowledgeBase = None,
        lambda_onto: float = 0.1,
        local_epochs: int = 1,
        lr: float = 1e-4
    ):
        self.cid = int(cid)
        self.vocab = vocab
        self.classes = classes
        self.data_dir = data_dir
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = vocab['<pad>'] if '<pad>' in vocab else 0
        self.use_semfedxai = use_semfedxai
        self.local_epochs = int(local_epochs) if local_epochs and local_epochs > 0 else 1
        self.lr = float(lr) if lr and lr > 0 else 1e-4

        if batch_size is not None and batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = 8 if model_type == "hybrid" else 32

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
            vocab_itos = extract_itos(self.vocab)
            _diag(f"cliente {self.cid}: datos listos, vocab_itos={'None' if vocab_itos is None else len(vocab_itos)}; cargando backbones...")

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
            )
            _diag(f"cliente {self.cid}: backbones cargados en CPU; moviendo a {self.device}...")
            self.model = self.model.to(self.device)
            _diag(f"cliente {self.cid}: modelo en {self.device}")

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)
        else:
            self.model = CNNTextClassifier(
                vocab_size=self.vocab_size,
                num_classes=len(self.classes),
                pad_idx=self.pad_idx
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.criterion = nn.CrossEntropyLoss()
        self.simulate_failures = simulate_failures

        # Inicialización de componentes semánticos locales (LKE, L_onto y LE)
        self.kb = knowledge_base
        self.lambda_onto = float(lambda_onto)
        self.enhancer = None
        self.onto_loss_fn = None
        self.explainer = None
        try:
            if self.kb is None and self.data_dir:
                self.kb = DSM5KnowledgeBase(base_dir=self.data_dir)
            if self.kb is not None:
                self.enhancer = LocalKnowledgeEnhancer(self.kb, self.classes, self.device)
                self.explainer = LocalExplainer(self.kb, self.enhancer, self.classes)
                if self.use_semfedxai:
                    self.onto_loss_fn = OntologyAlignmentLoss(self.enhancer)
                    _diag(f"cliente {self.cid}: módulos SemFedXAI (LKE, L_onto, LE) activos.")
                else:
                    _diag(f"cliente {self.cid}: LocalExplainer activo para evaluación comparativa (FedAvg).")
        except Exception as e:
            _diag(f"cliente {self.cid}: módulos semánticos omitidos ({e})")

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
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn
            ),
            DataLoader(
                TextDataset(texts[split_idx:], labels[split_idx:]),
                batch_size=self.batch_size,
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
        _diag(f"cliente {self.cid}: fit() inicio ({len(self.trainloader.dataset)} muestras, batch={self.batch_size}, epochs={self.local_epochs})")

        max_batches = config.get("max_batches") if config else None
        total_samples = 0
        correct = 0
        done = 0
        self.model.train()
        for epoch in range(self.local_epochs):
            for step, (texts, labels) in enumerate(self.trainloader):
                texts, labels = texts.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                if self.model_type == "hybrid" and self.use_semfedxai and self.onto_loss_fn is not None:
                    outputs, o_fused, mental_sent = self.model(texts, return_features=True)
                    loss_ce = self.criterion(outputs, labels)
                    loss_onto = self.onto_loss_fn(o_fused, labels, mental_sent)
                    loss = loss_ce + self.lambda_onto * loss_onto
                else:
                    outputs = self.model(texts)
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
                if epoch == 0 and step == 0:
                    _diag(f"cliente {self.cid}: primer batch OK (loss={loss.item():.4f})")
                done += 1
                if max_batches is not None and done >= int(max_batches):
                    break
            if max_batches is not None and done >= int(max_batches):
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generación de Explicaciones Semánticas Locales (SemFedXAI y FedAvg)
        sem_metrics = {}
        if self.explainer is not None and self.model_type == "hybrid":
            self.model.eval()
            all_z = []
            all_logits = []
            all_sent = []
            all_labels = []
            count = 0
            with torch.no_grad():
                for texts, labels in self.valloader:
                    texts, labels = texts.to(self.device), labels.to(self.device)
                    out, z, sent = self.model(texts, return_features=True)
                    all_z.append(z)
                    all_logits.append(out)
                    all_sent.append(sent)
                    all_labels.append(labels)
                    count += len(labels)
                    if count >= 64:
                        break
            if all_z:
                cat_z = torch.cat(all_z, dim=0)
                cat_logits = torch.cat(all_logits, dim=0)
                cat_sent = torch.cat(all_sent, dim=0)
                cat_labels = torch.cat(all_labels, dim=0)
                sem_metrics = self.explainer.explain(
                    cat_z,
                    cat_logits,
                    cat_sent,
                    self.model.classify_fused,
                    labels=cat_labels
                )

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        sr_val = float(sem_metrics.get("semantic_relevance", 0.0))
        dc_val = float(sem_metrics.get("domain_consistency", 0.0))
        eq_val = float(sem_metrics.get("explanation_quality", 0.0))
        _diag(f"cliente {self.cid}: fit() fin (acc={accuracy:.2%}, EQ={eq_val:.3f}, SR={sr_val:.3f}, DC={dc_val:.3f})")

        # Flower solo acepta en `metrics` valores escalares (int | float | str | bytes | bool).
        # Todo array/lista se serializa como JSON string para que Flower/Ray no rechace los resultados de fit.
        fit_metrics = {
            "accuracy": float(accuracy),
            "failed": False,
            "semantic_relevance": sr_val,
            "domain_consistency": dc_val,
            "explanation_quality": eq_val,
        }
        exp_vec = sem_metrics.get("explanation_vector")
        if exp_vec is not None and len(exp_vec) > 0:
            fit_metrics["explanation_vector_json"] = json.dumps([float(v) for v in exp_vec])

        class_exp = sem_metrics.get("class_explanations")
        if class_exp is not None:
            fit_metrics["class_explanations_json"] = json.dumps(class_exp.tolist())

        class_cnt = sem_metrics.get("class_counts")
        if class_cnt is not None:
            fit_metrics["class_counts_json"] = json.dumps([int(c) for c in class_cnt])

        return self.get_parameters({}), len(self.trainloader.dataset), fit_metrics

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
