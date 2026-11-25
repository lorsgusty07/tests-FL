import os
import sys
import json
import time
import faulthandler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from model import (
    TextDataset,
    HybridTokenizedDataset,
    CNNTextClassifier,
    HybridTransformerClassifier,
    make_collate_fn,
    make_hybrid_collate_fn,
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


# Caché a nivel de proceso de los .npz y de las secuencias ya tokenizadas de cada cliente.
# Flower crea un FlowerClient nuevo en cada fit/evaluate; con esto:
#   - el .npz se lee desde Google Drive 1 sola vez por proceso, y
#   - ids_to_texts + mental_tokenizer + mel_tokenizer se ejecutan 1 sola vez por proceso/cliente
#     en lugar de repetirse en cada batch de las N rondas.
_PROCESS_DATA_CACHE = {}
_PROCESS_TOKENIZED_CACHE = {}


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
        lr: float = 1e-4,
        use_amp: bool = False
    ):
        self.cid = int(cid)
        self.use_amp = bool(use_amp)
        self.vocab = vocab
        self.classes = classes
        self.data_dir = data_dir
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = vocab['<pad>'] if '<pad>' in vocab else 0
        self.max_seq_len = int(max_seq_len)
        self.use_semfedxai = use_semfedxai
        self.local_epochs = int(local_epochs) if local_epochs and local_epochs > 0 else 1
        self.lr = float(lr) if lr and lr > 0 else 1e-4

        if batch_size is not None and batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = 8 if model_type == "hybrid" else 32

        self.collate_fn = make_collate_fn(self.pad_idx)
        raw_texts, labels, train_idx, val_idx, cache_key = self._prepare_splits()

        # Determinar el tamaño de vocabulario necesario para evitar IndexError
        if vocab_size is not None and vocab_size > 0:
            self.vocab_size = vocab_size
        else:
            max_tok = 0
            for i in train_idx:
                seq = raw_texts[i]
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
                max_seq_len=self.max_seq_len,
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
            self.model.use_amp = self.use_amp
            if self.use_amp and self.device.type == "cuda":
                # TF32 en los productos de matrices restantes (GPUs Ampere o más nuevas)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            _diag(f"cliente {self.cid}: modelo en {self.device} (amp={'on' if self.use_amp else 'off'})")

            self.trainloader, self.valloader = self._build_hybrid_loaders(
                raw_texts, labels, train_idx, val_idx, cache_key
            )

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)
        else:
            self.trainloader, self.valloader = self._build_raw_loaders(
                raw_texts, labels, train_idx, val_idx
            )
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

    def _prepare_splits(self):
        file_path = self._resolve_client_data_path()
        cache_key = os.path.abspath(file_path)
        if cache_key not in _PROCESS_DATA_CACHE:
            data = np.load(file_path, allow_pickle=True)
            # Conversión robusta
            _PROCESS_DATA_CACHE[cache_key] = (
                [torch.tensor(seq.astype(np.int64)) for seq in data['texts']],
                torch.tensor(data['labels'].astype(np.int64)),
            )
        texts, labels = _PROCESS_DATA_CACHE[cache_key]

        # División 80-20 ESTRATIFICADA y barajada con semilla fija.
        # Antes se cortaba por posición (texts[:80%]); si el .npz está agrupado por clase,
        # entrenamiento y validación terminaban con clases distintas.
        labels_np = labels.numpy()
        rng = np.random.RandomState(42 + self.cid)
        train_idx, val_idx = [], []
        for c in np.unique(labels_np):
            ci = rng.permutation(np.where(labels_np == c)[0])
            k = len(ci) if len(ci) == 1 else int(round(0.8 * len(ci)))
            train_idx.extend(ci[:k].tolist())
            val_idx.extend(ci[k:].tolist())
        train_idx = rng.permutation(train_idx).tolist()
        val_idx = rng.permutation(val_idx).tolist() if val_idx else []

        self.label_order_runs = int(np.sum(labels_np[1:] != labels_np[:-1])) if len(labels_np) > 1 else 0
        self.train_label_counts = np.bincount(labels_np[train_idx], minlength=len(self.classes))
        self.val_label_counts = np.bincount(labels_np[val_idx], minlength=len(self.classes)) if val_idx else np.zeros(len(self.classes), dtype=int)
        return texts, labels, train_idx, val_idx, cache_key

    def _build_raw_loaders(self, texts, labels, train_idx, val_idx):
        return (
            DataLoader(
                TextDataset([texts[i] for i in train_idx], labels[train_idx]),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn
            ),
            DataLoader(
                TextDataset([texts[i] for i in val_idx], labels[val_idx]),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
        )

    def _build_hybrid_loaders(self, raw_texts, labels, train_idx, val_idx, cache_key):
        """
        Pre-tokeniza todas las muestras del cliente 1 sola vez por proceso y las guarda en RAM.
        En rondas siguientes omite ids_to_texts y ambos tokenizadores en CPU.
        """
        itos = getattr(self.model, "vocab_itos", None)
        if not itos:
            self._diag_data_sanity_raw([raw_texts[i] for i in train_idx], labels[train_idx])
            return self._build_raw_loaders(raw_texts, labels, train_idx, val_idx)

        tok_key = (cache_key, self.max_seq_len)
        if tok_key not in _PROCESS_TOKENIZED_CACHE:
            self._diag_data_sanity_raw([raw_texts[i] for i in train_idx], labels[train_idx])
            try:
                t_tok = time.perf_counter()
                all_strs = self.model.ids_to_texts(raw_texts)
                m_ids_all, r_ids_all = [], []
                for start in range(0, len(all_strs), 1024):
                    chunk = all_strs[start:start + 1024]
                    m_enc = self.model.mental_tokenizer(
                        chunk, padding=False, truncation=True, max_length=self.max_seq_len
                    )["input_ids"]
                    r_enc = self.model.mel_tokenizer(
                        chunk, padding=False, truncation=True, max_length=self.max_seq_len
                    )["input_ids"]
                    m_ids_all.extend(torch.tensor(s, dtype=torch.int32) for s in m_enc)
                    r_ids_all.extend(torch.tensor(s, dtype=torch.int32) for s in r_enc)
                _PROCESS_TOKENIZED_CACHE[tok_key] = (m_ids_all, r_ids_all)
                _diag(
                    f"cliente {self.cid}: {len(all_strs)} textos pre-tokenizados en caché RAM "
                    f"en {time.perf_counter() - t_tok:.2f}s"
                )
            except Exception as e:
                _diag(f"cliente {self.cid}: pre-tokenización no disponible ({e}); usando modo dinámico")
                return self._build_raw_loaders(raw_texts, labels, train_idx, val_idx)

        m_ids_all, r_ids_all = _PROCESS_TOKENIZED_CACHE[tok_key]
        m_pad = 0 if getattr(self.model.mental_tokenizer, "pad_token_id", None) is None else int(self.model.mental_tokenizer.pad_token_id)
        r_pad = 1 if getattr(self.model.mel_tokenizer, "pad_token_id", None) is None else int(self.model.mel_tokenizer.pad_token_id)
        hybrid_collate = make_hybrid_collate_fn(m_pad, r_pad)

        return (
            DataLoader(
                HybridTokenizedDataset(
                    [m_ids_all[i] for i in train_idx],
                    [r_ids_all[i] for i in train_idx],
                    labels[train_idx],
                ),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=hybrid_collate,
            ),
            DataLoader(
                HybridTokenizedDataset(
                    [m_ids_all[i] for i in val_idx],
                    [r_ids_all[i] for i in val_idx],
                    labels[val_idx],
                ),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=hybrid_collate,
            ),
        )

    def load_data(self):
        texts, labels, train_idx, val_idx, _ = self._prepare_splits()
        return self._build_raw_loaders(texts, labels, train_idx, val_idx)

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
        t_start = time.perf_counter()
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
                    if count >= 128:
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
        eq_chance = float(sem_metrics.get("explanation_chance", 0.0))
        eq_pred = float(sem_metrics.get("explanation_quality_pred", 0.0))
        _diag(f"cliente {self.cid}: fit() fin en {time.perf_counter() - t_start:.1f}s (acc={accuracy:.2%}, EQ={eq_val:.3f} [azar={eq_chance:.3f}, EQ_pred={eq_pred:.3f}], SR={sr_val:.3f}, DC={dc_val:.3f})")

        # Flower solo acepta en `metrics` valores escalares (int | float | str | bytes | bool).
        # Todo array/lista se serializa como JSON string para que Flower/Ray no rechace los resultados de fit.
        fit_metrics = {
            "accuracy": float(accuracy),
            "failed": False,
            "semantic_relevance": sr_val,
            "domain_consistency": dc_val,
            "explanation_quality": eq_val,
            "explanation_chance": eq_chance,
            "explanation_quality_pred": eq_pred,
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
        t_start = time.perf_counter()

        correct, total, loss = 0, 0, 0.0
        pred_counts = torch.zeros(len(self.classes), dtype=torch.long)
        n_cls = len(self.classes)
        confusion = torch.zeros(n_cls, n_cls, dtype=torch.long)   # filas = clase real, columnas = predicha
        with torch.no_grad():
            for texts, labels in self.valloader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                loss += self.criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                pred_counts += torch.bincount(preds.cpu(), minlength=n_cls)
                confusion += torch.bincount(
                    labels.cpu() * n_cls + preds.cpu(), minlength=n_cls * n_cls
                ).reshape(n_cls, n_cls)

        # Diagnóstico de colapso: si el modelo predice casi siempre una sola clase,
        # su accuracy coincide con la proporción de la clase mayoritaria (baseline).
        if total > 0:
            top = int(pred_counts.argmax())
            baseline = float(self.val_label_counts.max()) / max(int(self.val_label_counts.sum()), 1)
            _diag(
                f"cliente {self.cid}: eval en {time.perf_counter() - t_start:.1f}s acc={correct / total:.2%} | baseline_mayoritaria={baseline:.2%} | "
                f"clase_predicha_dominante={self.classes[top]} ({pred_counts[top].item() / total:.0%} de las predicciones)"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0.0
        return float(loss / total), total, {
            "accuracy": accuracy,
            "confusion_json": json.dumps(confusion.tolist()),
        }

    def _diag_data_sanity_raw(self, sample_texts, sample_labels):
        """Verifica que los IDs del .npz se decodifican a texto real antes de llegar a MentalBERT/MelBERT."""
        try:
            n_check = min(200, len(sample_texts))
            itos = getattr(self.model, "vocab_itos", None)
            if not itos or n_check == 0:
                _diag(f"cliente {self.cid}: [SANITY] sin vocab_itos: los IDs NO se decodifican a texto (entrada inválida para BERT)")
                return
            texts = self.model.ids_to_texts(sample_texts[:n_check])
            n_empty = sum(1 for t in texts if t == "empty")
            all_ids = torch.cat(sample_texts[:n_check])
            oov = float((all_ids >= len(itos)).float().mean()) if all_ids.numel() else 0.0
            names = [str(c) for c in self.classes]
            tr = dict(zip(names, self.train_label_counts.tolist()))
            va = dict(zip(names, self.val_label_counts.tolist()))
            _diag(
                f"cliente {self.cid}: [SANITY] textos vacíos={n_empty}/{n_check} | IDs fuera de vocab={oov:.1%} | "
                f"cambios de etiqueta en el .npz={self.label_order_runs}"
            )
            _diag(f"cliente {self.cid}: [SANITY] train={tr} | val={va}")
            _diag(f"cliente {self.cid}: [SANITY] ejemplo (label={names[int(sample_labels[0])]}): '{texts[0][:160]}'")
        except Exception as e:
            _diag(f"cliente {self.cid}: [SANITY] no se pudo verificar ({e})")

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
