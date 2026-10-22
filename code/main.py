import os

# Evita que transformers cargue TensorFlow/JAX (instalados en Colab). Deben fijarse ANTES de
# importar torch/transformers/ray para que los workers de Ray hereden estas variables.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import flwr as fl

# Permitir importaciones locales si se corre directamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from client import FlowerClient
from strategy import CustomStrategy, SemFedXAIStrategy
from semantic_modules import DSM5KnowledgeBase


def resolve_file(base_dir: str, filename: str) -> str:
    """Busca un archivo en base_dir o en rutas comunes."""
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    candidates.extend([
        os.path.join(".", "dataset", "federated_text_data", filename),
        os.path.join(".", "dataset", filename),
        os.path.join("..", "dataset", "federated_text_data", filename),
        os.path.join("/content/drive/My Drive/federated_text_data", filename),
        filename
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


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
        for a in args:
            self._extract(a)

    def _extract(self, obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            # Lista homogénea de tokens donde los elementos son strings
            if obj and all(isinstance(x, str) for x in obj[:100]) and all(isinstance(x, str) for x in obj[-10:]):
                clean_tokens = [str(x) for x in obj if isinstance(x, str)]
                if clean_tokens:
                    self.itos = clean_tokens
                    self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
                    return
            for item in obj:
                self._extract(item)
        elif isinstance(obj, dict):
            if "itos" in obj and isinstance(obj["itos"], (list, tuple)):
                self._extract(obj["itos"])
                return
            if "_itos" in obj and isinstance(obj["_itos"], (list, tuple)):
                self._extract(obj["_itos"])
                return
            if "stoi" in obj and isinstance(obj["stoi"], dict):
                self._extract(obj["stoi"])
                return
            if "_stoi" in obj and isinstance(obj["_stoi"], dict):
                self._extract(obj["_stoi"])
                return

            # Pares reales token (str) -> índice (int estricto, nunca booleans ni dicts)
            token_map = {str(k): v for k, v in obj.items() if isinstance(k, str) and type(v) is int and not k.startswith("_")}
            if token_map:
                self.stoi.update(token_map)
                max_idx = max(self.stoi.values())
                if len(self.itos) <= max_idx:
                    self.itos = ["<unk>"] * (max_idx + 1)
                    for tok, idx in self.stoi.items():
                        if 0 <= idx <= max_idx:
                            self.itos[idx] = tok
            else:
                for k, v in obj.items():
                    if k not in ["_modules", "_parameters", "_buffers", "training"]:
                        self._extract(v)
        elif hasattr(obj, "__dict__"):
            self._extract(obj.__dict__)

    def __setstate__(self, state):
        self.stoi = {}
        self.itos = []
        if isinstance(state, dict):
            self._extract(state)
        elif isinstance(state, (list, tuple)):
            self._extract(state)

        if not self.stoi and self.itos:
            self.stoi = {tok: idx for idx, tok in enumerate(self.itos) if isinstance(tok, str)}
        elif self.stoi and not self.itos:
            valid_stoi = {str(k): v for k, v in self.stoi.items() if isinstance(k, str) and type(v) is int}
            if valid_stoi:
                max_idx = max(valid_stoi.values())
                self.itos = ["<unk>"] * (max_idx + 1)
                for tok, idx in valid_stoi.items():
                    if 0 <= idx <= max_idx:
                        self.itos[idx] = tok

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>", 0))

    def __len__(self):
        return len(self.itos) if self.itos else len(self.stoi)

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


def count_backbone_params(hf_token: str = None) -> int:
    """Cuenta los parámetros de MentalBERT + MelBERT en el dispositivo 'meta' (sin reservar memoria)."""
    try:
        from transformers import AutoConfig, AutoModel
        total = 0
        for name in ("mental/mental-bert-base-uncased", "EhsanAghazadeh/melbert-roberta"):
            cfg = AutoConfig.from_pretrained(name, token=hf_token)
            with torch.device("meta"):
                m = AutoModel.from_config(cfg)
            total += sum(p.numel() for p in m.parameters())
        return total
    except Exception as e:
        print(f"⚠️ No se pudo contar los parámetros de los backbones: {e}")
        return 0


def run_sequential(build_client, strategy, num_clients: int, num_rounds: int):
    """
    FedAvg en un único proceso (sin Ray): cada cliente se entrena uno tras otro y los
    pesos se promedian ponderando por número de muestras (igual que FedAvg de Flower).
    Cualquier error aparece con su traceback real en lugar de 'ActorPool is empty'.
    """
    import gc
    from flwr.server.strategy.aggregate import aggregate

    global_params = build_client(0).get_parameters({})

    for rnd in range(1, num_rounds + 1):
        print(f"\n[ROUND {rnd}]", flush=True)

        fit_results, fit_accs, payload_bytes = [], [], 0
        for cid in range(num_clients):
            client = build_client(cid)
            new_params, n_samples, metrics = client.fit(global_params, {})
            if metrics.get("failed", False) or n_samples == 0:
                print(f"⚠️ Cliente {cid} reportó fallo; se omite en la agregación.", flush=True)
            else:
                fit_results.append((new_params, n_samples, metrics))
                fit_accs.append(metrics["accuracy"])
                payload_bytes = sum(p.nbytes for p in new_params)
            del client
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not fit_results:
            raise RuntimeError("Ningún cliente devolvió resultados válidos en esta ronda.")

        if isinstance(strategy, SemFedXAIStrategy):
            n_samples = np.array([res[1] for res in fit_results], dtype=np.float32)
            sr_vals = np.array([float(res[2].get("semantic_relevance", 0.0)) for res in fit_results], dtype=np.float32)
            dc_vals = np.array([float(res[2].get("domain_consistency", 0.0)) for res in fit_results], dtype=np.float32)
            exp_vectors = [res[2].get("explanation_vector", []) for res in fit_results]

            K = len(fit_results)
            sd_vals = np.ones(K, dtype=np.float32)
            has_vectors = K > 0 and all(len(v) > 0 for v in exp_vectors) and len({len(v) for v in exp_vectors}) == 1
            exp_mat = np.array(exp_vectors, dtype=np.float32) if has_vectors else None
            if has_vectors and K > 1:
                for i in range(K):
                    other_centroid = np.mean(np.delete(exp_mat, i, axis=0), axis=0)
                    norm_i = np.linalg.norm(exp_mat[i])
                    norm_c = np.linalg.norm(other_centroid)
                    if norm_i > 1e-6 and norm_c > 1e-6:
                        sim = float(np.dot(exp_mat[i], other_centroid) / (norm_i * norm_c))
                        sd_vals[i] = float(np.clip(1.0 - sim, 0.0, 1.0))
                    else:
                        sd_vals[i] = 0.5

            raw_w = (
                n_samples *
                (1.0 + strategy.alpha * sr_vals) *
                (1.0 + strategy.beta * dc_vals) *
                np.power(strategy.gamma, sd_vals)
            )
            total_raw = float(np.sum(raw_w))
            sem_weights = raw_w / (total_raw if total_raw > 0 else 1.0)
            weighted_fit = [(res[0], float(w)) for res, w in zip(fit_results, sem_weights)]
            global_params = aggregate(weighted_fit)

            client_class_exps = []
            client_class_counts = []
            for res in fit_results:
                raw_ce = res[2].get("class_explanations_json")
                if raw_ce:
                    try:
                        client_class_exps.append(json.loads(raw_ce))
                    except Exception:
                        client_class_exps.append([])
                else:
                    client_class_exps.append([])
                client_class_counts.append(res[2].get("class_counts", []))

            if has_vectors and any(len(e) > 0 for e in client_class_exps):
                strategy.kg_manager.update_graph(
                    sem_weights.tolist(),
                    client_class_exps,
                    client_class_counts,
                    rnd
                )
                kg_file = os.path.join(strategy.data_dir or ".", "global_knowledge_graph_latest.ttl")
                try:
                    strategy.kg_manager.export_turtle(kg_file)
                except Exception as e:
                    print(f"⚠️ Error al exportar grafo Turtle: {e}")

            print(f"\n" + "=" * 62)
            print(f"🧠 AGREGACIÓN SEMÁNTICA SemFedXAI — RONDA {rnd}".center(62))
            print("=" * 62)
            print(f"{'Cliente':<8} | {'Muestras':<8} | {'SR_i':<8} | {'DC_i':<8} | {'SD_i':<8} | {'Peso Sem':<10}")
            print("-" * 62)
            for i in range(K):
                print(f"C_{i:<6} | {int(n_samples[i]):<8} | {sr_vals[i]:<8.3f} | {dc_vals[i]:<8.3f} | {sd_vals[i]:<8.3f} | {sem_weights[i]:<9.2%}")
            print("-" * 62)
            print(f"Promedios: SR={np.mean(sr_vals):.3f} | DC={np.mean(dc_vals):.3f} | SD={np.mean(sd_vals):.3f}")

            if has_vectors and strategy.kg_manager.last_round > 0:
                print("\n📋 NARRATIVAS CLÍNICAS EXPLICABLES (Explanation Generator):")
                for c_idx in range(min(2, len(strategy.class_names))):
                    narrative = strategy.expl_generator.global_class_narrative(strategy.kg_manager, c_idx, top_k=2)
                    print(narrative)
                    print()
            print("=" * 62 + "\n")
        else:
            global_params = aggregate([(res[0], res[1]) for res in fit_results])

        strategy.payload_mb_per_client = payload_bytes / (1024 ** 2)
        strategy.payload_params_per_client = payload_bytes // 4
        best_acc = max(fit_accs)
        if best_acc > strategy.best_accuracy:
            strategy.best_accuracy = best_acc
            print(f"✅ Nuevo mejor modelo (Accuracy: {best_acc:.2%})", flush=True)

        eval_accs, eval_losses = [], []
        for cid in range(num_clients):
            client = build_client(cid)
            loss, _, metrics = client.evaluate(global_params, {})
            eval_accs.append(metrics["accuracy"])
            eval_losses.append(loss)
            del client
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_acc = float(np.mean(eval_accs))
        total_mb_round = len(fit_results) * strategy.payload_mb_per_client * 2
        strategy.history_accuracy.append(avg_acc)
        strategy.history_network_mb.append(total_mb_round)

        print(f"--- Ronda {rnd} ---")
        print(f"📈 Accuracy: {avg_acc:.2%}  (loss media: {float(np.mean(eval_losses)):.4f})")
        print(f"📡 Tráfico Red: {total_mb_round:.2f} MB")
        print("-" * 20, flush=True)

    return global_params


def main(
    data_dir: str = None,
    num_clients: int = None,
    num_rounds: int = None,
    simulate_failures: bool = False,
    output_csv: str = None,
    output_plot: str = None,
    mock_data: bool = False,
    strategy_name: str = None,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    lambda_onto: float = None
):
    default_data_dir = (
        "/content/drive/My Drive/federated_text_data"
        if os.path.exists("/content/drive/My Drive/federated_text_data")
        else "./dataset/federated_text_data"
    )
    parser = argparse.ArgumentParser(description="Ejecución de Aprendizaje Federado (PyTorch + Flower)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Directorio con datos federados")
    parser.add_argument("--num-clients", type=int, default=5, help="Número de clientes federados")
    parser.add_argument("--num-rounds", type=int, default=10, help="Número de rondas de comunicación")
    parser.add_argument("--strategy", type=str, choices=["semfedxai", "fedavg"], default="semfedxai", help="Estrategia federada ('semfedxai' o 'fedavg')")
    parser.add_argument("--alpha", type=float, default=0.5, help="SemFedXAI: Ponderación de Relevancia Semántica (SR)")
    parser.add_argument("--beta", type=float, default=0.5, help="SemFedXAI: Ponderación de Consistencia Ontológica (DC)")
    parser.add_argument("--gamma", type=float, default=1.2, help="SemFedXAI: Base exponencial para Diversidad Semántica (SD)")
    parser.add_argument("--lambda-onto", type=float, default=0.1, help="SemFedXAI: Peso de la pérdida ontológica auxiliar L_onto")
    parser.add_argument("--model-type", type=str, choices=["hybrid", "cnn"], default="hybrid", help="Arquitectura del modelo ('hybrid' o 'cnn')")
    parser.add_argument("--no-freeze-transformers", action="store_true", help="No congelar las capas base de Transformers (por defecto están congeladas para eficiencia federada)")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Longitud máxima de secuencia para los Transformers")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch por cliente (por defecto 8 para hybrid, 32 para cnn)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", None), help="Token de autenticación de Hugging Face para modelos gated")
    parser.add_argument("--simulate-failures", action="store_true", help="Habilitar simulación de fallos aleatorios")
    parser.add_argument("--output-csv", type=str, default="tabla_precision_por_ronda.csv", help="Ruta para guardar tabla de precisión CSV")
    parser.add_argument("--output-plot", type=str, default="evolucion_rendimiento.png", help="Ruta para guardar gráfica de evolución")
    parser.add_argument("--mock-data", action="store_true", help="Generar datos sintéticos para probar la ejecución sin archivos reales")
    parser.add_argument("--no-ray", action="store_true", help="Ejecutar FedAvg secuencial en un solo proceso, sin Ray (muestra el error real si algo falla y evita que el worker muera por memoria)")
    parser.add_argument("--skip-preflight", action="store_true", help="Omitir la prueba previa de 1 cliente en un subproceso aislado")
    parser.add_argument("--preflight-child", action="store_true", help=argparse.SUPPRESS)
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
    if strategy_name is not None:
        args.strategy = strategy_name
    if alpha is not None:
        args.alpha = alpha
    if beta is not None:
        args.beta = beta
    if gamma is not None:
        args.gamma = gamma
    if lambda_onto is not None:
        args.lambda_onto = lambda_onto

    print("==================================================")
    print(" INICIANDO APRENDIZAJE FEDERADO ")
    if args.strategy == "semfedxai":
        print(f" ESTRATEGIA: SemFedXAI (α={args.alpha}, β={args.beta}, γ={args.gamma}, λ_onto={args.lambda_onto})")
    else:
        print(" ESTRATEGIA: FedAvg Convencional (Baseline)")
    if args.model_type == "hybrid":
        print(" MODELO HÍBRIDO (MentalBERT + MelBERT + CNN-1D)")
    else:
        print(" MODELO BASELINE DENSO (CNN)")
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
        classes = ["Depresion", "Ansiedad", "BPD", "PTSD"]
        
        np.savez(metadata_path, classes=np.array(classes))
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
            
        for cid in range(args.num_clients):
            n_samples = 40
            dummy_texts = [np.random.randint(1, 500, size=np.random.randint(10, 30)) for _ in range(n_samples)]
            dummy_labels = np.random.randint(0, len(classes), size=n_samples)
            np.savez(os.path.join(args.data_dir, f"client_{cid}.npz"), texts=np.array(dummy_texts, dtype=object), labels=dummy_labels)
        print("✅ Datos sintéticos de prueba generados exitosamente.\n")

    # 1. Cargar metadatos y vocabulario (Celda 9)
    metadata_path = resolve_file(args.data_dir, "label_metadata.npz")
    vocab_path = resolve_file(args.data_dir, "vocab.pkl")

    if not os.path.exists(metadata_path) or not os.path.exists(vocab_path):
        print(f"❌ Error: Archivos no encontrados en {args.data_dir}.")
        print(f"   Metadata buscado en: {metadata_path}")
        print(f"   Vocab buscado en: {vocab_path}")
        sys.exit(1)

    print(f"📂 Cargando metadatos desde: {metadata_path}")
    label_metadata = np.load(metadata_path, allow_pickle=True)
    vocab = load_vocab(vocab_path)

    classes = label_metadata["classes"].tolist()

    # 2. Calcular vocab_size global seguro verificando tokens en los clientes
    global_vocab_size = len(vocab)
    for cid in range(args.num_clients):
        c_path = resolve_file(args.data_dir, f"client_{cid}.npz")
        if os.path.exists(c_path):
            data_c = np.load(c_path, allow_pickle=True)
            if "texts" in data_c:
                for seq in data_c["texts"]:
                    if len(seq) > 0:
                        m = int(np.max(seq))
                        if m >= global_vocab_size:
                            global_vocab_size = m + 1

    print(f"✅ Vocabulario cargado ({len(vocab)} tokens). Clases: {len(classes)} ({classes})")
    print(f"📊 Modelo seleccionado: {args.model_type.upper()}")

    # Pre-descargar modelos base si se usa la arquitectura híbrida
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if args.model_type == "hybrid":
        if hf_token:
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token, add_to_git_credential=False)
                print("🔑 Autenticación con Hugging Face exitosa.")
            except Exception:
                pass

        try:
            import transformers  # noqa: F401  (solo verifica que esté instalado)
        except ImportError:
            import subprocess
            print("📦 transformers no detectado. Instalando automáticamente...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"])

        from huggingface_hub import snapshot_download
        print("⏳ Descargando / verificando archivos de MentalBERT y MelBERT (sin cargarlos en RAM del proceso principal)...")
        mental_name = "mental/mental-bert-base-uncased"
        mel_name = "EhsanAghazadeh/melbert-roberta"
        snapshot_download(mental_name, token=hf_token)
        snapshot_download(mel_name, token=hf_token)
        print("✅ Archivos de MentalBERT y MelBERT disponibles en la caché local de Hugging Face.")

    # Cargar base de conocimiento ontológica para SemFedXAI
    kb = None
    if args.strategy == "semfedxai":
        try:
            kb = DSM5KnowledgeBase(base_dir=args.data_dir)
            print(f"📚 Base de Conocimiento DSM-5 cargada: {len(kb.criteria)} criterios clínicos.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar la base ontológica DSM-5 ({e}). Se usará FedAvg estándar.")
            args.strategy = "fedavg"

    # 3. Función generadora de clientes (Celda 12)
    def client_fn(cid: str):
        client = FlowerClient(
            cid=int(cid),
            vocab=vocab,
            classes=classes,
            data_dir=args.data_dir,
            simulate_failures=args.simulate_failures,
            vocab_size=global_vocab_size,
            model_type=args.model_type,
            freeze_transformers=not args.no_freeze_transformers,
            shared_models=None,
            max_seq_len=args.max_seq_len,
            hf_token=hf_token,
            batch_size=args.batch_size,
            use_semfedxai=(args.strategy == "semfedxai"),
            knowledge_base=kb,
            lambda_onto=args.lambda_onto
        )
        return client.to_client() if hasattr(client, "to_client") else client

    # 3. Configuración de Estrategia y Recursos (Celda 12)
    if args.strategy == "semfedxai" and kb is not None:
        strategy = SemFedXAIStrategy(
            client_fn=client_fn,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            knowledge_base=kb,
            class_names=classes,
            data_dir=args.data_dir
        )
    else:
        strategy = CustomStrategy(client_fn=client_fn)

    client_resources = {
        "num_cpus": 1,
        "num_gpus": 1.0 if torch.cuda.is_available() else 0.0
    }

    def build_numpy_client(cid: int):
        return FlowerClient(
            cid=int(cid),
            vocab=vocab,
            classes=classes,
            data_dir=args.data_dir,
            simulate_failures=args.simulate_failures,
            vocab_size=global_vocab_size,
            model_type=args.model_type,
            freeze_transformers=not args.no_freeze_transformers,
            shared_models=None,
            max_seq_len=args.max_seq_len,
            hf_token=hf_token,
            batch_size=args.batch_size,
            use_semfedxai=(args.strategy == "semfedxai"),
            knowledge_base=kb,
            lambda_onto=args.lambda_onto
        )

    # Modo hijo del preflight: ejecuta 1 cliente real (2 batches) y termina
    if args.preflight_child:
        print("🧪 [preflight] construyendo cliente 0 y ejecutando 2 batches...", flush=True)
        c = build_numpy_client(0)
        p = c.get_parameters({})
        c.fit(p, {"max_batches": 2})
        c.evaluate(p, {})
        print("✅ PREFLIGHT OK: el pipeline completo funciona en un proceso aislado.", flush=True)
        return

    # Preflight: si el proceso muere aquí, se ve la causa real (en vez de la cascada de Ray)
    if (not args.no_ray and not args.skip_preflight and args.model_type == "hybrid"
            and os.path.basename(sys.argv[0]) == "main.py"):
        import subprocess
        print("\n🧪 Preflight: probando 1 cliente en un subproceso aislado antes de lanzar Ray...", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--preflight-child"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            causas = {
                -9: "SIGKILL: el sistema mató el proceso por FALTA DE RAM (OOM). Reduce --batch-size/--max-seq-len o usa un runtime con más memoria.",
                137: "SIGKILL: el sistema mató el proceso por FALTA DE RAM (OOM). Reduce --batch-size/--max-seq-len o usa un runtime con más memoria.",
                -11: "SIGSEGV: fallo nativo (segfault) en una librería (torch/transformers/tokenizers/CUDA).",
                139: "SIGSEGV: fallo nativo (segfault) en una librería (torch/transformers/tokenizers/CUDA).",
                -6: "SIGABRT: abort nativo (suele ser CUDA/driver o una aserción de librería).",
                134: "SIGABRT: abort nativo (suele ser CUDA/driver o una aserción de librería).",
            }
            print("\n" + "=" * 60)
            print(f"❌ PREFLIGHT FALLÓ (código de salida {rc}).")
            print(f"   Causa probable: {causas.get(rc, 'excepción de Python; revisa el traceback impreso arriba.')}")
            print("   Se aborta antes de lanzar Ray para no inundar la consola con 'ActorPool is empty'.")
            print("   Pega las líneas '[diag ...]' y el traceback de arriba para localizar el punto exacto.")
            print("=" * 60)
            sys.exit(1)
        print("✅ Preflight correcto. Lanzando la simulación con Ray...\n", flush=True)

    if args.no_ray:
        print(f"\n🚀 Iniciando FedAvg SECUENCIAL (sin Ray) con {args.num_clients} clientes por {args.num_rounds} rondas...")
        run_sequential(build_numpy_client, strategy, args.num_clients, args.num_rounds)
    else:
        print(f"\n🚀 Iniciando simulación con {args.num_clients} clientes por {args.num_rounds} rondas...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={
                "include_dashboard": False,
                "ignore_reinit_error": True,
                # Por defecto Ray reserva ~30% de la RAM; los parámetros comunicados pesan ~1.5 MB
                "object_store_memory": 512 * 1024 * 1024,
                "runtime_env": {"env_vars": {
                    "USE_TF": "0", "USE_FLAX": "0", "USE_JAX": "0",
                    "TRANSFORMERS_NO_TF": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
                    "TOKENIZERS_PARALLELISM": "false",
                }},
            },
        )

        # Si ningún cliente logró completar un fit, los workers de Ray murieron: se reintenta sin Ray
        if strategy.payload_params_per_client == 0:
            print("\n" + "=" * 60)
            print("⚠️ Ray no completó ningún entrenamiento (los workers murieron).")
            print("   Reintentando automáticamente con FedAvg secuencial SIN Ray...")
            print("=" * 60, flush=True)
            try:
                import ray
                ray.shutdown()
            except Exception:
                pass
            strategy.history_accuracy.clear()
            strategy.history_network_mb.clear()
            strategy.best_accuracy = 0.0
            run_sequential(build_numpy_client, strategy, args.num_clients, args.num_rounds)

    # 4. Reporte Técnico Real (Celda 12)
    print("\n" + "=" * 50)
    print("📡 MÉTRICAS DE RED: APRENDIZAJE FEDERADO".center(50))
    print("=" * 50)

    # Todo se calcula con el payload real medido en aggregate_fit; no se instancia ningún modelo en el driver
    active_p = strategy.payload_params_per_client
    mb_round = strategy.payload_mb_per_client
    total_p = active_p
    if args.model_type == "hybrid" and not args.no_freeze_transformers:
        total_p = active_p + count_backbone_params(hf_token)
    total_transmision_mb = sum(strategy.history_network_mb)
    active_pct = (100.0 * active_p / total_p) if total_p else 0.0

    print("1. Estructura del Modelo:")
    print(f"   - Parámetros Totales: {total_p:,}")
    print(f"   - Parámetros Comunicados/Entrenables: {active_p:,} ({active_pct:.2f}%)")

    print("\n2. Carga de Comunicación (Overhead):")
    print(f"   - Payload por cliente (Subida): {mb_round:.2f} MB")
    print(f"   - Tráfico total en la simulación: {total_transmision_mb:.2f} MB")

    print("\n3. Estado para el Workshop:")
    print("   - Los backbones MentalBERT/MelBERT están congelados y NO se transmiten.")
    print("   - Cada comunicación transmite solo las CNN-1D + capas densas.")
    print("=" * 50)

    # 5. Visualización de Resultados (Celda 13)
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
        print(f"\n📊 Gráfica de evolución guardada en: {args.output_plot}")

    # 6. Exportación de métricas a CSV (Celda 14)
    if strategy.history_accuracy:
        df_precision = pd.DataFrame({
            "Ronda de Comunicación": list(range(1, len(strategy.history_accuracy) + 1)),
            "Precisión": strategy.history_accuracy
        })
        df_precision.to_csv(args.output_csv, index=False)
        print(f"📄 Tabla de precisión exportada a: {args.output_csv}")
        print("\nResumen de precisión por ronda:")
        print(df_precision.to_string(index=False))


if __name__ == "__main__":
    main()
