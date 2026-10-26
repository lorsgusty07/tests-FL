"""
Diagnóstico rápido (CPU, segundos) de los datos federados ANTES de entrenar.

Detecta las causas típicas de que la accuracy se quede pegada en la proporción de la
clase mayoritaria (~27%):
  1. vocab.pkl no corresponde a los IDs del .npz  -> el texto reconstruido es basura/vacío.
  2. El .npz de cada cliente está ordenado por clase -> el corte 80/20 por posición separa clases.
  3. Desbalance de clases -> baseline mayoritaria (accuracy de predecir siempre la misma clase).

Uso en Colab:
    !python code/diagnose_data.py --data-dir "/content/drive/My Drive/federated_text_data" --num-clients 12
"""
import os
import sys
import argparse
from collections import Counter

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import load_vocab, resolve_file  # noqa: E402
from model import extract_itos  # noqa: E402


def decode(seq, itos):
    words = [itos[t] for t in seq if 0 <= t < len(itos) and itos[t] not in ("<pad>", "<unk>", None)]
    return " ".join(words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/content/drive/My Drive/federated_text_data")
    p.add_argument("--num-clients", type=int, default=12)
    args = p.parse_args()

    vocab = load_vocab(resolve_file(args.data_dir, "vocab.pkl"))
    itos = extract_itos(vocab) or []
    classes = np.load(resolve_file(args.data_dir, "label_metadata.npz"), allow_pickle=True)["classes"].tolist()

    print("=" * 70)
    print(f"Vocab: tipo={type(vocab).__name__} | len(itos)={len(itos)}")
    print(f"Primeros tokens: {itos[:15]}")
    for sp in ("<pad>", "<unk>"):
        print(f"  índice de {sp}: {itos.index(sp) if sp in itos else 'NO ENCONTRADO'}")
    print(f"Clases ({len(classes)}): {classes}")
    print("=" * 70)

    global_counts = Counter()
    problems = set()
    for cid in range(args.num_clients):
        path = resolve_file(args.data_dir, f"client_{cid}.npz")
        if not os.path.exists(path):
            print(f"[cliente {cid}] no encontrado: {path}")
            continue
        d = np.load(path, allow_pickle=True)
        texts = [np.asarray(s, dtype=np.int64) for s in d["texts"]]
        labels = np.asarray(d["labels"], dtype=np.int64)
        n = len(labels)
        global_counts.update(labels.tolist())

        all_ids = np.concatenate([t for t in texts if len(t)]) if n else np.array([], dtype=np.int64)
        oov = float(np.mean(all_ids >= len(itos))) if all_ids.size else 0.0
        unk_idx = itos.index("<unk>") if "<unk>" in itos else -1
        unk = float(np.mean(all_ids == unk_idx)) if all_ids.size else 0.0
        empty = sum(1 for t in texts[:500] if not decode(t, itos).strip())

        runs = int(np.sum(labels[1:] != labels[:-1])) if n > 1 else 0
        n_cls = len(np.unique(labels))
        split = int(0.8 * n)
        tr_cls, va_cls = set(labels[:split].tolist()), set(labels[split:].tolist())
        val_counts = np.bincount(labels[split:], minlength=len(classes))
        baseline = val_counts.max() / max(val_counts.sum(), 1)

        print(f"[cliente {cid:>2}] n={n:<6} clases={n_cls} | IDs fuera de vocab={oov:.1%} | <unk>={unk:.1%} | "
              f"vacíos(500)={empty} | cambios de etiqueta={runs} | baseline val={baseline:.1%}")
        if runs <= 2 * n_cls:
            print(f"      -> ORDENADO POR CLASE: el corte 80/20 por posición deja train={sorted(tr_cls)} val={sorted(va_cls)}")
            problems.add("sorted")
        if oov > 0.01 or unk > 0.5 or empty > 50:
            problems.add("vocab")

    print("=" * 70)
    total = sum(global_counts.values())
    print("Distribución global de clases:")
    for c, k in sorted(global_counts.items(), key=lambda x: -x[1]):
        print(f"  {classes[c]:<22} {k:>7} ({k / total:.1%})")
    print(f"Baseline mayoritaria global: {max(global_counts.values()) / total:.1%}")

    d0 = np.load(resolve_file(args.data_dir, "client_0.npz"), allow_pickle=True)
    print("\nTextos reconstruidos del cliente 0 (deben leerse como inglés coherente):")
    for i in range(min(3, len(d0["labels"]))):
        print(f"  [{classes[int(d0['labels'][i])]}] {decode(np.asarray(d0['texts'][i], dtype=np.int64), itos)[:200]}")

    print("\n" + "=" * 70)
    if "vocab" in problems:
        print("PROBLEMA: vocab.pkl NO corresponde a los IDs del .npz. MentalBERT/MelBERT reciben texto basura")
        print("          o vacío y el modelo solo puede aprender a predecir la clase mayoritaria.")
    if "sorted" in problems:
        print("PROBLEMA: los .npz están ordenados por clase (ya corregido en client.py con split estratificado).")
    if not problems:
        print("Datos OK: vocab coherente y etiquetas mezcladas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
