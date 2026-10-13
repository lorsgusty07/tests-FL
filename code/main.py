import os
import sys
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
from strategy import CustomStrategy


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


def load_vocab(vocab_path: str):
    """Carga el vocabulario original usando torchtext tal como se hacía en el notebook."""
    try:
        import torchtext
    except ModuleNotFoundError:
        import subprocess
        print("📦 torchtext no detectado. Instalando automáticamente para deserializar el vocabulario...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchtext"])
        import torchtext

    with open(vocab_path, "rb") as f:
        return pickle.load(f)


def main(
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
    parser = argparse.ArgumentParser(description="Ejecución de Aprendizaje Federado (PyTorch + Flower)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Directorio con datos federados")
    parser.add_argument("--num-clients", type=int, default=5, help="Número de clientes federados")
    parser.add_argument("--num-rounds", type=int, default=10, help="Número de rondas de comunicación")
    parser.add_argument("--simulate-failures", action="store_true", help="Habilitar simulación de fallos aleatorios")
    parser.add_argument("--output-csv", type=str, default="tabla_precision_por_ronda.csv", help="Ruta para guardar tabla de precisión CSV")
    parser.add_argument("--output-plot", type=str, default="evolucion_rendimiento.png", help="Ruta para guardar gráfica de evolución")
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
    print(f"📊 Tamaño efectivo del vocabulario para la CNN: {global_vocab_size}")

    # 3. Función generadora de clientes (Celda 12)
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

    # 3. Configuración de Estrategia y Recursos (Celda 12)
    strategy = CustomStrategy(client_fn=client_fn)
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.5 if torch.cuda.is_available() else 0.0
    }

    print(f"\n🚀 Iniciando simulación con {args.num_clients} clientes por {args.num_rounds} rondas...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources
    )

    # 4. Reporte Técnico Real (Celda 12)
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

    print("\n3. Estado para el Workshop:")
    print("   - Este es el baseline denso (Sparsity: 0%)")
    print("   - Cada comunicación transmite el modelo completo.")
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
