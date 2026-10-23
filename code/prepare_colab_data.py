import os
import zipfile


def mount_and_extract_drive(
    zip_path: str = "/content/drive/My Drive/federated_text_data/prueba01.zip",
    dest_folder: str = "/content/drive/My Drive/federated_text_data/"
):
    """
    Monta Google Drive (si está en Google Colab) y descomprime el dataset federado.
    """
    try:
        from google.colab import drive
        print("[INFO] Montando Google Drive en /content/drive...")
        drive.mount('/content/drive')
    except ImportError:
        print("[INFO] Entorno local detectado. Omitiendo drive.mount().")

    if not os.path.exists(zip_path):
        print(f"[WARN] No se encontró el archivo ZIP en: {zip_path}")
        print("Verifica la ruta en Google Drive o sube prueba01.zip a dicha ubicación.")
        return

    os.makedirs(dest_folder, exist_ok=True)
    print(f"[INFO] Descomprimiendo {zip_path} en {dest_folder}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    print("[INFO] Contenido descomprimido:")
    files = os.listdir(dest_folder)
    for f in sorted(files)[:15]:
        print(f" - {f}")
    if len(files) > 15:
        print(f" ... y {len(files) - 15} archivos más.")


if __name__ == "__main__":
    mount_and_extract_drive()
