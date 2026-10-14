"""Helper script to download pre-trained quantized MobileNet-SSD COCO TFLite model and labels."""

import os
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABELS_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"

MODEL_PATH = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_coco.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "coco_labels.txt")


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[INFO] File already exists: {dest_path}")
        return

    print(f"[INFO] Downloading {url} -> {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[SUCCESS] Downloaded {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_file(MODEL_URL, MODEL_PATH)
    download_file(LABELS_URL, LABELS_PATH)
    print("\n[READY] All model assets downloaded successfully into models/ directory.")


if __name__ == "__main__":
    main()
