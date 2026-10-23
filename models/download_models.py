"""Helper script to download pre-trained ML models:
1. Google LiteRT Quantized MobileNet-SSD COCO detector
2. OpenCV YuNet deep face detector with 5-point landmark alignment
3. OpenCV SFace deep face recognition embedding model
"""

import os
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABELS_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YOLOV8_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"

MODEL_PATH = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_coco.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "coco_labels.txt")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
YOLOV8_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[INFO] File already exists: {dest_path}")
        return

    print(f"[INFO] Downloading {url} -> {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[SUCCESS] Downloaded {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_file(YOLOV8_URL, YOLOV8_PATH)
    download_file(MODEL_URL, MODEL_PATH)
    download_file(LABELS_URL, LABELS_PATH)
    download_file(YUNET_URL, YUNET_PATH)
    download_file(SFACE_URL, SFACE_PATH)
    print("\n[READY] All model assets downloaded successfully into models/ directory.")


if __name__ == "__main__":
    main()
