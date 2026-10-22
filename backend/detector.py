import os
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Dynamic import: LiteRT (ai-edge-litert) with fallbacks
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# Target category groupings for clean reporting (relevant domestic security classes)
CATEGORIES = {
    "person": ["person"],
    "vehicle": ["car", "motorcycle", "bus", "truck", "bicycle"],
    "animal": ["cat", "dog", "bird"],
}


class ObjectDetector:
    """TensorFlow Lite / Google LiteRT object detection engine."""

    def __init__(
        self,
        model_path: str = "models/ssd_mobilenet_v2_coco.tflite",
        labels_path: str = "models/coco_labels.txt",
        min_confidence: float = 0.50,
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.min_confidence = min_confidence

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run 'python3 models/download_models.py' first."
            )

        # Load labels
        self.labels = self._load_labels(self.labels_path)

        # Initialize LiteRT Interpreter
        print(f"[INFO] Loading LiteRT model: {self.model_path} with {num_threads} threads...")
        self.interpreter = Interpreter(model_path=self.model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        # Get input & output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input tensor properties
        self.input_shape = self.input_details[0]["shape"]
        self.input_height = int(self.input_shape[1])
        self.input_width = int(self.input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]

        print(
            f"[INFO] LiteRT Detector ready! Expected input: {self.input_width}x{self.input_height} ({self.input_dtype})"
        )

    def _load_labels(self, path: str) -> Dict[int, str]:
        labels = {}
        if not os.path.exists(path):
            return labels
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                elif len(parts) == 1:
                    labels[len(labels)] = parts[0]
        return labels

    def _get_category(self, label: str) -> str:
        for cat, names in CATEGORIES.items():
            if label.lower() in names:
                return cat
        return "other"

    def detect(self, bgr_frame: np.ndarray) -> Tuple[List[dict], float]:
        """Runs inference on a full frame and returns filtered detections and latency in ms."""
        start_time = time.perf_counter()

        # 1. Resize and convert color space to RGB
        resized = cv2.resize(bgr_frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 2. Add batch dimension [1, H, W, C] and cast to uint8 / float32 as required
        if self.input_dtype == np.uint8:
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            # Normalized float32
            input_data = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0

        # 3. Set tensor and invoke interpreter
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # 4. Extract output tensors
        # Typical postprocessed SSD outputs:
        # [0] boxes: [1, N, 4] normalized [ymin, xmin, ymax, xmax]
        # [1] classes: [1, N]
        # [2] scores: [1, N]
        # [3] count: [1]
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        detections = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score >= self.min_confidence:
                class_id = int(classes[i])
                label = self.labels.get(class_id, f"class_{class_id}")
                box = [float(coord) for coord in boxes[i]]  # [ymin, xmin, ymax, xmax]

                detections.append({
                    "label": label,
                    "confidence": round(score, 3),
                    "category": self._get_category(label),
                    "box": box,
                })

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
        return detections, latency_ms
