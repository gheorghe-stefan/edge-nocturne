import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Target category groupings for domestic security & wildlife monitoring
CATEGORIES = {
    "person": ["person"],
    "vehicle": ["car", "motorcycle", "bus", "truck", "bicycle", "train"],
    "animal": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)):
    """Letterbox resize image to 640x640 maintaining aspect ratio with border padding."""
    shape = im.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class ObjectDetector:
    """High-accuracy YOLOv8/11 640x640 detector with LiteRT fallback."""

    def __init__(
        self,
        onnx_model_path: str = "models/yolov8n.onnx",
        tflite_fallback_path: str = "models/ssd_mobilenet_v2_coco.tflite",
        min_confidence: float = 0.35,
        target_size: int = 640,
    ):
        self.min_confidence = min_confidence
        self.target_size = target_size
        self.is_yolo = False
        self.net = None
        self.interpreter = None
        self.lock = threading.Lock()

        # Check for YOLO ONNX model (prefer 960x960 for max wildlife/yard resolution, then 640x640)
        if os.path.exists("models/yolov8n_960.onnx"):
            onnx_model_path = "models/yolov8n_960.onnx"
            self.target_size = 960
        elif not os.path.exists(onnx_model_path):
            alt_path = "models/yolo11n.onnx"
            if os.path.exists(alt_path):
                onnx_model_path = alt_path

        if os.path.exists(onnx_model_path):
            try:
                print(f"[INFO] Initializing High-Accuracy YOLO Detector ({onnx_model_path})...")
                self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.is_yolo = True
                print(f"[INFO] YOLO Detector ready! Resolution: {self.target_size}x{self.target_size} Letterbox")
                return
            except Exception as e:
                print(f"[WARN] Failed to load YOLO ONNX ({e}), attempting LiteRT fallback...")

        # Fallback to LiteRT MobileNet SSD
        if os.path.exists(tflite_fallback_path):
            try:
                try:
                    from ai_edge_litert.interpreter import Interpreter
                except ImportError:
                    from tflite_runtime.interpreter import Interpreter
                print(f"[INFO] Loading LiteRT fallback model: {tflite_fallback_path}...")
                self.interpreter = Interpreter(model_path=tflite_fallback_path, num_threads=4)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.input_height = int(self.input_details[0]["shape"][1])
                self.input_width = int(self.input_details[0]["shape"][2])
                self.input_dtype = self.input_details[0]["dtype"]
                print(f"[INFO] LiteRT fallback ready ({self.input_width}x{self.input_height})")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize any detector: {e}")
        else:
            raise FileNotFoundError(f"Neither {onnx_model_path} nor {tflite_fallback_path} found.")

    def _get_category(self, label: str) -> str:
        label_lower = label.lower()
        for cat, names in CATEGORIES.items():
            if label_lower in names:
                return cat
        return "other"

    def detect(self, bgr_frame: np.ndarray, wait: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], float]:
        """Runs inference and returns normalized detections [ymin, xmin, ymax, xmax] and latency ms."""
        acquired = self.lock.acquire(blocking=wait)
        if not acquired:
            return None, 0.0

        try:
            start_time = time.perf_counter()
            if self.is_yolo:
                detections = self._detect_yolo(bgr_frame)
            else:
                detections = self._detect_litert(bgr_frame)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
            return detections, latency_ms
        finally:
            self.lock.release()

    def _detect_yolo(self, bgr_frame: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = bgr_frame.shape[:2]
        lb_img, ratio, (pad_w, pad_h) = letterbox(bgr_frame, (self.target_size, self.target_size))

        blob = cv2.dnn.blobFromImage(lb_img, 1.0 / 255.0, (self.target_size, self.target_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # YOLO output shape: [1, 84, 8400] -> transpose to [8400, 84]
        data = outputs[0].T

        boxes = []
        confidences = []
        class_ids = []

        for row in data:
            classes_scores = row[4:]
            max_idx = int(np.argmax(classes_scores))
            conf = float(classes_scores[max_idx])

            if conf >= self.min_confidence:
                label_candidate = COCO_CLASSES[max_idx] if max_idx < len(COCO_CLASSES) else "unknown"
                cat = self._get_category(label_candidate)
                # Only keep person, vehicle, and animal
                if cat not in ("person", "vehicle", "animal"):
                    continue

                cx, cy, bw, bh = row[0:4]
                # Unpad to original frame pixel coords
                x1 = int((cx - 0.5 * bw - pad_w) / ratio)
                y1 = int((cy - 0.5 * bh - pad_h) / ratio)
                x2 = int((cx + 0.5 * bw - pad_w) / ratio)
                y2 = int((cy + 0.5 * bh - pad_h) / ratio)

                w_box = max(1, x2 - x1)
                h_box = max(1, y2 - y1)

                boxes.append([x1, y1, w_box, h_box])
                confidences.append(conf)
                class_ids.append(max_idx)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, 0.45)
        detections = []

        for idx in indices:
            i = int(idx)
            label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "unknown"
            cat = self._get_category(label)
            bx, by, bw, bh = boxes[i]

            # Convert to normalized [ymin, xmin, ymax, xmax] format for Nocturne pipeline
            ymin = max(0.0, min(1.0, by / orig_h))
            xmin = max(0.0, min(1.0, bx / orig_w))
            ymax = max(0.0, min(1.0, (by + bh) / orig_h))
            xmax = max(0.0, min(1.0, (bx + bw) / orig_w))

            detections.append({
                "label": label,
                "confidence": round(confidences[i], 3),
                "category": cat,
                "box": [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)],
            })

        return detections

    def _detect_litert(self, bgr_frame: np.ndarray) -> List[Dict[str, Any]]:
        resized = cv2.resize(bgr_frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        detections = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score >= self.min_confidence:
                class_id = int(classes[i])
                label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                cat = self._get_category(label)
                if cat not in ("person", "vehicle", "animal"):
                    continue

                detections.append({
                    "label": label,
                    "confidence": round(score, 3),
                    "category": cat,
                    "box": [float(coord) for coord in boxes[i]],
                })
        return detections
