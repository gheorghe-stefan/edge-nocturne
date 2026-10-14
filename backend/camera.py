import cv2
import threading
import time
from typing import List, Optional, Tuple
from backend.detector import ObjectDetector
from backend.recorder import VideoRecorder


class VideoCamera:
    """Threaded camera capture + asynchronous background LiteRT ML + dual-mode recording."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        enable_ml: bool = True,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_ml = enable_ml

        # Initialize camera with V4L2 backend and MJPG hardware compression
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.grabbed, self.frame = self.cap.read()
        self.frame_lock = threading.Lock()
        self.ml_lock = threading.Lock()
        self.stopped = False

        self.latest_detections: List[dict] = []
        self.inference_ms: float = 0.0

        if not self.grabbed:
            print(f"[WARN] Failed to read initial frame from camera device {self.device_id}")

        # Initialize Dual-mode Video Recorder (Event pre-roll + continuous DVR loop)
        self.recorder = VideoRecorder(width=self.width, height=self.height, fps=self.fps)

        # Thread 1: High-speed ingestion loop (30 FPS, zero lag)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Thread 2: Asynchronous ML inference worker loop
        if self.enable_ml:
            try:
                self.detector = ObjectDetector()
                self.ml_thread = threading.Thread(target=self._inference_loop, daemon=True)
                self.ml_thread.start()
            except Exception as e:
                print(f"[WARN] ObjectDetector initialization skipped: {e}")
                self.detector = None

    def _capture_loop(self):
        """Dedicated thread to constantly drain camera buffer at full frame rate."""
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.frame_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                # Feed frame into recorder pre-roll ring buffer and continuous DVR chunk
                self.recorder.add_frame(frame)
            else:
                time.sleep(0.01)

    def _inference_loop(self):
        """Asynchronous ML worker running inference on the newest frame."""
        while not self.stopped:
            if not self.detector:
                time.sleep(0.5)
                continue

            success, frame = self.get_latest_frame()
            if success and frame is not None:
                try:
                    detections, latency = self.detector.detect(frame)
                    with self.ml_lock:
                        self.latest_detections = detections
                        self.inference_ms = latency

                    # Trigger event recording if person, vehicle, or animal is detected
                    target_detections = [
                        d for d in detections
                        if d.get("category") in ("person", "vehicle", "animal")
                    ]
                    if target_detections:
                        self.recorder.trigger_event(target_detections)

                except Exception as e:
                    print(f"[ERROR] Inference error: {e}")

            # Pace inference (~15–20 runs per second to balance CPU/thermal load)
            time.sleep(0.05)

    def get_latest_frame(self) -> Tuple[bool, Optional[any]]:
        """Returns the most recent raw OpenCV BGR frame (clean, unadorned)."""
        with self.frame_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get_latest_detections(self) -> Tuple[List[dict], float]:
        """Returns current detections and inference latency in ms."""
        with self.ml_lock:
            return list(self.latest_detections), self.inference_ms

    def get_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Encodes the clean raw frame as JPEG bytes for HTTP streaming."""
        success, frame = self.get_latest_frame()
        if not success or frame is None:
            return None

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            return None
        return buffer.tobytes()

    def set_continuous_recording(self, enabled: bool):
        self.recorder.set_continuous_recording(enabled)

    def is_continuous_recording(self) -> bool:
        return self.recorder.continuous_enabled

    def stop(self):
        self.stopped = True
        self.recorder.stop()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, "ml_thread") and self.ml_thread.is_alive():
            self.ml_thread.join(timeout=1.0)
        self.cap.release()
