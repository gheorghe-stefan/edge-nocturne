import cv2
import threading
import time
from typing import List, Optional, Tuple
from backend.detector import ObjectDetector
from backend.recorder import VideoRecorder
from backend.tracker import MultiObjectTracker


class VideoCamera:
    """Threaded camera capture + asynchronous background LiteRT ML + dual-mode recording."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        enable_ml: bool = True,
    ):
        if device_id is None:
            device_id = self._find_camera_device()

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

        # Fallback search if specified device failed to grab initial frame
        if not self.grabbed:
            print(f"[WARN] Failed to read initial frame from camera device {self.device_id}, scanning...")
            self.cap.release()
            discovered = self._find_camera_device()
            if discovered != self.device_id:
                self.device_id = discovered
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
        self.tracker = MultiObjectTracker(min_iou=0.25, max_age=3, min_hits_for_active=2)

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

    @staticmethod
    def _find_camera_device() -> int:
        for idx in range(10):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"[INFO] Discovered camera on device index {idx}")
                        return idx
            except Exception:
                continue
        print("[WARN] No working camera device found on scan, defaulting to index 0")
        return 0

    def _capture_loop(self):
        """Dedicated thread to constantly drain camera buffer at full frame rate."""
        while not self.stopped:
            try:
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.frame_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                    self.recorder.add_frame(frame)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[WARN] Camera capture loop warning: {e}")
                time.sleep(0.02)

    def _inference_loop(self):
        """Asynchronous ML worker running high-accuracy inference (~1.5 FPS) with online multi-object tracking."""
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

                    # Update online multi-object tracker
                    active_tracks = self.tracker.update(detections)

                    # Trigger event recording if verified active moving targets exist
                    if active_tracks:
                        active_dets = [
                            {
                                "label": t.label,
                                "category": t.category,
                                "confidence": t.confidence,
                                "box": t.box,
                                "track_id": t.track_id,
                            }
                            for t in active_tracks
                        ]
                        self.recorder.trigger_event(active_dets)

                except Exception as e:
                    print(f"[ERROR] Inference error: {e}")

            # Pace inference to ~1.0 FPS (for 960x960) or ~1.5 FPS (for 640x640)
            sleep_interval = 0.90 if (self.detector and self.detector.target_size >= 960) else 0.65
            time.sleep(sleep_interval)

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

    def get_jpeg_bytes(self, quality: int = 80, resolution: Optional[str] = "1080p") -> Optional[bytes]:
        """Encodes the clean raw frame as JPEG bytes for HTTP streaming at requested resolution."""
        success, frame = self.get_latest_frame()
        if not success or frame is None:
            return None

        if resolution == "720p":
            frame = cv2.resize(frame, (1280, 720))
        elif resolution == "480p":
            frame = cv2.resize(frame, (640, 480))

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
