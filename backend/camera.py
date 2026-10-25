import cv2
import threading
import time
from typing import Any, List, Optional, Tuple
from backend.detector import ObjectDetector
from backend.recorder import VideoRecorder
from backend.tracker import MultiObjectTracker
from backend.fingerprint import fingerprint_engine
from backend.database import find_matching_entity


class VideoCamera:
    """Threaded camera capture (Local V4L2 USB or Remote RTSP/Tapo) + ML + dual-mode recording."""

    def __init__(
        self,
        camera_id: int = 1,
        name: str = "Arducam USB (Local)",
        source: Any = None,
        cam_type: str = "local",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        enable_ml: bool = True,
        shared_detector: Optional[ObjectDetector] = None,
    ):
        self.camera_id = camera_id
        self.name = name
        self.cam_type = cam_type
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_ml = enable_ml

        # Determine stream source type
        is_rtsp = False
        if cam_type == "rtsp" or (isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"))):
            is_rtsp = True
            self.cam_type = "rtsp"
            self.source = str(source)
            self.device_id = None
        else:
            self.cam_type = "local"
            if source is None:
                self.device_id = self._find_camera_device()
            elif isinstance(source, int):
                self.device_id = source
            elif isinstance(source, str) and source.isdigit():
                self.device_id = int(source)
            else:
                self.device_id = self._find_camera_device()
            self.source = str(self.device_id)

        # Initialize capture backend
        if is_rtsp:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            print(f"[INFO] Connecting to RTSP Camera #{self.camera_id} '{self.name}' ({self.source})...")
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.grabbed, self.frame = self.cap.read()
            if self.grabbed and self.frame is not None:
                h, w = self.frame.shape[:2]
                self.height, self.width = h, w
                print(f"[INFO] RTSP Camera #{self.camera_id} connected! Native resolution: {self.width}x{self.height}")
            else:
                print(f"[WARN] Initial frame grab failed for RTSP Camera #{self.camera_id} ({self.source}). Will retry in loop.")
                self.grabbed = False
                self.frame = None
        else:
            # Initialize camera with V4L2 backend and MJPG hardware compression
            print(f"[INFO] Initializing Local USB Camera #{self.camera_id} on index {self.device_id}...")
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
        self.is_capture_active = True

        self.latest_detections: List[dict] = []
        self.inference_ms: float = 0.0
        self.standby_frame = self._create_standby_frame(self.width, self.height)

        if not self.grabbed and self.cam_type == "local":
            print(f"[WARN] Failed to read initial frame from camera device {self.device_id}")

        # Initialize Dual-mode Video Recorder (Event pre-roll + continuous DVR loop)
        self.recorder = VideoRecorder(width=self.width, height=self.height, fps=self.fps)
        self.tracker = MultiObjectTracker(min_iou=0.15, max_center_distance=0.25, max_age=15, min_hits_for_active=2)

        # Thread 1: High-speed ingestion loop (zero lag)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Thread 2: Asynchronous ML inference worker loop
        if self.enable_ml:
            try:
                self.detector = shared_detector or ObjectDetector()
                self.ml_thread = threading.Thread(target=self._inference_loop, daemon=True)
                self.ml_thread.start()
            except Exception as e:
                print(f"[WARN] ObjectDetector initialization skipped: {e}")
                self.detector = None

    @staticmethod
    def _create_standby_frame(width: int = 1920, height: int = 1080) -> any:
        """Generates a professional dark standby frame when live capture is paused."""
        import numpy as np
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (26, 17, 10)  # Dark midnight slate (#0a111a)

        # Ambient inner frame border
        cv2.rectangle(img, (24, 24), (width - 24, height - 24), (55, 42, 28), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "LIVE CAMERA CAPTURE PAUSED"
        sub1 = "Video stream ingestion and AI object detection are temporarily suspended."
        sub2 = "Click 'Start Capture' in the dashboard controls to resume."

        # Centered Title
        t_size = cv2.getTextSize(title, font, 1.3, 3)[0]
        t_x = (width - t_size[0]) // 2
        t_y = height // 2 - 40
        cv2.putText(img, title, (t_x, t_y), font, 1.3, (245, 245, 245), 3, cv2.LINE_AA)

        # Subtitle lines
        s1_size = cv2.getTextSize(sub1, font, 0.85, 2)[0]
        s1_x = (width - s1_size[0]) // 2
        cv2.putText(img, sub1, (s1_x, t_y + 55), font, 0.85, (170, 185, 200), 2, cv2.LINE_AA)

        s2_size = cv2.getTextSize(sub2, font, 0.85, 2)[0]
        s2_x = (width - s2_size[0]) // 2
        cv2.putText(img, sub2, (s2_x, t_y + 105), font, 0.85, (0, 235, 255), 2, cv2.LINE_AA)

        return img

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
        """Dedicated thread to constantly drain camera buffer at full frame rate with RTSP auto-reconnect."""
        consecutive_failures = 0
        last_reconnect_time = 0.0

        while not self.stopped:
            if not self.is_capture_active:
                # Flush underlying device buffer slowly without burning CPU
                try:
                    self.cap.grab()
                except Exception:
                    pass
                time.sleep(0.08)
                continue

            try:
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    consecutive_failures = 0
                    with self.frame_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                    self.recorder.add_frame(frame)
                else:
                    consecutive_failures += 1
                    if self.cam_type == "rtsp" and consecutive_failures >= 15:
                        now = time.time()
                        if now - last_reconnect_time > 4.0:
                            print(f"[WARN] Camera #{self.camera_id} ('{self.name}') feed lost ({consecutive_failures} fails). Reconnecting to RTSP...")
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            time.sleep(1.0)
                            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                            last_reconnect_time = time.time()
                            consecutive_failures = 0
                    time.sleep(0.02)
            except Exception as e:
                consecutive_failures += 1
                print(f"[WARN] Camera #{self.camera_id} capture loop warning: {e}")
                time.sleep(0.05)

    def _inference_loop(self):
        """Asynchronous ML worker running high-accuracy inference (~1.5 FPS) with online multi-object tracking."""
        while not self.stopped:
            if not self.detector or not self.is_capture_active:
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

                    # In-flight identity locking: check face/appearance while subject is moving
                    for t in active_tracks:
                        if not t.has_face:
                            try:
                                emb = fingerprint_engine.extract_embedding(frame, t.box, category=t.category)
                                if emb:
                                    m_id, m_name, sim = find_matching_entity(emb, category=t.category)
                                    if m_id:
                                        t.lock_identity(m_id, m_name, embedding=emb, has_face=(emb.get("face") is not None))
                                        print(f"[TRACKER] Locked Track #{t.track_id} -> '{m_name}' (conf: {sim:.2f})")
                                    elif t.best_embedding is None:
                                        t.best_embedding = emb
                            except Exception:
                                pass

                    # Trigger event recording if verified active moving targets exist
                    if active_tracks:
                        active_dets = [
                            {
                                "label": t.label,
                                "category": t.category,
                                "confidence": t.confidence,
                                "box": t.box,
                                "track_id": t.track_id,
                                "entity_id": t.entity_id,
                                "entity_name": t.entity_name,
                                "embedding": t.best_embedding,
                                "camera_id": self.camera_id,
                                "camera_name": self.name,
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
        """Returns the most recent raw OpenCV BGR frame (or standby frame if paused)."""
        if not self.is_capture_active:
            return True, self.standby_frame.copy()

        with self.frame_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def set_capture_active(self, active: bool):
        """Starts or pauses live camera capture, streaming, and ML inference."""
        with self.frame_lock:
            self.is_capture_active = active
        with self.ml_lock:
            if not active:
                self.latest_detections = []
                self.inference_ms = 0.0
        print(f"[CAMERA] Live capture state changed: is_capture_active={active}")

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
