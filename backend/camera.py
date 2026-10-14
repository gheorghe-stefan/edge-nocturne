import cv2
import threading
import time
from typing import Optional, Tuple


class VideoCamera:
    """Threaded camera capture to prevent buffer lag and ensure real-time streaming."""

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        # Initialize camera with V4L2 backend and MJPG hardware compression
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.grabbed, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False

        if not self.grabbed:
            print(f"[WARN] Failed to read initial frame from camera device {self.device_id}")

        # Start continuous frame reader thread
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_latest_frame(self) -> Tuple[bool, Optional[any]]:
        """Returns the most recent raw OpenCV BGR frame."""
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Encodes the latest frame as JPEG bytes for HTTP streaming."""
        success, frame = self.get_latest_frame()
        if not success or frame is None:
            return None

        # Encode to JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            return None
        return buffer.tobytes()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
