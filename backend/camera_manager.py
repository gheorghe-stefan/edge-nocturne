import os
import re
import threading
from typing import Any, Dict, List, Optional
from backend.camera import VideoCamera
from backend.database import add_db_camera, delete_db_camera, get_db_cameras, update_db_camera_status
from backend.detector import ObjectDetector


def mask_rtsp_url(url: str) -> str:
    """Masks password in RTSP URLs for secure display (e.g. rtsp://user:****@host:554/...)."""
    return re.sub(r":([^:@]+)@", r":****@", url)


class CameraManager:
    """Orchestrates multiple camera sources: local USB V4L2 cameras and remote TP-Link Tapo RTSP cameras."""

    def __init__(self):
        self.cameras: Dict[int, VideoCamera] = {}
        self.lock = threading.Lock()
        self.shared_detector: Optional[ObjectDetector] = None

        try:
            print("[INFO] Initializing shared high-accuracy YOLO detector for CameraManager...")
            self.shared_detector = ObjectDetector()
        except Exception as e:
            print(f"[WARN] Failed to initialize shared detector: {e}")

    def load_cameras_from_db(self):
        """Initializes all configured cameras from SQLite."""
        with self.lock:
            db_cams = get_db_cameras()
            for c in db_cams:
                cid = c["id"]
                if c["enabled"] and cid not in self.cameras:
                    try:
                        cam = VideoCamera(
                            camera_id=cid,
                            name=c["name"],
                            source=c["source"],
                            cam_type=c["cam_type"],
                            enable_ml=bool(c["enable_ml"]),
                            shared_detector=self.shared_detector,
                        )
                        self.cameras[cid] = cam
                        print(f"[CAMERA MANAGER] Camera #{cid} '{c['name']}' started.")
                    except Exception as e:
                        print(f"[ERROR] Failed to start camera #{cid} ({c['name']}): {e}")

    def get_camera(self, camera_id: int) -> Optional[VideoCamera]:
        with self.lock:
            return self.cameras.get(camera_id)

    def get_primary_camera(self) -> Optional[VideoCamera]:
        with self.lock:
            if 1 in self.cameras:
                return self.cameras[1]
            if self.cameras:
                return next(iter(self.cameras.values()))
            return None

    def add_camera(self, name: str, source: str, enable_ml: bool = True) -> Dict[str, Any]:
        """Adds a new RTSP camera (e.g. TP-Link Tapo) and starts ingestion immediately."""
        with self.lock:
            cam_id = add_db_camera(name, source, cam_type="rtsp", enable_ml=1 if enable_ml else 0)
            cam = VideoCamera(
                camera_id=cam_id,
                name=name,
                source=source,
                cam_type="rtsp",
                enable_ml=enable_ml,
                shared_detector=self.shared_detector,
            )
            self.cameras[cam_id] = cam
            print(f"[CAMERA MANAGER] New camera #{cam_id} '{name}' registered and online.")
            return {
                "id": cam_id,
                "name": name,
                "source": mask_rtsp_url(source),
                "cam_type": "rtsp",
                "enabled": True,
                "enable_ml": enable_ml,
            }

    def remove_camera(self, camera_id: int) -> bool:
        """Stops and deletes a camera."""
        with self.lock:
            if camera_id in self.cameras:
                cam = self.cameras.pop(camera_id)
                cam.stop()
            deleted = delete_db_camera(camera_id)
            print(f"[CAMERA MANAGER] Camera #{camera_id} removed.")
            return deleted

    def set_camera_capture(self, camera_id: int, active: bool) -> bool:
        """Starts or stops live capture for a specific camera."""
        cam = self.get_camera(camera_id)
        if not cam:
            return False
        cam.set_capture_active(active)
        return True

    def set_all_capture(self, active: bool):
        """Starts or pauses live capture on all cameras simultaneously."""
        with self.lock:
            for cam in self.cameras.values():
                cam.set_capture_active(active)

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Returns real-time status, FPS, resolution, and detections for all configured cameras."""
        db_cams = get_db_cameras()
        status_list = []
        with self.lock:
            for c in db_cams:
                cid = c["id"]
                cam = self.cameras.get(cid)
                is_active = cam.is_capture_active if cam else False
                is_online = (cam is not None and cam.grabbed)

                dets = []
                inf_ms = 0.0
                if cam:
                    dets, inf_ms = cam.get_latest_detections()

                status_list.append({
                    "id": cid,
                    "name": c["name"],
                    "cam_type": c["cam_type"],
                    "source": c["source"] if c["cam_type"] == "local" else mask_rtsp_url(c["source"]),
                    "enabled": bool(c["enabled"]),
                    "enable_ml": bool(c["enable_ml"]),
                    "is_active": is_active,
                    "is_online": is_online,
                    "resolution": f"{cam.width}x{cam.height}" if cam else "offline",
                    "fps": cam.fps if (cam and is_active) else 0,
                    "inference_ms": inf_ms,
                    "detections": dets,
                })
        return status_list

    def stop_all(self):
        """Stops all running cameras cleanly."""
        with self.lock:
            for cid, cam in list(self.cameras.items()):
                try:
                    cam.stop()
                except Exception:
                    pass
            self.cameras.clear()
