import collections
import glob
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from backend.database import (
    ARCHIVE_DIR,
    CONTINUOUS_DIR,
    EVENTS_DIR,
    find_matching_entity,
    get_storage_stats,
    log_sighting,
)
from backend.fingerprint import fingerprint_engine


class VideoRecorder:
    """Manages event-triggered recording with pre-roll buffer and continuous DVR loop."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        pre_roll_seconds: float = 2.0,
        event_duration_seconds: float = 10.0,
        event_cooldown_seconds: float = 15.0,
        retention_hours: float = 24.0,
        min_free_percent: float = 15.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.event_duration = event_duration_seconds
        self.event_cooldown = event_cooldown_seconds
        self.retention_hours = retention_hours
        self.min_free_percent = min_free_percent

        # Rolling pre-roll buffer in RAM (60 frames at 30 FPS = 2 seconds)
        buffer_len = int(fps * pre_roll_seconds)
        self.frame_buffer = collections.deque(maxlen=buffer_len)
        self.buffer_lock = threading.Lock()

        # Event recording state
        self.is_recording_event = False
        self.last_event_time = 0.0

        # Continuous DVR recording state
        self.continuous_enabled = False
        self.continuous_writer: Optional[cv2.VideoWriter] = None
        self.continuous_chunk_start: float = 0.0
        self.chunk_duration_seconds = 300.0  # 5-minute chunks

        self.stopped = False

        # Dedicated DVR queue & worker thread to decouple disk encoding from camera capture
        self.dvr_queue = queue.Queue(maxsize=120)
        self.dvr_thread = threading.Thread(target=self._dvr_worker, daemon=True)
        self.dvr_thread.start()

        # Background worker for continuous chunk rotation and disk safety pruning
        self.pruning_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.pruning_thread.start()

    def add_frame(self, frame: np.ndarray):
        """Adds a fresh live frame to circular pre-roll buffer and DVR queue without blocking."""
        if self.stopped or frame is None:
            return

        with self.buffer_lock:
            self.frame_buffer.append(frame.copy())

        if self.continuous_enabled:
            try:
                self.dvr_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Drop frame instead of stuttering camera capture loop

    def trigger_event(self, detections: List[Dict[str, Any]]) -> bool:
        """Triggers an event-clip recording if cooldown has passed and target is detected."""
        now = time.time()
        if self.is_recording_event or (now - self.last_event_time < self.event_cooldown):
            return False

        if not detections:
            return False

        # Sort by confidence to find the best detection
        best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
        if best_det.get("confidence", 0.0) < 0.60:
            return False

        self.is_recording_event = True
        self.last_event_time = now

        # Launch asynchronous clip recording thread
        threading.Thread(
            target=self._record_event_worker,
            args=(best_det,),
            daemon=True,
        ).start()
        return True

    def _record_event_worker(self, best_det: Dict[str, Any]):
        """Records the pre-roll frames + subsequent frames into an MP4 clip."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_filename = f"event_{timestamp_str}.mp4"
        snap_filename = f"snap_{timestamp_str}.jpg"

        clip_path = os.path.join(EVENTS_DIR, clip_filename)
        snap_path = os.path.join(EVENTS_DIR, snap_filename)
        archive_snap_path = os.path.join(ARCHIVE_DIR, snap_filename)

        # 1. Grab pre-roll frames from RAM
        with self.buffer_lock:
            pre_roll_frames = list(self.frame_buffer)

        # 2. Save best 1080p snapshot
        if pre_roll_frames:
            best_frame = pre_roll_frames[-1]
            cv2.imwrite(snap_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(archive_snap_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        raw_clip_path = clip_path + ".raw.mp4"

        # 3. Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_clip_path, fourcc, self.fps, (self.width, self.height))

        # Write pre-roll frames
        for f in pre_roll_frames:
            writer.write(f)

        # Record remaining duration (e.g. next 8 seconds)
        remaining_frames = int((self.event_duration - 2.0) * self.fps)

        for _ in range(remaining_frames):
            time.sleep(1.0 / self.fps)
            with self.buffer_lock:
                if self.frame_buffer:
                    writer.write(self.frame_buffer[-1])

        writer.release()

        # Convert to web-standard H.264 (avc1) with +faststart for instant browser playback
        try:
            cmd = [
                "ffmpeg", "-y", "-i", raw_clip_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", clip_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(clip_path):
                if os.path.exists(raw_clip_path):
                    os.remove(raw_clip_path)
            else:
                if os.path.exists(raw_clip_path):
                    os.rename(raw_clip_path, clip_path)
        except Exception as e:
            print(f"[WARN] ffmpeg conversion error: {e}")
            if os.path.exists(raw_clip_path):
                os.rename(raw_clip_path, clip_path)

        self.is_recording_event = False

        # 4. Extract Tier 2 visual embedding and check for existing entity match
        embedding = None
        matched_entity_id = None
        if pre_roll_frames and best_frame is not None:
            try:
                embedding = fingerprint_engine.extract_embedding(best_frame, best_det.get("box"))
                matched_entity_id, matched_name, sim = find_matching_entity(
                    embedding, category=best_det.get("category", "other"), min_similarity=0.88, min_margin=0.03
                )
                if matched_entity_id:
                    print(
                        f"[RE-ID] Auto-matched sighting to '{matched_name}' (confidence: {sim:.2f})"
                    )
            except Exception as e:
                print(f"[WARN] Error extracting embedding: {e}")

        # 5. Log sighting to SQLite database
        log_sighting(
            label=best_det.get("label", "unknown"),
            category=best_det.get("category", "other"),
            confidence=best_det.get("confidence", 0.0),
            snapshot_filename=snap_filename,
            clip_filename=clip_filename,
            metadata={"box": best_det.get("box"), "resolution": f"{self.width}x{self.height}"},
            embedding=embedding,
            entity_id=matched_entity_id,
        )
        print(f"[RECORDER] Sighting recorded: {best_det.get('label')} -> {clip_filename}")

    def _dvr_worker(self):
        """Dedicated background worker that writes DVR chunks safely without touching capture loop."""
        while not self.stopped:
            if not self.continuous_enabled:
                if self.continuous_writer is not None:
                    try:
                        self.continuous_writer.release()
                    except Exception as e:
                        print(f"[WARN] Error releasing DVR writer: {e}")
                    self.continuous_writer = None
                    print("[DVR] Continuous chunk finalized.")

                # Drain queue while disabled
                while not self.dvr_queue.empty():
                    try:
                        self.dvr_queue.get_nowait()
                    except queue.Empty:
                        break
                time.sleep(0.1)
                continue

            try:
                frame = self.dvr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                continue

            now = time.time()
            if (
                self.continuous_writer is None
                or (now - self.continuous_chunk_start) >= self.chunk_duration_seconds
            ):
                if self.continuous_writer is not None:
                    try:
                        self.continuous_writer.release()
                    except Exception as e:
                        print(f"[WARN] Error rotating DVR chunk: {e}")

                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                chunk_filename = f"dvr_{timestamp_str}.mp4"
                chunk_path = os.path.join(CONTINUOUS_DIR, chunk_filename)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.continuous_writer = cv2.VideoWriter(
                    chunk_path, fourcc, self.fps, (self.width, self.height)
                )
                self.continuous_chunk_start = now
                print(f"[DVR] Started new 5-min continuous chunk: {chunk_filename}")

            if self.continuous_writer is not None:
                try:
                    self.continuous_writer.write(frame)
                except Exception as e:
                    print(f"[WARN] Error writing continuous DVR frame: {e}")

    def set_continuous_recording(self, enabled: bool):
        """Toggles continuous DVR loop on or off safely."""
        self.continuous_enabled = enabled
        if not enabled:
            print("[DVR] Continuous loop stopped by user.")

    def _maintenance_loop(self):
        """Runs every 60 seconds to prune old continuous chunks and enforce the 15% disk safety guard."""
        while not self.stopped:
            try:
                self._prune_continuous_storage()
            except Exception as e:
                print(f"[WARN] Storage maintenance error: {e}")
            time.sleep(60.0)

    def _prune_continuous_storage(self):
        """Enforces retention hours and guarantees min 15% free space by purging oldest DVR chunks."""
        # Get list of continuous chunk files sorted by modification time (oldest first)
        chunks = sorted(
            glob.glob(os.path.join(CONTINUOUS_DIR, "dvr_*.mp4")),
            key=os.path.getmtime,
        )
        if not chunks:
            return

        now = time.time()
        retention_seconds = self.retention_hours * 3600.0

        # 1. Purge chunks older than configured retention window
        for chunk in list(chunks):
            file_age = now - os.path.getmtime(chunk)
            if file_age > retention_seconds:
                try:
                    os.remove(chunk)
                    chunks.remove(chunk)
                    print(f"[DVR PRUNE] Deleted expired chunk (> {self.retention_hours}h): {os.path.basename(chunk)}")
                except OSError:
                    pass

        # 2. Disk Safety Guard: Ensure free space remains above min_free_percent (15%)
        stats = get_storage_stats()
        free_pct = stats.get("free_percent", 100.0)

        while free_pct < self.min_free_percent and chunks:
            oldest_chunk = chunks.pop(0)
            try:
                os.remove(oldest_chunk)
                print(f"[DISK GUARD] Free space ({free_pct}%) < {self.min_free_percent}%. Purged: {os.path.basename(oldest_chunk)}")
                # Re-check disk space
                stats = get_storage_stats()
                free_pct = stats.get("free_percent", 100.0)
            except OSError:
                break

    def stop(self):
        self.stopped = True
        self.set_continuous_recording(False)
