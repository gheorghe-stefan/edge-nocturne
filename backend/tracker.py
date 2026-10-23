import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between two boxes in [ymin, xmin, ymax, xmax] format."""
    yminA, xminA, ymaxA, xmaxA = boxA
    yminB, xminB, ymaxB, xmaxB = boxB

    inter_ymin = max(yminA, yminB)
    inter_xmin = max(xminA, xminB)
    inter_ymax = min(ymaxA, ymaxB)
    inter_xmax = min(xmaxA, xmaxB)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (xmaxA - xminA) * (ymaxA - yminA))
    areaB = max(0.0, (xmaxB - xminB) * (ymaxB - yminB))
    union_area = areaA + areaB - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


class Tracklet:
    """Represents a single tracked physical subject over time."""

    def __init__(self, track_id: int, detection: Dict[str, Any]):
        self.track_id = track_id
        self.label = detection.get("label", "unknown")
        self.category = detection.get("category", "other")
        self.confidence = detection.get("confidence", 0.0)
        self.box = list(detection.get("box", [0, 0, 0, 0]))

        self.hits = 1
        self.time_since_update = 0
        self.first_seen = time.time()
        self.last_seen = self.first_seen

        # Track trajectory for motion persistence verification
        cx = (self.box[1] + self.box[3]) / 2.0
        cy = (self.box[0] + self.box[2]) / 2.0
        self.positions: List[Tuple[float, float]] = [(cx, cy)]
        self.has_moved = False

    def update(self, detection: Dict[str, Any]):
        self.label = detection.get("label", self.label)
        self.confidence = max(self.confidence, detection.get("confidence", 0.0))
        self.box = list(detection.get("box", self.box))
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = time.time()

        cx = (self.box[1] + self.box[3]) / 2.0
        cy = (self.box[0] + self.box[2]) / 2.0
        self.positions.append((cx, cy))
        if len(self.positions) > 10:
            self.positions.pop(0)

        # Check total displacement across tracklet history
        if len(self.positions) >= 2:
            init_x, init_y = self.positions[0]
            dist = np.sqrt((cx - init_x) ** 2 + (cy - init_y) ** 2)
            # 2% displacement across frame confirms genuine physical motion
            if dist > 0.02 or self.hits >= 4:
                self.has_moved = True

    def mark_missed(self):
        self.time_since_update += 1


class MultiObjectTracker:
    """Lightweight online tracker maintaining track IDs and motion persistence across ticks."""

    def __init__(
        self,
        min_iou: float = 0.25,
        max_age: int = 3,
        min_hits_for_active: int = 2,
    ):
        self.min_iou = min_iou
        self.max_age = max_age
        self.min_hits_for_active = min_hits_for_active
        self.tracks: List[Tracklet] = []
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Tracklet]:
        """Updates tracks with new detections and returns list of confirmed active moving tracks."""
        # 1. Match new detections to existing tracks via IoU
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    # Same category preference
                    if track.category == det.get("category"):
                        iou_matrix[t_idx, d_idx] = compute_iou(track.box, det.get("box", [0, 0, 0, 0]))

            # Greedy assignment
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.min_iou:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                self.tracks[t_idx].update(detections[d_idx])
                matched_tracks.add(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # 2. Mark missed tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # 3. Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            new_track = Tracklet(self._next_id, detections[d_idx])
            self._next_id += 1
            self.tracks.append(new_track)

        # 4. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 5. Return confirmed active tracks (observed for >= min_hits and verified moved)
        active_confirmed = [
            t for t in self.tracks
            if t.hits >= self.min_hits_for_active and t.time_since_update == 0 and t.has_moved
        ]
        return active_confirmed
