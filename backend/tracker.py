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


def compute_center_distance(boxA: List[float], boxB: List[float]) -> float:
    """Computes normalized Euclidean distance between box centers (0.0 to 1.414)."""
    cA_x = (boxA[1] + boxA[3]) / 2.0
    cA_y = (boxA[0] + boxA[2]) / 2.0
    cB_x = (boxB[1] + boxB[3]) / 2.0
    cB_y = (boxB[0] + boxB[2]) / 2.0
    return float(np.sqrt((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2))


class Tracklet:
    """Represents a continuous physical subject tracked across frames with identity locking."""

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

        # Persistent identity lock (e.g. Stefan or Monica once identified)
        self.entity_id: Optional[int] = None
        self.entity_name: Optional[str] = None
        self.has_face: bool = False
        self.best_embedding: Optional[Dict[str, Any]] = None

        # Trajectory history for movement verification
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
        if len(self.positions) > 15:
            self.positions.pop(0)

        # 1.5% displacement across history or persistent active presence confirms physical subject
        if len(self.positions) >= 2:
            init_x, init_y = self.positions[0]
            dist = np.sqrt((cx - init_x) ** 2 + (cy - init_y) ** 2)
            if dist > 0.015 or self.hits >= 3:
                self.has_moved = True

    def lock_identity(self, entity_id: int, entity_name: str, embedding: Optional[Dict[str, Any]] = None, has_face: bool = False):
        """Locks this track to a known identity. Once locked, the identity persists across all movements."""
        self.entity_id = entity_id
        self.entity_name = entity_name
        if embedding:
            self.best_embedding = embedding
        if has_face:
            self.has_face = True

    def mark_missed(self):
        self.time_since_update += 1


class MultiObjectTracker:
    """Online spatial-temporal tracker linking detections by IoU and center proximity across ticks."""

    def __init__(
        self,
        min_iou: float = 0.15,
        max_center_distance: float = 0.25,
        max_age: int = 15,  # Keep track alive for 15 seconds across pauses or turns
        min_hits_for_active: int = 2,
    ):
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.max_age = max_age
        self.min_hits_for_active = min_hits_for_active
        self.tracks: List[Tracklet] = []
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Tracklet]:
        """Updates tracks with new detections via IoU + proximity, preserving locked identities."""
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            # Build match score matrix: prefers IoU, falls back to center proximity
            scores = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    if track.category == det.get("category"):
                        d_box = det.get("box", [0, 0, 0, 0])
                        iou = compute_iou(track.box, d_box)
                        c_dist = compute_center_distance(track.box, d_box)

                        if iou >= self.min_iou:
                            scores[t_idx, d_idx] = 1.0 + iou
                        elif c_dist <= self.max_center_distance:
                            # Proximity match (e.g. person bent over or turned, changing aspect ratio)
                            scores[t_idx, d_idx] = 1.0 - c_dist

            # Greedy assignment
            while True:
                max_val = float(np.max(scores))
                if max_val <= 0.0:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(scores), scores.shape)
                self.tracks[t_idx].update(detections[d_idx])
                matched_tracks.add(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
                scores[t_idx, :] = -1.0
                scores[:, d_idx] = -1.0

        # Mark missed tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            new_track = Tracklet(self._next_id, detections[d_idx])
            self._next_id += 1
            self.tracks.append(new_track)

        # Prune dead tracks (retained for up to max_age seconds)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed active tracks (observed for >= min_hits and verified moved)
        active_confirmed = [
            t for t in self.tracks
            if t.hits >= self.min_hits_for_active and t.time_since_update == 0 and t.has_moved
        ]
        return active_confirmed
