import json
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class FingerprintEngine:
    """Tier 2 visual identity fingerprinting and re-identification engine.

    Extracts multi-feature embeddings (spatial Lab color distribution + HSV histograms + gradient textures)
    and performs cosine similarity matching to re-identify unique people, vehicles, and animals.
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size

    def extract_embedding(
        self,
        bgr_frame: np.ndarray,
        box: Optional[List[float]] = None,
    ) -> List[float]:
        """Extracts a 192-dimensional L2-normalized feature vector from an object crop.

        Args:
            bgr_frame: Full BGR image from camera (e.g. 1920x1080).
            box: Optional normalized bounding box [ymin, xmin, ymax, xmax].
        """
        h, w = bgr_frame.shape[:2]

        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            y1 = max(0, int(ymin * h))
            x1 = max(0, int(xmin * w))
            y2 = min(h, int(ymax * h))
            x2 = min(w, int(xmax * w))
            if y2 > y1 and x2 > x1:
                crop = bgr_frame[y1:y2, x1:x2]
            else:
                crop = bgr_frame
        else:
            crop = bgr_frame

        # Resize to fixed standard size
        resized = cv2.resize(crop, self.target_size)

        features = []

        # 1. Spatial Lab Color Grid (4x4 grid, 16 cells)
        # Perceptually uniform Lab color space captures clothing, vehicle paint, and fur reliably
        # across daylight and IR illuminator modes.
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        cell_h = self.target_size[1] // 4
        cell_w = self.target_size[0] // 4

        for r in range(4):
            for c in range(4):
                cell = lab[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                mean_l, mean_a, mean_b = np.mean(cell, axis=(0, 1))
                std_l, std_a, std_b = np.std(cell, axis=(0, 1))
                # Normalize values
                features.extend([
                    mean_l / 255.0,
                    mean_a / 255.0,
                    mean_b / 255.0,
                    std_l / 128.0,
                    std_a / 128.0,
                    std_b / 128.0,
                ])
        # 16 cells * 6 features = 96 features

        # 2. HSV Color Distribution (32 features: 16 Hue, 8 Sat, 8 Val)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

        for hist in (hist_h, hist_s, hist_v):
            norm = np.linalg.norm(hist) + 1e-7
            features.extend((hist / norm).tolist())
        # 96 + 32 = 128 features

        # 3. Gradient Shape / Silhouette Histogram (32 features)
        # Grayscale Sobel gradients across 4 quadrants (8 gradient direction bins each)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        quad_h = self.target_size[1] // 2
        quad_w = self.target_size[0] // 2

        for qr in range(2):
            for qc in range(2):
                sub_ang = angle[qr * quad_h : (qr + 1) * quad_h, qc * quad_w : (qc + 1) * quad_w]
                sub_mag = mag[qr * quad_h : (qr + 1) * quad_h, qc * quad_w : (qc + 1) * quad_w]
                hist_grad, _ = np.histogram(sub_ang, bins=8, range=(0, 360), weights=sub_mag)
                norm = np.linalg.norm(hist_grad) + 1e-7
                features.extend((hist_grad / norm).tolist())
        # 128 + 32 = 160 features

        # 4. Intensity Profile / LBP Variance (32 features)
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        # Downsample to 16 rows and 16 cols
        sub_rows = cv2.resize(row_means.reshape(-1, 1), (1, 16)).flatten() / 255.0
        sub_cols = cv2.resize(col_means.reshape(1, -1), (16, 1)).flatten() / 255.0
        features.extend(sub_rows.tolist())
        features.extend(sub_cols.tolist())
        # Total: 160 + 32 = 192 features

        feat_arr = np.array(features, dtype=np.float32)
        l2_norm = np.linalg.norm(feat_arr)
        if l2_norm > 1e-7:
            feat_arr = feat_arr / l2_norm

        return [round(float(v), 5) for v in feat_arr]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two unit-normalized embedding vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        dot = float(np.dot(a, b))
        return max(0.0, min(1.0, dot))

    @staticmethod
    def merge_embeddings(
        current_centroid: Optional[List[float]],
        new_vec: List[float],
        count: int = 1,
    ) -> List[float]:
        """Calculates updated centroid vector when a new sighting is added to an entity."""
        if not current_centroid:
            return list(new_vec)
        c = np.array(current_centroid, dtype=np.float32)
        v = np.array(new_vec, dtype=np.float32)
        updated = (c * count + v) / (count + 1)
        norm = np.linalg.norm(updated)
        if norm > 1e-7:
            updated = updated / norm
        return [round(float(x), 5) for x in updated]


# Global singleton instance
fingerprint_engine = FingerprintEngine()
