import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")


class FingerprintEngine:
    """Tier 2 visual identity fingerprinting and deep face re-identification engine.

    Combines:
    1. Deep Face Landmark Alignment (YuNet 5-point) + Affine Rotational Invariance + Deep SFace embeddings.
    2. Center-weighted multi-scale body appearance features (elliptical Gaussian masked Lab + HSV + Sobel gradients).
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        self.face_detector = None
        self.face_recognizer = None

        self._init_face_models()

    def _init_face_models(self):
        """Initializes OpenCV YuNet face detector and SFace deep recognizer if model weights exist."""
        try:
            if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
                self.face_detector = cv2.FaceDetectorYN.create(
                    YUNET_PATH, "", (320, 320), score_threshold=0.35, nms_threshold=0.3
                )
                self.face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
                print("[INFO] Deep Face Engine ready (YuNet + SFace initialized with affine landmark alignment)")
            else:
                print(f"[WARN] Face models not found in {MODELS_DIR}. Falling back to visual body features.")
        except Exception as e:
            print(f"[WARN] Failed to initialize face models: {e}")

    def extract_embedding(
        self,
        bgr_frame: np.ndarray,
        box: Optional[List[float]] = None,
        category: Optional[str] = "person",
    ) -> Dict[str, Any]:
        """Extracts dual-stream feature embeddings:
        - Deep rotation-aligned face embedding (128D) via YuNet + SFace when face is visible.
        - Center-weighted multi-scale body appearance embedding (192D).
        """
        h, w = bgr_frame.shape[:2]

        crop = bgr_frame
        crop_box = None
        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = box
            y1 = max(0, int(ymin * h))
            x1 = max(0, int(xmin * w))
            y2 = min(h, int(ymax * h))
            x2 = min(w, int(xmax * w))
            if y2 > y1 and x2 > x1:
                crop = bgr_frame[y1:y2, x1:x2]
                crop_box = (x1, y1, x2 - x1, y2 - y1)

        # 1. Extract Deep Aligned Face Embedding
        face_feature = self._extract_face_feature(bgr_frame, crop, crop_box)

        # 2. Extract Center-Weighted Body Appearance Embedding (192D)
        body_feature = self._extract_body_feature(crop)

        return {
            "version": 2,
            "face": [round(float(v), 5) for v in face_feature] if face_feature is not None else None,
            "body": [round(float(v), 5) for v in body_feature],
        }

    def _extract_face_feature(
        self,
        full_frame: np.ndarray,
        crop: np.ndarray,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Detects face with YuNet, performs affine rotation alignment to horizontal eyes, and extracts SFace 128D vector."""
        if not self.face_detector or not self.face_recognizer:
            return None

        # Prefer searching inside crop with small margin if person box is known, else search full frame
        target_img = crop
        offset_x, offset_y = 0, 0
        if crop_box is not None:
            # Add 20% margin around crop within full frame to ensure head contour isn't clipped
            fh, fw = full_frame.shape[:2]
            bx, by, bw, bh = crop_box
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(fw, bx + bw + pad_x)
            y2 = min(fh, by + bh + pad_y)
            if y2 > y1 and x2 > x1:
                target_img = full_frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

        th, tw = target_img.shape[:2]
        if th < 30 or tw < 30:
            return None

        try:
            self.face_detector.setInputSize((tw, th))
            _, faces = self.face_detector.detect(target_img)

            if faces is None or len(faces) == 0:
                # Fallback search on full frame if crop missed the face
                if target_img is not full_frame and full_frame.shape[:2] != (th, tw):
                    fh, fw = full_frame.shape[:2]
                    self.face_detector.setInputSize((fw, fh))
                    _, faces = self.face_detector.detect(full_frame)
                    target_img = full_frame

            if faces is None or len(faces) == 0:
                return None

            # Pick face with largest area / highest confidence
            best_face = sorted(faces, key=lambda f: f[2] * f[3] * f[-1], reverse=True)[0]

            # SFace alignCrop rotates the face via affine warp so eyes are horizontally level
            aligned_face = self.face_recognizer.alignCrop(target_img, best_face)
            feat = self.face_recognizer.feature(aligned_face).flatten()
            norm = np.linalg.norm(feat)
            if norm > 1e-7:
                feat = feat / norm
            return feat
        except Exception as e:
            return None

    def _extract_body_feature(self, crop: np.ndarray) -> np.ndarray:
        """Extracts 192D center-weighted color and gradient appearance features."""
        resized = cv2.resize(crop, self.target_size)

        # Center-weighting elliptical Gaussian mask to attenuate peripheral backgrounds / lights
        y, x = np.ogrid[:self.target_size[1], :self.target_size[0]]
        cy, cx = self.target_size[1] // 2, self.target_size[0] // 2
        mask = np.exp(-(((x - cx)**2) / (2 * 45**2) + ((y - cy)**2) / (2 * 55**2)))
        mask = (mask / mask.max()).astype(np.float32)

        features = []

        # 1. Spatial Lab Color Grid (4x4 grid, 16 cells)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_weighted = lab * mask[:, :, np.newaxis]
        cell_h = self.target_size[1] // 4
        cell_w = self.target_size[0] // 4

        for r in range(4):
            for c in range(4):
                cell = lab_weighted[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                cell_m = mask[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                m_sum = np.sum(cell_m) + 1e-5
                mean_l = np.sum(cell[:, :, 0]) / m_sum
                mean_a = np.sum(cell[:, :, 1]) / m_sum
                mean_b = np.sum(cell[:, :, 2]) / m_sum
                std_l = np.std(cell[:, :, 0])
                std_a = np.std(cell[:, :, 1])
                std_b = np.std(cell[:, :, 2])
                features.extend([
                    mean_l / 255.0,
                    mean_a / 255.0,
                    mean_b / 255.0,
                    std_l / 128.0,
                    std_a / 128.0,
                    std_b / 128.0,
                ])
        # 16 * 6 = 96 features

        # 2. HSV Color Distribution (32 features)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

        for hist in (hist_h, hist_s, hist_v):
            norm = np.linalg.norm(hist) + 1e-7
            features.extend((hist / norm).tolist())
        # 96 + 32 = 128 features

        # 3. Gradient Shape / Silhouette Histogram (32 features)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag = mag * mask

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

        # 4. Intensity Profile (32 features)
        row_means = np.mean(gray * mask, axis=1)
        col_means = np.mean(gray * mask, axis=0)
        sub_rows = cv2.resize(row_means.reshape(-1, 1), (1, 16)).flatten() / 255.0
        sub_cols = cv2.resize(col_means.reshape(1, -1), (16, 1)).flatten() / 255.0
        features.extend(sub_rows.tolist())
        features.extend(sub_cols.tolist())
        # 160 + 32 = 192 features

        feat_arr = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(feat_arr)
        if norm > 1e-7:
            feat_arr = feat_arr / norm
        return feat_arr

    def cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Computes similarity between two embeddings.
        - If both have deep face embeddings, computes face cosine similarity (rotation-invariant).
        - If face is absent in either, falls back to center-weighted appearance similarity.
        """
        f1, b1 = self._unpack_vector(vec1)
        f2, b2 = self._unpack_vector(vec2)

        # Deep Face matching: gold standard with rotation invariance
        if f1 is not None and f2 is not None:
            raw_sim = float(np.dot(f1, f2))
            return self._scale_face_sim(raw_sim)

        # Body appearance fallback
        if b1 is not None and b2 is not None:
            body_sim = float(np.dot(b1, b2))
            return max(0.0, min(1.0, body_sim))

        return 0.0

    @staticmethod
    def _scale_face_sim(raw_sim: float) -> float:
        """Scales SFace hyperspherical cosine similarity to standard [0.0, 1.0] confidence scale.
        SFace baseline: >= 0.35 is true identity match, < 0.25 is distinct person.
        """
        if raw_sim <= 0.0:
            return 0.0
        if raw_sim < 0.35:
            # 0.0 .. 0.35 maps to 0.0 .. 0.88
            return (raw_sim / 0.35) * 0.88
        else:
            # 0.35 .. 1.0 maps to 0.88 .. 1.0
            return min(1.0, 0.88 + ((raw_sim - 0.35) / 0.65) * 0.12)

    def _unpack_vector(self, vec: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Unpacks vector from dict or legacy list."""
        if not vec:
            return None, None
        if isinstance(vec, dict):
            f = np.array(vec["face"], dtype=np.float32) if vec.get("face") else None
            b = np.array(vec["body"], dtype=np.float32) if vec.get("body") else None
            if f is not None and np.linalg.norm(f) > 1e-7:
                f = f / np.linalg.norm(f)
            if b is not None and np.linalg.norm(b) > 1e-7:
                b = b / np.linalg.norm(b)
            return f, b
        elif isinstance(vec, (list, tuple)):
            arr = np.array(vec, dtype=np.float32)
            if len(arr) == 128:
                return arr / (np.linalg.norm(arr) + 1e-7), None
            else:
                return None, arr / (np.linalg.norm(arr) + 1e-7)
        return None, None

    def merge_embeddings(
        self,
        current_centroid: Any,
        new_vec: Any,
        count: int = 1,
    ) -> Dict[str, Any]:
        """Merges two embeddings (face and body) into an updated centroid."""
        f1, b1 = self._unpack_vector(current_centroid)
        f2, b2 = self._unpack_vector(new_vec)

        merged_face = None
        if f1 is not None and f2 is not None:
            mf = (f1 * count + f2) / (count + 1)
            merged_face = (mf / (np.linalg.norm(mf) + 1e-7)).tolist()
            merged_face = [round(float(x), 5) for x in merged_face]
        elif f1 is not None:
            merged_face = [round(float(x), 5) for x in f1]
        elif f2 is not None:
            merged_face = [round(float(x), 5) for x in f2]

        merged_body = None
        if b1 is not None and b2 is not None:
            mb = (b1 * count + b2) / (count + 1)
            merged_body = (mb / (np.linalg.norm(mb) + 1e-7)).tolist()
            merged_body = [round(float(x), 5) for x in merged_body]
        elif b1 is not None:
            merged_body = [round(float(x), 5) for x in b1]
        elif b2 is not None:
            merged_body = [round(float(x), 5) for x in b2]

        return {
            "version": 2,
            "face": merged_face,
            "body": merged_body,
        }


# Global singleton instance
fingerprint_engine = FingerprintEngine()
