# src/face_signals.py
"""
Face signals extraction (EAR for blink/eyes-closed, mouth-to-face ratio for smile)
Based on Part 2 Guide: Face Tracking with Identity Lock (Appendix A1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: Tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    width = max(distance(p1, p4), 1e-6)
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool


class _FaceMeshTaskAdapter:
    """Fallback adapter for MediaPipe Tasks API matching the legacy FaceMesh interface."""
    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path("models/face_landmarker.task")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_image: np.ndarray):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        res = self.landmarker.detect(mp_img)

        class _Result:
            def __init__(self, landmarks):
                self.multi_face_landmarks = [landmarks] if landmarks else []

        class _Face:
            def __init__(self, lm):
                self.landmark = lm

        if res.face_landmarks:
            return _Result(_Face(res.face_landmarks[0]))
        return _Result(None)

    def close(self) -> None:
        pass


class FaceSignalExtractor:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_on: float = 0.38,
        smile_off: float = 0.35,
    ):
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.smile_on = smile_on
        self.smile_off = smile_off
        self.low_ear_frames = 0
        self.smiling = False

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.mesh = _FaceMeshTaskAdapter()

    def reset(self) -> None:
        self.low_ear_frames = 0
        self.smiling = False

    def close(self) -> None:
        self.mesh.close()

    def analyze(self, frame: np.ndarray, bbox) -> Optional[FaceSignals]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.12 * bw), int(0.18 * bh)
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        result = self.mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None

        rh, rw = roi.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        points = np.array(
            [[p.x * rw + rx1, p.y * rh + ry1] for p in lm],
            dtype=np.float32,
        )

        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        ear = 0.5 * (left_ear + right_ear)

        blink = False
        if ear < self.ear_threshold:
            self.low_ear_frames += 1
        else:
            if self.blink_min_frames <= self.low_ear_frames <= self.blink_max_frames:
                blink = True
            self.low_ear_frames = 0

        eyes_closed = self.low_ear_frames >= self.closed_frames

        face_width = max(distance(points[FACE_LEFT], points[FACE_RIGHT]), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
        smile_score = mouth_width / face_width

        if self.smiling:
            self.smiling = smile_score >= self.smile_off
        else:
            self.smiling = smile_score >= self.smile_on

        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
        )
