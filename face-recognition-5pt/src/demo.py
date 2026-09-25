# src/demo.py
"""
Falcon Eye — Full-screen Face Locking & Expression Tracking Demo.

Features:
  1. External HD Camera support (index 2 / DirectShow) with fullscreen display.
  2. Target Identity Locking: Identifies enrolled target (e.g., Sandra) vs Unknown/Background faces.
  3. Background Face Handling: Strangers marked as 'Unknown' and ignored by tracking.
  4. Nose Tip Tracking: Computes distance (pixels) & direction relative to frame center.
  5. Multi-Expression Tracking: Smiling, Frowning, Sadness, Blinking (with counter), and Grimacing.
  6. Missing Face Alert: Displays a prominent RED/ORANGE warning banner when target is missing.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt, FaceKpsBox
from .cam_config import get_cam_index
from .recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    MatchResult,
    DISTANCE_THRESHOLD,
)

DB_PATH = Path("data/db/face_db.npz")
ARC_FACE_MODEL = "models/embedder_arcface.onnx"
WINDOW_NAME = "Falcon Eye - Face Locking System"


# ============================================================
# DRAWING UTILITIES
# ============================================================

def draw_text_with_outline(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.65,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
):
    """Draw crisp text with an outline for readability over any background."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, outline_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_centered_banner(
    frame: np.ndarray,
    text: str,
    subtext: str = "",
    bg_color: Tuple[int, int, int] = (0, 0, 180),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.75,
    y_center: Optional[int] = None,
):
    """Draw a semi-transparent banner across the screen."""
    h, w = frame.shape[:2]
    yc = y_center if y_center is not None else h // 2
    banner_h = 100 if subtext else 70
    y1 = max(0, yc - banner_h // 2)
    y2 = min(h, yc + banner_h // 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (w - tw) // 2
    ty = y1 + 40 if subtext else y1 + (banner_h + th) // 2
    draw_text_with_outline(frame, text, tx, ty, 1.0, text_color, 2)

    if subtext:
        (sw, sh), _ = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        sx = (w - sw) // 2
        sy = ty + 35
        draw_text_with_outline(frame, subtext, sx, sy, 0.65, (230, 230, 230), 1)


def draw_scanning_animation(frame: np.ndarray, t: float):
    """Animated futuristic scanner with pulsing 'Looking for match...' text."""
    h, w = frame.shape[:2]

    # Dim background slightly
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Animated laser scanline
    scan_y = int((t * 120) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 220, 255), 2)
    cv2.line(frame, (0, (scan_y - 2) % h), (w, (scan_y - 2) % h), (0, 150, 255), 1)

    # Pulsing status
    dots = "." * (int(t * 3) % 4 + 1)
    draw_centered_banner(
        frame,
        f"LOOKING FOR MATCH{dots}",
        "Position your face in front of the camera",
        bg_color=(50, 40, 20),
        text_color=(0, 215, 255),
        alpha=0.6,
        y_center=h // 2,
    )


def draw_center_crosshair(frame: np.ndarray, cx: int, cy: int):
    """Draw crosshair at the center of the frame."""
    size = 18
    color = (255, 255, 0)
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
    cv2.circle(frame, (cx, cy), 6, color, 1)


def draw_nose_vector(
    frame: np.ndarray,
    cx: int,
    cy: int,
    nx: int,
    ny: int,
    dx: int,
    dy: int,
    dist_px: float,
    dir_str: str,
):
    """Draw vector line, reticle, and distance/direction stats for nose tip."""
    # Line from frame center to nose tip
    cv2.line(frame, (cx, cy), (nx, ny), (0, 255, 255), 2, cv2.LINE_AA)
    # Nose target reticle
    cv2.circle(frame, (nx, ny), 7, (0, 255, 255), -1)
    cv2.circle(frame, (nx, ny), 12, (0, 200, 255), 2)

    # Offset tag near nose
    tag = f"Nose: {dist_px:.0f}px | {dir_str}"
    tag_x = min(frame.shape[1] - 260, max(20, nx + 15))
    tag_y = max(30, ny - 10)
    draw_text_with_outline(frame, tag, tag_x, tag_y, 0.55, (0, 255, 255), 1)


def draw_expressions_card(
    frame: np.ndarray,
    face: FaceKpsBox,
    blink_count: int,
    x: int = 20,
    y: int = 90,
):
    """Draw HUD card showing tracking for 5 expressions/actions."""
    w_card = 310
    h_card = 200

    # Semi-transparent background card
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_card, y + h_card), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w_card, y + h_card), (70, 85, 105), 1)

    draw_text_with_outline(frame, "LIVE EXPRESSION SIGNALS", x + 15, y + 26, 0.58, (255, 215, 0), 2)
    cv2.line(frame, (x + 15, y + 33), (x + w_card - 15, y + 33), (80, 100, 120), 1)

    expressions = [
        ("Smiling", face.is_smiling, face.smile_score, (0, 255, 120)),
        ("Frowning", face.is_frowning, face.frown_score, (0, 160, 255)),
        ("Sadness", face.is_sad, face.sad_score, (255, 150, 50)),
        ("Blinking", face.is_blinking, face.blink_score, (0, 255, 255)),
        ("Grimacing", face.is_grimacing, face.grimace_score, (220, 100, 255)),
    ]

    row_y = y + 60
    for name, is_active, score, active_color in expressions:
        status_txt = "ACTIVE" if is_active else "Off"
        color = active_color if is_active else (140, 150, 160)
        draw_text_with_outline(frame, f"{name}:", x + 15, row_y, 0.50, (220, 220, 220), 1)
        draw_text_with_outline(frame, f"{status_txt} ({score:.2f})", x + 120, row_y, 0.50, color, 2 if is_active else 1)

        # Mini progress bar
        bar_x = x + 215
        bar_w = 75
        bar_h = 10
        cv2.rectangle(frame, (bar_x, row_y - 10), (bar_x + bar_w, row_y - 10 + bar_h), (50, 60, 70), -1)
        fill_w = int(max(0.0, min(1.0, score)) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, row_y - 10), (bar_x + fill_w, row_y - 10 + bar_h), color, -1)

        row_y += 26


# ============================================================
# MAIN DEMO PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("  FALCON EYE — Full Screen Face Locking & Expression Tracking")
    print("=" * 65)

    # 1. Load detector
    print("\n[1/4] Initializing FaceLandmarker & Haar detector...")
    detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    # 2. Load ArcFace embedder
    print("[2/4] Initializing ArcFace ONNX embedder...")
    embedder = ArcFaceEmbedderONNX(model_path=ARC_FACE_MODEL, input_size=(112, 112))

    # 3. Load face database
    print("[3/4] Loading enrolled identities database...")
    db = load_db_npz(DB_PATH)
    matcher = FaceDBMatcher(db=db, dist_thresh=DISTANCE_THRESHOLD)
    print(f"       Enrolled identities: {', '.join(sorted(db.keys())) if db else '(none)'}")

    # 4. Open External HD Camera
    cam_idx = get_cam_index()
    print(f"\n[4/4] Opening External HD Camera (Index {cam_idx})...")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_idx}")

    # Set HD resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create Fullscreen Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nSystem ready!")
    print("  [q] Quit | [f] Toggle Fullscreen | [r] Reset Target Lock\n")

    # State tracking
    target_locked = False
    locked_identity: Optional[str] = None
    lock_timestamp = 0.0
    blink_counter = 0
    was_blinking = False
    fullscreen = True

    # Performance
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = now

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            vis = frame.copy()

            # Draw center crosshair
            draw_center_crosshair(vis, cx, cy)

            # Detect faces
            faces = detector.detect(frame, max_faces=5)

            target_face: Optional[FaceKpsBox] = None
            target_result: Optional[MatchResult] = None

            # Process all detected faces
            for face in faces:
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                res = matcher.match(emb)

                # Check if this face is our enrolled target
                if res.accepted:
                    # Enrolled target found
                    target_face = face
                    target_result = res

                    # Draw Target bounding box in GREEN
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 100), 2)
                    label = f"TARGET: {res.name} ({res.similarity * 100:.1f}%)"
                    draw_text_with_outline(vis, label, face.x1, max(25, face.y1 - 10), 0.70, (0, 255, 100), 2)

                    # 5-pt landmarks
                    for px, py in face.kps.astype(int):
                        cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 255), -1)
                else:
                    # Background / Stranger face -> Mark as Unknown and Ignore
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (120, 120, 120), 1)
                    draw_text_with_outline(
                        vis,
                        "Unknown (Ignored)",
                        face.x1,
                        max(20, face.y1 - 8),
                        0.55,
                        (180, 180, 180),
                        1,
                    )

            # State Machine: Search vs Locked vs Missing
            if not target_locked:
                if target_face is not None and target_result is not None:
                    # Transition to LOCKED
                    target_locked = True
                    locked_identity = target_result.name
                    lock_timestamp = now
                    print(f"  [LOCK ACQUIRED] Identity: {locked_identity} (Conf: {target_result.similarity*100:.1f}%)")
                else:
                    # Searching state
                    draw_scanning_animation(vis, now)
            else:
                # Target was previously locked
                if target_face is not None:
                    # Target is currently in frame: Track nose tip & expressions
                    time_since_lock = now - lock_timestamp
                    if time_since_lock < 2.5:
                        draw_centered_banner(
                            vis,
                            "MATCH DETECTED!",
                            f"Locked on {locked_identity} ({target_result.similarity * 100:.1f}%)",
                            bg_color=(0, 130, 50),
                            text_color=(255, 255, 255),
                            alpha=0.75,
                            y_center=h // 2 - 120,
                        )

                    # 1. Nose Tip Distance & Direction relative to center
                    nx, ny = int(target_face.kps[2, 0]), int(target_face.kps[2, 1])
                    dx = nx - cx
                    dy = ny - cy
                    dist_px = math.sqrt(dx * dx + dy * dy)

                    h_dir = "RIGHT" if dx > 15 else ("LEFT" if dx < -15 else "CENTER")
                    v_dir = "DOWN" if dy > 15 else ("UP" if dy < -15 else "CENTER")
                    dir_str = f"{h_dir} {abs(dx)}px, {v_dir} {abs(dy)}px" if (h_dir != "CENTER" or v_dir != "CENTER") else "CENTERED"

                    draw_nose_vector(vis, cx, cy, nx, ny, dx, dy, dist_px, dir_str)

                    # 2. Blink Counter
                    if target_face.is_blinking and not was_blinking:
                        blink_counter += 1
                        was_blinking = True
                    elif not target_face.is_blinking:
                        was_blinking = False

                    # 3. Live Expression Signals Card
                    draw_expressions_card(vis, target_face, blink_counter, x=20, y=80)

                    # Top Status Bar
                    draw_text_with_outline(
                        vis,
                        f"LOCKED: {locked_identity}",
                        20,
                        40,
                        0.85,
                        (0, 255, 120),
                        2,
                    )
                else:
                    # Target is MISSING from frame: Display prominent RED/ORANGE warning!
                    # Pulsing orange/red border alert
                    pulse_color = (0, 69, 255) if int(now * 4) % 2 == 0 else (0, 0, 255)
                    cv2.rectangle(vis, (0, 0), (w, h), pulse_color, 8)

                    draw_centered_banner(
                        vis,
                        "WARNING: TARGET FACE MISSING FROM FRAME!",
                        f"Target '{locked_identity}' not detected in camera view",
                        bg_color=(0, 0, 180),
                        text_color=(255, 255, 255),
                        alpha=0.85,
                        y_center=110,
                    )

            # Top right info
            draw_text_with_outline(vis, f"FPS: {fps:.0f}", w - 160, 35, 0.55, (100, 255, 100), 1)
            draw_text_with_outline(vis, f"Blinks: {blink_counter}", w - 160, 62, 0.55, (0, 255, 255), 1)
            draw_text_with_outline(vis, "Press [Q] Exit | [F] Fullscreen | [R] Reset", 20, h - 20, 0.50, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord("r"):
                target_locked = False
                locked_identity = None
                blink_counter = 0
                print("  [RESET] Resetting target lock, searching again...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
