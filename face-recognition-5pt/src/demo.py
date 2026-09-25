# src/demo.py
"""
Falcon Eye — Full-screen face recognition demo.
Clean, stable single-window experience:
  1. "Looking for a match..." until a known face is found
  2. "Match detected! — <Name>" once recognized
  3. Then shows smile/blink detection in real-time
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .cam_config import get_cam_index
from .face_signals import FaceSignalExtractor

# We reuse the embedder, matcher, and DB loader from recognize.py
from .recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    MatchResult,
    DISTANCE_THRESHOLD,
)

DB_PATH = Path("data/db/face_db.npz")
ARC_FACE_MODEL = "models/embedder_arcface.onnx"

WINDOW = "Falcon Eye"


# ============================================================
# DRAWING HELPERS
# ============================================================

def put_text(frame, text, x, y, scale=0.7, color=(255, 255, 255), thickness=2):
    """Draw text with dark outline for readability on any background."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def put_center_text(frame, text, scale=1.2, color=(255, 255, 255), y_offset=0):
    """Draw large centered text."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    x = (w - tw) // 2
    y = (h + th) // 2 + y_offset
    put_text(frame, text, x, y, scale, color, 2)


def draw_scanning_overlay(frame, t):
    """Animated scanning line + pulsing text."""
    h, w = frame.shape[:2]

    # Dim the frame slightly
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Pulsing "Looking for a match..." text
    pulse = abs(int(time.time() * 3) % 6 - 3)  # 0-3 pulse
    dots = "." * (pulse + 1)
    put_center_text(frame, f"Looking for a match{dots}", 1.0, (100, 200, 255))

    # Animated scanning line
    scan_y = int((time.time() * 80) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 180, 255), 2)


def draw_match_banner(frame, name, similarity, t_since_match):
    """Celebratory match detected banner that fades after a few seconds."""
    h, w = frame.shape[:2]
    alpha = max(0.0, min(1.0, 1.0 - (t_since_match - 2.0) / 1.0))  # fade after 2s

    if alpha > 0:
        banner_h = 80
        y_top = h // 2 - 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (w, y_top + banner_h), (0, 140, 0), -1)
        cv2.addWeighted(overlay, 0.6 * alpha, frame, 1.0 - 0.6 * alpha, 0, frame)

        put_center_text(frame, "MATCH DETECTED!", 1.1, (0, 255, 100), y_offset=-100)
        put_center_text(frame, name, 1.4, (255, 255, 255), y_offset=-50)
        conf_text = f"Confidence: {similarity * 100:.1f}%"
        put_center_text(frame, conf_text, 0.7, (200, 255, 200), y_offset=0)


def draw_face_box(frame, face, name, result):
    """Draw bounding box and identity label."""
    color = (0, 255, 100) if (result and result.accepted) else (0, 140, 255)

    cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)

    # Corner accents
    corner_len = 20
    for (cx, cy) in [(face.x1, face.y1), (face.x2, face.y1),
                     (face.x1, face.y2), (face.x2, face.y2)]:
        dx = corner_len if cx == face.x1 else -corner_len
        dy = corner_len if cy == face.y1 else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy), color, 3)

    # Keypoints
    for x, y in face.kps.astype(int):
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    # Name label
    label = name if name else "Unknown"
    label_y = max(25, face.y1 - 12)
    put_text(frame, label, face.x1, label_y, 0.75, color, 2)

    if result:
        conf = f"{result.similarity * 100:.1f}%"
        put_text(frame, conf, face.x1, label_y + 25, 0.55, (200, 200, 200), 1)


def draw_signals(frame, face, signals):
    """Draw smile and blink signals below the face box."""
    y = min(frame.shape[0] - 60, face.y2 + 25)
    x = face.x1

    # Smile
    if signals.smiling:
        put_text(frame, "SMILING", x, y, 0.65, (0, 255, 200), 2)
    else:
        put_text(frame, "Neutral", x, y, 0.55, (160, 160, 160), 1)

    # Blink / Eyes
    y += 28
    if signals.eyes_closed:
        put_text(frame, "EYES CLOSED", x, y, 0.65, (0, 140, 255), 2)
    elif signals.blink:
        put_text(frame, "BLINK!", x, y, 0.65, (0, 255, 255), 2)
    else:
        put_text(frame, "Eyes open", x, y, 0.55, (160, 160, 160), 1)


def draw_hud(frame, fps, blink_count, state):
    """Top-right HUD with FPS and blink counter."""
    h, w = frame.shape[:2]
    put_text(frame, f"FPS: {fps:.0f}", w - 160, 30, 0.55, (100, 255, 100), 1)
    put_text(frame, f"Blinks: {blink_count}", w - 160, 55, 0.55, (255, 255, 100), 1)
    put_text(frame, state, 15, 30, 0.6, (255, 200, 100), 1)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print()
    print("=" * 55)
    print("  FALCON EYE — Face Recognition Demo")
    print("=" * 55)
    print()

    # Load pipeline
    print("[1/4] Loading face detector...")
    detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    print("[2/4] Loading ArcFace embedder...")
    embedder = ArcFaceEmbedderONNX(model_path=ARC_FACE_MODEL, input_size=(112, 112))

    print("[3/4] Loading face database...")
    db = load_db_npz(DB_PATH)
    matcher = FaceDBMatcher(db=db, dist_thresh=DISTANCE_THRESHOLD)
    print(f"       Enrolled: {', '.join(sorted(db.keys())) if db else '(empty)'}")

    print("[4/4] Loading face signal analyzer...")
    signal_extractor = FaceSignalExtractor()

    # Camera
    cam_idx = get_cam_index()
    print(f"\nOpening camera {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_idx}")

    # Full-screen window
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\nReady! Press 'q' to quit, 'f' to toggle fullscreen.\n")

    # State
    match_found = False
    match_name = None
    match_similarity = 0.0
    match_time = 0.0
    blink_count = 0
    fullscreen = True

    # FPS tracking
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            t_now = time.time()
            dt = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t_prev = t_now

            vis = frame.copy()

            # --- Detect faces ---
            faces = detector.detect(frame, max_faces=5)

            best_result = None
            best_face = None

            for face in faces:
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                embedding = embedder.embed(aligned)
                result = matcher.match(embedding)

                draw_face_box(vis, face, result.name if result.accepted else "Stranger", result)

                if result.accepted:
                    if best_result is None or result.similarity > best_result.similarity:
                        best_result = result
                        best_face = face

            # --- State transitions ---
            if best_result is not None and best_result.accepted:
                if not match_found:
                    match_found = True
                    match_name = best_result.name
                    match_similarity = best_result.similarity
                    match_time = t_now
                    print(f"  MATCH: {match_name} (confidence {match_similarity*100:.1f}%)")
                else:
                    match_name = best_result.name
                    match_similarity = best_result.similarity

            # --- Draw overlays ---
            if not match_found:
                # Phase 1: Searching
                draw_scanning_overlay(vis, t_now)
                draw_hud(vis, fps, blink_count, "SEARCHING")
            else:
                # Phase 2: Match found — show signals
                t_since = t_now - match_time
                if t_since < 3.0:
                    draw_match_banner(vis, match_name, match_similarity, t_since)

                # Analyze smile/blink on the best matched face
                if best_face is not None:
                    box = (best_face.x1, best_face.y1, best_face.x2, best_face.y2)
                    sig = signal_extractor.analyze(frame, box)
                    if sig is not None:
                        draw_signals(vis, best_face, sig)
                        if sig.blink:
                            blink_count += 1

                draw_hud(vis, fps, blink_count, f"LOCKED: {match_name}")

            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if key == ord("r"):
                # Reset: go back to searching
                match_found = False
                match_name = None
                blink_count = 0
                signal_extractor.reset()
                print("  Reset — searching again...")

    finally:
        cap.release()
        signal_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
