"""
AI Gesture Puzzle
-----------------
A webcam hand-gesture controlled jigsaw puzzle.

Instead of capturing your face live from the webcam, this version slices
the puzzle directly from a photo you provide (default: my_photo.jpg).
Your hand is still tracked live via the webcam so you can play with gestures:

    PINCH (thumb + index tip together)  -> pick up / drag a puzzle piece
    OPEN PALM                            -> release a held piece
    FIST held ~1.2s                      -> reshuffle the scattered pieces
    (Fist held again after a full reset) -> full reset of the board

Keys:
    c = re-capture / reload the source photo
    s = shuffle
    r = reset
    g = toggle cinematic color grading
    q = quit

Usage:
    python app.py                     # uses my_photo.jpg in this folder
    python app.py --photo path.jpg    # uses a specific photo
    python app.py --camera 1          # use a different webcam index
"""

import argparse
import math
import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GRID_SIZE = 3                     # 3x3 = 9 pieces
BOARD_SIZE = 300                  # rendered size (px) of the target board
PIECE_MARGIN = 4                  # gap drawn around each scattered piece
SNAP_RADIUS = 40                  # px distance to auto-snap a piece into place
PINCH_THRESHOLD = 40              # px distance between thumb tip & index tip
FIST_HOLD_SECONDS = 1.2           # how long a fist must be held to trigger reshuffle
WINDOW_NAME = "AI Gesture Puzzle"

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def _rgb(r, g, b):
    """Write colors as RGB (readable) but store as BGR (what cv2 wants)."""
    return (b, g, r)


# ---- Color palette (modern dark theme) ----
COL_PANEL = _rgb(20, 20, 28)
COL_PANEL_BORDER_LINE = _rgb(80, 84, 100)
COL_TEXT = _rgb(240, 242, 248)
COL_TEXT_MUTED = _rgb(148, 152, 168)
COL_ACCENT_TEAL = _rgb(45, 212, 191)      # open palm / release
COL_ACCENT_AMBER = _rgb(251, 191, 36)     # pinch / grab
COL_ACCENT_RED = _rgb(248, 113, 113)      # fist / reshuffle
COL_ACCENT_GREEN = _rgb(74, 222, 128)     # success / placed
COL_ACCENT_INDIGO = _rgb(129, 140, 248)   # board border / branding
COL_SHADOW = _rgb(0, 0, 0)

GESTURE_COLORS = {
    "PINCH": COL_ACCENT_AMBER,
    "OPEN_PALM": COL_ACCENT_TEAL,
    "FIST": COL_ACCENT_RED,
    "NONE": COL_TEXT_MUTED,
}

# Deterministic confetti particles (reused every frame, animated purely via time)
_CONFETTI_RNG = np.random.default_rng(7)
_CONFETTI_N = 60
_CONFETTI_X = _CONFETTI_RNG.uniform(0, 1, _CONFETTI_N)
_CONFETTI_Y0 = _CONFETTI_RNG.uniform(-1, 0, _CONFETTI_N)
_CONFETTI_SPEED = _CONFETTI_RNG.uniform(60, 160, _CONFETTI_N)
_CONFETTI_SIZE = _CONFETTI_RNG.uniform(3, 7, _CONFETTI_N)
_CONFETTI_COLORS = [COL_ACCENT_TEAL, COL_ACCENT_AMBER, COL_ACCENT_GREEN, COL_ACCENT_INDIGO, COL_ACCENT_RED]
_CONFETTI_COLOR_IDX = _CONFETTI_RNG.integers(0, len(_CONFETTI_COLORS), _CONFETTI_N)


# --------------------------------------------------------------------------- #
# Low-level drawing helpers (rounded panels, shadows, rounded-corner pieces)
# --------------------------------------------------------------------------- #
def draw_rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_glass_panel(img, pt1, pt2, radius=14, color=COL_PANEL, alpha=0.6, border_color=None):
    """Semi-transparent rounded panel, used for all HUD chrome."""
    overlay = img.copy()
    draw_rounded_rect(overlay, pt1, pt2, radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color is not None:
        draw_rounded_rect(img, pt1, pt2, radius, border_color, 1)


def put_text(img, text, org, scale=0.6, color=COL_TEXT, thickness=1, font=FONT, shadow=True):
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 2), font, scale, COL_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def text_width(text, scale=0.6, thickness=1, font=FONT):
    (w, _h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w


def rounded_mask(w, h, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_rounded_rect(mask, (0, 0), (w, h), radius, 255, -1)
    return mask


def paste_rounded(dst, patch, x, y, radius, shadow=True):
    """Blend `patch` onto `dst` at (x, y) with rounded corners and an optional
    soft drop shadow, clipped to the destination bounds."""
    H, W = dst.shape[:2]
    h, w = patch.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    x1, y1 = max(x, 0), max(y, 0)
    if x2 <= x1 or y2 <= y1:
        return

    if shadow:
        sx1, sy1 = max(x1 + 4, 0), max(y1 + 6, 0)
        sx2, sy2 = min(x2 + 4, W), min(y2 + 6, H)
        if sx2 > sx1 and sy2 > sy1:
            shadow_overlay = dst.copy()
            draw_rounded_rect(shadow_overlay, (sx1, sy1), (sx2, sy2), radius, COL_SHADOW, -1)
            cv2.addWeighted(shadow_overlay, 0.35, dst, 0.65, 0, dst)

    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    patch_crop = patch[py1:py2, px1:px2]

    mask = rounded_mask(w, h, radius)[py1:py2, px1:px2]
    mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    roi = dst[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1 - mask3) + patch_crop.astype(np.float32) * mask3
    dst[y1:y2, x1:x2] = blended.astype(np.uint8)


def draw_glow_ring(img, center, radius, color, thickness=3, pulse=0.0):
    """A soft pulsing ring, used to highlight the piece currently held."""
    r = int(radius + 4 * math.sin(pulse * 6.0))
    overlay = img.copy()
    cv2.circle(overlay, center, max(r, 1), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)


# --------------------------------------------------------------------------- #
# Color grading (applied to the webcam feed only, never to detection input)
# --------------------------------------------------------------------------- #
ENABLE_COLOR_GRADE = True

_VIGNETTE_CACHE = {}


def _vignette_mask(h, w, strength=0.25):
    key = (h, w)
    if key not in _VIGNETTE_CACHE:
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        mask = 1.0 - strength * (dist / max_dist) ** 2
        _VIGNETTE_CACHE[key] = np.clip(mask, 0, 1)[..., None].astype(np.float32)
    return _VIGNETTE_CACHE[key]


def apply_color_grade(frame):
    """Standard cinematic grade: gentle S-curve contrast, teal-shadow /
    orange-highlight split toning, a light saturation lift, and a vignette."""
    img = frame.astype(np.float32) / 255.0

    # Gentle S-curve contrast
    img = np.clip((img - 0.5) * 1.08 + 0.5, 0.0, 1.0)

    # Teal/orange split toning based on per-pixel luminance
    luminance = img.mean(axis=2, keepdims=True)
    shadow_mask = np.clip(1.0 - luminance * 1.6, 0.0, 1.0)
    highlight_mask = np.clip((luminance - 0.45) * 1.8, 0.0, 1.0)
    teal_shadows = np.array([0.025, 0.015, -0.010], dtype=np.float32)     # BGR: lift blue/green in shadows
    orange_highlights = np.array([-0.020, 0.010, 0.030], dtype=np.float32)  # BGR: lift red in highlights
    img = img + shadow_mask * teal_shadows + highlight_mask * orange_highlights
    img = np.clip(img, 0.0, 1.0)

    graded = (img * 255).astype(np.uint8)

    # Saturation lift
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Vignette
    h, w = graded.shape[:2]
    graded = np.clip(graded.astype(np.float32) * _vignette_mask(h, w), 0, 255).astype(np.uint8)

    return graded


# --------------------------------------------------------------------------- #
# Puzzle piece
# --------------------------------------------------------------------------- #
class PuzzlePiece:
    """A single jigsaw tile cut from the source photo."""

    def __init__(self, image_patch, grid_row, grid_col, piece_size):
        self.image = image_patch                 # the cropped patch (BGR)
        self.grid_row = grid_row
        self.grid_col = grid_col
        self.size = piece_size
        self.placed = False

        # current top-left position on screen (scattered position, filled in later)
        self.x = 0
        self.y = 0

        # the correct top-left position once placed on the board
        self.target_x = 0
        self.target_y = 0

    def contains(self, px, py):
        return self.x <= px <= self.x + self.size and self.y <= py <= self.y + self.size

    def center(self):
        return self.x + self.size // 2, self.y + self.size // 2


# --------------------------------------------------------------------------- #
# Hand gesture helpers
# --------------------------------------------------------------------------- #
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def fingers_extended(landmarks, w, h):
    """
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended, using simple landmark
    y/x comparisons (works reasonably well for a front-facing hand).
    """
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]

    fingers = []

    # Thumb: compare x of tip vs ip joint (mirrors depending on hand orientation,
    # so we just check horizontal spread from the wrist)
    wrist = pts[0]
    thumb_tip = pts[4]
    thumb_ip = pts[3]
    fingers.append(distance(thumb_tip, wrist) > distance(thumb_ip, wrist))

    # Other four fingers: tip above pip joint (smaller y = higher on screen = extended)
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_idx, pip_idx in tips_pips:
        fingers.append(pts[tip_idx][1] < pts[pip_idx][1])

    return fingers


def classify_gesture(landmarks, w, h):
    """Returns one of: 'PINCH', 'OPEN_PALM', 'FIST', 'NONE'"""
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    thumb_tip = pts[4]
    index_tip = pts[8]

    if distance(thumb_tip, index_tip) < PINCH_THRESHOLD:
        return "PINCH"

    fingers = fingers_extended(landmarks, w, h)
    extended_count = sum(fingers)

    if extended_count >= 4:
        return "OPEN_PALM"
    if extended_count <= 1:
        return "FIST"
    return "NONE"


# --------------------------------------------------------------------------- #
# Puzzle setup
# --------------------------------------------------------------------------- #
def load_source_image(photo_path):
    if not os.path.exists(photo_path):
        raise FileNotFoundError(
            f"Could not find '{photo_path}'. Put your photo in this folder "
            f"or pass --photo <path>."
        )
    img = cv2.imread(photo_path)
    if img is None:
        raise ValueError(f"'{photo_path}' exists but could not be read as an image.")

    # Crop to a centered square, then resize to BOARD_SIZE x BOARD_SIZE
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = img[y0:y0 + side, x0:x0 + side]
    square = cv2.resize(square, (BOARD_SIZE, BOARD_SIZE), interpolation=cv2.INTER_AREA)
    return square


def build_pieces(source_square, frame_w, frame_h):
    """Slice source_square into GRID_SIZE x GRID_SIZE pieces and scatter them
    on the left side of the frame."""
    piece_size = BOARD_SIZE // GRID_SIZE
    pieces = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y0 = row * piece_size
            x0 = col * piece_size
            patch = source_square[y0:y0 + piece_size, x0:x0 + piece_size].copy()
            piece = PuzzlePiece(patch, row, col, piece_size)
            pieces.append(piece)

    scatter_pieces(pieces, frame_w, frame_h, piece_size)
    return pieces


def scatter_pieces(pieces, frame_w, frame_h, piece_size):
    """Randomly place all *unplaced* pieces in the left scatter zone."""
    scatter_zone_w = frame_w // 2 - piece_size - 20
    scatter_zone_h = frame_h - piece_size - 40

    for piece in pieces:
        if piece.placed:
            continue
        piece.x = random.randint(10, max(10, scatter_zone_w))
        piece.y = random.randint(40, max(40, scatter_zone_h))


def reset_pieces(pieces, frame_w, frame_h, piece_size):
    for piece in pieces:
        piece.placed = False
    scatter_pieces(pieces, frame_w, frame_h, piece_size)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_board(frame, board_origin, source_square, pieces, piece_size, t=0.0):
    bx, by = board_origin
    pad = 14

    # Frosted backing panel behind the whole board
    draw_glass_panel(
        frame,
        (bx - pad, by - pad - 30),
        (bx + BOARD_SIZE + pad, by + BOARD_SIZE + pad),
        radius=16,
        color=COL_PANEL,
        alpha=0.55,
        border_color=COL_ACCENT_INDIGO,
    )

    # Title with a small accent bar
    cv2.line(frame, (bx, by - 20), (bx, by - 6), COL_ACCENT_INDIGO, 3, cv2.LINE_AA)
    put_text(frame, "PUZZLE BOARD", (bx + 10, by - 8), scale=0.62, color=COL_TEXT, thickness=1)

    # Faint ghost preview of the full target image (placement guide)
    overlay = frame.copy()
    overlay[by:by + BOARD_SIZE, bx:bx + BOARD_SIZE] = source_square
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Grid lines
    for i in range(1, GRID_SIZE):
        cv2.line(frame, (bx + i * piece_size, by), (bx + i * piece_size, by + BOARD_SIZE), COL_PANEL_BORDER_LINE, 1, cv2.LINE_AA)
        cv2.line(frame, (bx, by + i * piece_size), (bx + BOARD_SIZE, by + i * piece_size), COL_PANEL_BORDER_LINE, 1, cv2.LINE_AA)
    draw_rounded_rect(frame, (bx, by), (bx + BOARD_SIZE, by + BOARD_SIZE), 6, COL_ACCENT_INDIGO, 2)

    # Already-placed pieces, each with a rounded frame + success glow
    for piece in pieces:
        if piece.placed:
            tx = bx + piece.grid_col * piece_size
            ty = by + piece.grid_row * piece_size
            inset = 3
            paste_rounded(
                frame, piece.image[inset:-inset, inset:-inset] if inset else piece.image,
                tx + inset, ty + inset, radius=8, shadow=False,
            )
            draw_rounded_rect(frame, (tx + 2, ty + 2), (tx + piece_size - 2, ty + piece_size - 2), 8, COL_ACCENT_GREEN, 2)
            # small checkmark badge in the corner
            bx2, by2 = tx + piece_size - 16, ty + 8
            cv2.circle(frame, (bx2, by2), 8, COL_ACCENT_GREEN, -1, cv2.LINE_AA)
            cv2.line(frame, (bx2 - 4, by2), (bx2 - 1, by2 + 3), COL_PANEL, 2, cv2.LINE_AA)
            cv2.line(frame, (bx2 - 1, by2 + 3), (bx2 + 4, by2 - 3), COL_PANEL, 2, cv2.LINE_AA)


def draw_scattered_pieces(frame, pieces, held_piece, t=0.0):
    for piece in pieces:
        if piece.placed:
            continue
        is_held = piece is held_piece
        radius = 10
        paste_rounded(frame, piece.image, piece.x, piece.y, radius=radius, shadow=True)

        color = COL_ACCENT_AMBER if is_held else COL_PANEL_BORDER_LINE
        thickness = 2 if is_held else 1
        draw_rounded_rect(
            frame, (piece.x, piece.y), (piece.x + piece.size, piece.y + piece.size),
            radius, color, thickness,
        )
        if is_held:
            cx, cy = piece.center()
            draw_glow_ring(frame, (cx, cy), piece.size // 2 + 6, COL_ACCENT_AMBER, thickness=2, pulse=t)


def draw_confetti(frame, t):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for i in range(_CONFETTI_N):
        y = ((_CONFETTI_Y0[i] * h) + t * _CONFETTI_SPEED[i]) % (h + 40) - 20
        x = int(_CONFETTI_X[i] * w)
        size = int(_CONFETTI_SIZE[i])
        color = _CONFETTI_COLORS[_CONFETTI_COLOR_IDX[i]]
        cv2.circle(overlay, (x, int(y)), size, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


def draw_hud(frame, gesture, placed_count, total, fps, fist_progress=0.0, t=0.0):
    h, w = frame.shape[:2]

    # ---- Top bar ----
    draw_glass_panel(frame, (16, 14), (w - 16, 68), radius=16, alpha=0.55, border_color=COL_PANEL_BORDER_LINE)

    # Brand / title, left
    cv2.circle(frame, (38, 38), 5, COL_ACCENT_INDIGO, -1, cv2.LINE_AA)
    put_text(frame, "AI GESTURE PUZZLE", (52, 44), scale=0.68, color=COL_TEXT, thickness=1)
    put_text(frame, "Developed by Bayazid Ahmed", (52, 58), scale=0.34, color=COL_TEXT_MUTED, thickness=1, font=FONT_SMALL)

    # Gesture pill, center
    gesture_color = GESTURE_COLORS.get(gesture, COL_TEXT_MUTED)
    label = f"{gesture.replace('_', ' ')}"
    label_w = text_width(label, 0.6, 1, FONT)
    pill_w = label_w + 60
    pill_x = w // 2 - pill_w // 2
    draw_glass_panel(frame, (pill_x, 20), (pill_x + pill_w, 56), radius=18, color=COL_PANEL, alpha=0.7)
    dot_pulse = 3 if gesture != "NONE" else 0
    cv2.circle(frame, (pill_x + 22, 38), 6 + (1 if gesture != "NONE" else 0), gesture_color, -1, cv2.LINE_AA)
    put_text(frame, label, (pill_x + 38, 44), scale=0.6, color=gesture_color, thickness=1)

    # Fist-hold radial progress ring around the gesture dot
    if fist_progress > 0:
        angle = int(360 * min(fist_progress, 1.0))
        cv2.ellipse(frame, (pill_x + 22, 38), (13, 13), -90, 0, angle, COL_ACCENT_RED, 2, cv2.LINE_AA)

    # Pieces counter, right
    complete = placed_count == total
    count_color = COL_ACCENT_GREEN if complete else COL_TEXT
    count_text = f"{placed_count} / {total}"
    count_w = text_width(count_text, 0.62, 1, FONT)
    cx2 = w - 32
    draw_glass_panel(frame, (cx2 - count_w - 46, 20), (cx2, 56), radius=18, color=COL_PANEL, alpha=0.7)
    put_text(frame, "PIECES", (cx2 - count_w - 38, 32), scale=0.35, color=COL_TEXT_MUTED, thickness=1, font=FONT_SMALL)
    put_text(frame, count_text, (cx2 - count_w - 38, 50), scale=0.62, color=count_color, thickness=1)

    if fist_progress > 0:
        hint = "HOLD FIST TO RESHUFFLE"
        hint_w = text_width(hint, 0.42, 1, FONT_SMALL)
        put_text(frame, hint, (w // 2 - hint_w // 2, 76), scale=0.42, color=COL_ACCENT_RED, thickness=1, font=FONT_SMALL)

    # ---- Bottom bar: key hint chips ----
    chips = [("C", "Reload Photo"), ("S", "Shuffle"), ("R", "Reset"), ("G", "Toggle Grade"), ("Q", "Quit")]
    chip_gap = 10
    chip_x = 20
    chip_y1, chip_y2 = h - 42, h - 16
    draw_glass_panel(frame, (12, chip_y1 - 8), (min(w - 12, 560), chip_y2 + 8), radius=16, alpha=0.5)
    for key, label_txt in chips:
        key_w = 22
        label_w = text_width(label_txt, 0.42, 1, FONT_SMALL)
        chip_w = key_w + label_w + 22
        draw_rounded_rect(frame, (chip_x, chip_y1), (chip_x + key_w, chip_y2), 6, COL_ACCENT_INDIGO, -1)
        put_text(frame, key, (chip_x + 6, chip_y2 - 7), scale=0.45, color=COL_PANEL, thickness=1, font=FONT_SMALL, shadow=False)
        put_text(frame, label_txt, (chip_x + key_w + 8, chip_y2 - 7), scale=0.42, color=COL_TEXT_MUTED, thickness=1, font=FONT_SMALL)
        chip_x += chip_w + chip_gap

    put_text(frame, f"{int(fps)} FPS", (w - 80, h - 22), scale=0.42, color=COL_TEXT_MUTED, thickness=1, font=FONT_SMALL)

    # ---- Completion celebration ----
    if complete:
        draw_confetti(frame, t)
        msg = "PUZZLE COMPLETE!"
        scale = 1.15 + 0.05 * math.sin(t * 3.0)
        msg_w = text_width(msg, scale, 3, FONT)
        panel_x1, panel_y1 = w // 2 - msg_w // 2 - 30, h // 2 - 40
        panel_x2, panel_y2 = w // 2 + msg_w // 2 + 30, h // 2 + 30
        draw_glass_panel(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), radius=20, color=COL_PANEL, alpha=0.6, border_color=COL_ACCENT_GREEN)
        put_text(frame, msg, (w // 2 - msg_w // 2, h // 2 + 12), scale=scale, color=COL_ACCENT_GREEN, thickness=3)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="AI Gesture Puzzle")
    parser.add_argument("--photo", default="my_photo.jpg", help="Path to the photo used for the puzzle")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    source_square = load_source_image(args.photo)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    ok, sample_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read a frame from the webcam.")
    frame_h, frame_w = sample_frame.shape[:2]

    piece_size = BOARD_SIZE // GRID_SIZE
    pieces = build_pieces(source_square, frame_w, frame_h)
    board_origin = (frame_w - BOARD_SIZE - 40, 70)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    held_piece = None
    fist_start_time = None
    prev_time = time.time()
    start_time = time.time()

    print("AI Gesture Puzzle running. Press Q in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if ENABLE_COLOR_GRADE:
            frame = apply_color_grade(frame)

        gesture = "NONE"
        fist_progress = 0.0
        hand_point = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            gesture = classify_gesture(landmarks, frame_w, frame_h)
            index_tip = landmarks[8]
            hand_point = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))

            if gesture == "PINCH":
                fist_start_time = None
                if held_piece is None:
                    for piece in pieces:
                        if not piece.placed and piece.contains(*hand_point):
                            held_piece = piece
                            break
                if held_piece is not None:
                    held_piece.x = hand_point[0] - held_piece.size // 2
                    held_piece.y = hand_point[1] - held_piece.size // 2

            elif gesture == "OPEN_PALM":
                fist_start_time = None
                if held_piece is not None:
                    # Check if it should snap onto its correct board slot
                    bx, by = board_origin
                    target_cx = bx + held_piece.grid_col * piece_size + piece_size // 2
                    target_cy = by + held_piece.grid_row * piece_size + piece_size // 2
                    cx, cy = held_piece.center()
                    if distance((cx, cy), (target_cx, target_cy)) < SNAP_RADIUS:
                        held_piece.placed = True
                    held_piece = None

            elif gesture == "FIST":
                if fist_start_time is None:
                    fist_start_time = time.time()
                fist_progress = (time.time() - fist_start_time) / FIST_HOLD_SECONDS
                if fist_progress >= 1.0:
                    reset_pieces(pieces, frame_w, frame_h, piece_size)
                    fist_start_time = None
                    fist_progress = 0.0
                    held_piece = None
            else:
                fist_start_time = None

        now = time.time()
        t = now - start_time
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_board(frame, board_origin, source_square, pieces, piece_size, t=t)
        draw_scattered_pieces(frame, pieces, held_piece, t=t)

        placed_count = sum(1 for p in pieces if p.placed)
        draw_hud(frame, gesture, placed_count, len(pieces), fps, fist_progress, t=t)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            reset_pieces(pieces, frame_w, frame_h, piece_size)
        elif key == ord("r"):
            for p in pieces:
                p.placed = False
            reset_pieces(pieces, frame_w, frame_h, piece_size)
        elif key == ord("c"):
            source_square = load_source_image(args.photo)
            pieces = build_pieces(source_square, frame_w, frame_h)
            held_piece = None
        elif key == ord("g"):
            globals()["ENABLE_COLOR_GRADE"] = not ENABLE_COLOR_GRADE

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
