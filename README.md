# AI Gesture Puzzle

A webcam hand-gesture controlled jigsaw puzzle built with OpenCV + MediaPipe.

Your own photo is sliced into a 3x3 grid of puzzle pieces. Use your bare hand
in front of the webcam to pick up, drag, and place each piece back into its
correct spot on the puzzle board — no mouse or keyboard needed for gameplay.

## Gestures

| Gesture              | Action                                  |
|-----------------------|------------------------------------------|
| **Pinch** (thumb + index touching) | Pick up / drag the piece under your fingertip |
| **Open palm**          | Release the piece you're holding (snaps into place if close enough) |
| **Fist held ~1.2s**    | Reshuffle all unplaced pieces |

## Keyboard shortcuts

- `C` — reload the source photo (in case you swap the file while running)
- `S` — shuffle the scattered pieces
- `R` — full reset (unplace everything and reshuffle)
- `G` — toggle the cinematic color grading on/off
- `Q` — quit

## Look & feel

The webcam feed gets a standard cinematic color grade applied before anything
is drawn on top: a gentle contrast S-curve, teal/orange split toning
(cool shadows, warm highlights), a light saturation lift, and a soft vignette.
This only affects what's displayed — hand detection always runs on the raw,
ungraded frame, so tracking accuracy isn't touched. Toggle it off anytime
with `G` if you'd rather see the plain feed.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put your photo in this folder and name it `my_photo.jpg`
   (or pass a custom path — see below). The photo is automatically
   center-cropped to a square and resized for the puzzle.

3. Run it:
   ```bash
   python app.py
   ```

## Options

```bash
python app.py --photo path/to/your_photo.jpg   # use a specific photo
python app.py --camera 1                       # use a different webcam
```

## Notes

- Works best in good, even lighting with your hand clearly visible against
  the background.
- The faint image ghosted onto the puzzle board is just a placement guide —
  it fades once you've filled in the real pieces.
- Grid size is 3x3 (9 pieces) by default. To change it, edit `GRID_SIZE` in
  `app.py` (must evenly divide `BOARD_SIZE`, default 300).
