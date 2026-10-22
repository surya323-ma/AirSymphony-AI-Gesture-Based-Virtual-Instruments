# AirSymphony – AI Gesture-Based Virtual Instruments

Play drums, an electro kit, piano and percussion in mid-air. Your webcam tracks both hands with MediaPipe;
every fingertip is its own drumstick, and a fast downward flick over a pad plays it (faster = louder).

## What's new
- **4 kits / 32 sounds**: Acoustic, Electro, Piano (8 notes), Percussion – all synthesized with numpy, no assets needed.
- **Velocity-sensitive taps**, two-hand support, all five fingers.
- **Hover-to-switch kits**: hold your index finger over a tab for ~1 s (or press 1-4 / click).
- **Loop recorder**: `R` record, `R` again to loop it, `C` clear. Plays across kit changes.
- **Metronome** with accent beat (`M`, `+`/`-` tempo), master volume (`[` `]`).
- **Polished UI**: glowing pads, ripples, particles, finger trails, hand skeleton, live HUD, help overlay (`H`).
- **Works without a camera**: keyboard `A S D F J K L ;` and mouse clicks trigger pads.
- **Bring your own sounds**: drop a `.wav` into `sounds/<Kit>/` with the same filename (e.g. `1_Kick.wav`) to replace it.
- Works with both old (`mp.solutions`) and new (Tasks API, auto-downloads `hand_landmarker.task`) MediaPipe.

## Run
```bash
python virtual_drum.py                # default camera
python virtual_drum.py --camera 1     # another webcam
python virtual_drum.py --no-camera    # keyboard / mouse only
python virtual_drum.py --fullscreen
```
Dependencies (`opencv-python`, `numpy`, `pygame`, `mediapipe`) install automatically on first run.

## Controls
| Key | Action | Key | Action |
|---|---|---|---|
| 1-4 | Switch kit | R / C | Record-loop / clear |
| A S D F J K L ; | Play pads 1-8 | M, + / - | Metronome, tempo |
| [ / ] | Volume | F / H | Fullscreen / help |
| Q / Esc | Quit | | |

## Tuning
In `virtual_drum.py`: `TAP_SPEED` (lower = more sensitive), `MIN_RETRIGGER`, `DWELL_TIME`.
Lag? Lower `W, H`. Wrong camera? `--camera N`. Good light and a plain background help tracking.

## Ideas
MIDI out, chord pads, custom kit editor, per-pad sample import UI.
