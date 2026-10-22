# AirSymphony-AI-Gesture 
A real-time, webcam-powered instrument that lets you play drum or piano sounds in mid-air. Hand landmarks are tracked with MediaPipe, visualized with OpenCV, and every finger tap triggers a pygame sound pad with immediate visual feedback.

# Features
Auto-installs OpenCV, MediaPipe, and pygame if missing.
Tracks one hand via webcam and detects downward finger taps per lane.
Five instrument pads (kick, snare, hi-hat, tom, clap) with animated overlays.
Procedurally generated wav samples; no external assets required.
Runs full-screen OpenCV window with responsive tap ripples and instrument art.
Quit cleanly by pressing q.
# Requirements
Windows, macOS, or Linux with a functional webcam.
Python 3.10+ (tested on 3.11).
Internet access for the first run (to download dependencies automatically).
Quick Start
# Fork this repo to your GitHub account.
Click the Fork button on GitHub.
Choose your account/org and confirm.
Clone your fork locally:
git clone https://github.com/surya323-ma/AirSymphony-AI-Gesture-Based-Virtual-Instruments
cd virtual-air-drum
Run the app (dependencies auto-install on first launch):
python virtual_drum.py
Point the webcam toward your hands. Tap downward over each virtual pad to trigger sounds.
Usage Notes
The script will create a sounds/ directory on first launch and drop synthesized wav files there.
If your webcam defaults to a different index, adjust cv2.VideoCapture(0) inside virtual_drum.py.
Keep your hand roughly centered; lanes are horizontal slices with unique instrument art.
To exit, focus the OpenCV window and press q.
Troubleshooting
Webcam fails to open: ensure no other app is using it; try a different camera index (0, 1, 2...).
Lag or low FPS: lower capture resolution by changing the CAP_PROP_FRAME_WIDTH/HEIGHT values.
No audio: confirm system volume, ensure pygame mixer initialized (console logs), and that the generated wav files exist under sounds/.
Dependency install errors: manually install with pip install opencv-python mediapipe pygame then rerun.
Contributing
Fork the repo (see Quick Start).
Create a branch for your feature/fix:
git checkout -b feature/my-improvement
Commit your work with clear messages.
Push to your fork:
git push origin feature/my-improvement
Open a Pull Request against the main branch of the upstream repository. Describe the change, testing steps, and include screenshots if UI-related.
Roadmap Ideas
Add multiple hand support for chords or layered instruments.
Replace synthesized tones with studio samples.
Provide MIDI output for integration with DAWs.
Enjoy jamming in mid-air!
