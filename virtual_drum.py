"""
AirSymphony - AI Gesture-Based Virtual Instruments
==================================================
Play drums, synths, piano and percussion in mid-air with your webcam.

Controls (also work without a camera - keyboard + mouse fallback):
    1-4          switch kit            (or hover your index finger over a tab)
    A S D F J K L ;   trigger pads 1-8 from the keyboard (or click a pad)
    M            metronome on/off      + / -   tempo
    R            record  -> press R again to loop what you played
    C            clear loop            [ / ]   master volume
    F            fullscreen            H       help overlay
    Q / Esc      quit
"""
import importlib
import math
import os
import subprocess
import sys
import time
import wave
import argparse
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
REQUIRED = {"cv2": "opencv-python", "numpy": "numpy", "pygame": "pygame"}
OPTIONAL = {"mediapipe": "mediapipe"}


def ensure_dependencies() -> None:
    for table, fatal in ((REQUIRED, True), (OPTIONAL, False)):
        for module, package in table.items():
            try:
                importlib.import_module(module)
            except ImportError:
                print(f"Installing missing dependency: {package}")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    if fatal:
                        raise
                    print(f"  (could not install {package}; running in keyboard/mouse mode)")


ensure_dependencies()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

try:
    import mediapipe as mp  # noqa: E402
except ImportError:  # pragma: no cover
    mp = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(HERE, "sounds")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WINDOW = "AirSymphony"
W, H = 1280, 720
TOP_H, BOT_H = 64, 74

FINGERS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_COLORS = {  # BGR
    "thumb": (110, 110, 255), "index": (40, 210, 255), "middle": (130, 230, 90),
    "ring": (255, 160, 100), "pinky": (255, 120, 210),
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
TAP_SPEED = 1.15       # frame-heights per second a fingertip must travel downward
REARM_SPEED = -0.25    # moving up faster than this re-arms a finger
MIN_RETRIGGER = 0.09   # seconds
DWELL_TIME = 0.8       # seconds to hover a tab with the index finger
FONT = cv2.FONT_HERSHEY_DUPLEX
PAD_KEYS = "asdfjkl;"

# --------------------------------------------------------------------------- #
# Procedural sound synthesis (all numpy, no external assets)
# --------------------------------------------------------------------------- #
_rng = np.random.default_rng(7)


def _t(d: float) -> np.ndarray:
    return np.linspace(0, d, int(SR * d), endpoint=False)


def _noise(t: np.ndarray) -> np.ndarray:
    return _rng.standard_normal(len(t))


def _hp(x: np.ndarray, passes: int = 1) -> np.ndarray:
    for _ in range(passes):
        x = np.concatenate(([x[0]], np.diff(x)))
    return x


def _lp(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def _sweep(t, f0, f1, rate):
    freq = f1 + (f0 - f1) * np.exp(-t * rate)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR)


def _finish(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    x = x / (np.max(np.abs(x)) + 1e-9) * peak
    fade = min(len(x), int(0.004 * SR))
    x[-fade:] *= np.linspace(1, 0, fade)
    return x


def kick(f0=150, f1=48, d=0.5, rate=28, decay=7):
    t = _t(d)
    click = _noise(t) * np.exp(-t * 400) * 0.25
    return _finish(_sweep(t, f0, f1, rate) * np.exp(-t * decay) + click)


def kick808(d=1.0):
    t = _t(d)
    return _finish(np.tanh(1.8 * _sweep(t, 130, 38, 14) * np.exp(-t * 3.2)))


def snare(tone=190, d=0.3):
    t = _t(d)
    body = np.sin(2 * np.pi * tone * t) * np.exp(-t * 28) * 0.55
    rattle = _lp(_hp(_noise(t)), 2) * np.exp(-t * 17)
    return _finish(body + rattle)


def e_snare(d=0.28):
    t = _t(d)
    return _finish(_sweep(t, 330, 180, 40) * np.exp(-t * 22) * 0.6 + _hp(_noise(t)) * np.exp(-t * 14))


def hat(d=0.09, decay=45):
    t = _t(d)
    return _finish(_hp(_noise(t), 2) * np.exp(-t * decay), 0.7)


def tom(f=120, d=0.55):
    t = _t(d)
    return _finish(_sweep(t, f * 1.5, f, 18) * np.exp(-t * 7) + _noise(t) * np.exp(-t * 250) * 0.2)


def clap(d=0.35):
    t = _t(d)
    env = np.zeros_like(t)
    for i, off in enumerate((0.0, 0.011, 0.023)):
        m = t >= off
        env[m] += np.exp(-(t[m] - off) * 120) * (0.8 if i < 2 else 1.0)
    tail = np.exp(-t * 16) * 0.35
    return _finish(_lp(_hp(_noise(t)), 3) * (env + tail))


def crash(d=1.8):
    t = _t(d)
    metal = sum(np.sin(2 * np.pi * f * t) for f in (3217, 4093, 5379, 6841, 8123)) * 0.1
    return _finish((_hp(_noise(t), 2) * 0.6 + metal) * np.exp(-t * 2.4), 0.8)


def ride(d=1.2):
    t = _t(d)
    metal = sum(np.sin(2 * np.pi * f * t) * a for f, a in
                ((2800, 1.0), (3560, 0.8), (4810, 0.7), (6210, 0.5), (7400, 0.4)))
    ping = np.sin(2 * np.pi * 520 * t) * np.exp(-t * 14) * 0.6
    return _finish((metal * np.exp(-t * 5) * 0.4 + _hp(_noise(t)) * np.exp(-t * 14) * 0.5 + ping), 0.75)


def cowbell(d=0.45):
    t = _t(d)
    sq = lambda f: np.sign(np.sin(2 * np.pi * f * t))
    return _finish(_lp(sq(562) + sq(845), 4) * np.exp(-t * 11), 0.8)


def zap(d=0.35):
    t = _t(d)
    return _finish(_sweep(t, 2400, 180, 9) * np.exp(-t * 7))


def laser(d=0.45):
    t = _t(d)
    return _finish(_sweep(t, 1200, 90, 5) * np.exp(-t * 4) * (1 + 0.4 * np.sin(2 * np.pi * 38 * t)))


def synth_bass(f=55, d=0.7):
    t = _t(d)
    saw = 2 * ((t * f) % 1.0) - 1
    return _finish(_lp(saw, 12) * np.exp(-t * 4.5))


def click(d=0.06, f=1800):
    t = _t(d)
    return _finish(np.sin(2 * np.pi * f * t) * np.exp(-t * 90))


def piano(f, d=1.6):
    t = _t(d)
    out = np.zeros_like(t)
    for h in range(1, 8):
        out += (1.0 / h ** 1.25) * np.sin(2 * np.pi * f * h * t + 0.3 * h) * np.exp(-t * (2.2 + 1.7 * h))
    out *= np.minimum(1.0, t / 0.004)
    return _finish(out, 0.8)


def conga(f=330, d=0.4):
    t = _t(d)
    return _finish(_sweep(t, f * 1.25, f, 40) * np.exp(-t * 11) + _noise(t) * np.exp(-t * 300) * 0.25)


def shaker(d=0.2):
    t = _t(d)
    env = (t / 0.03) * np.exp(-t * 18)
    return _finish(_hp(_noise(t), 2) * env, 0.6)


def tambourine(d=0.5):
    t = _t(d)
    jingles = sum(np.sin(2 * np.pi * f * t) for f in (5200, 6400, 7700, 9100)) * 0.12
    return _finish((_hp(_noise(t), 2) * 0.7 + jingles) * np.exp(-t * 11) * (1 + np.sin(2 * np.pi * 45 * t) * 0.3), 0.7)


def woodblock(f=880, d=0.18):
    t = _t(d)
    return _finish((np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.5 * t)) * np.exp(-t * 38))


def triangle(d=1.4):
    t = _t(d)
    return _finish(sum(np.sin(2 * np.pi * f * t) * a for f, a in ((2750, 1), (5100, .6), (8200, .35))) * np.exp(-t * 3.2), 0.7)


# --------------------------------------------------------------------------- #
# Kits
# --------------------------------------------------------------------------- #
class PadSpec:
    def __init__(self, name: str, synth: Callable[[], np.ndarray], color: Tuple[int, int, int], icon: str):
        self.name, self.synth, self.color, self.icon = name, synth, color, icon


NOTE_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C"]
KEY_COLORS = [(255, 190, 90), (255, 170, 110), (255, 150, 140), (245, 140, 175),
              (225, 135, 210), (200, 140, 240), (170, 150, 250), (140, 165, 255)]

KITS: Dict[str, Dict] = {
    "Acoustic": {"layout": "grid", "pads": [
        PadSpec("Kick", lambda: kick(), (60, 120, 255), "drum"),
        PadSpec("Snare", lambda: snare(), (60, 210, 255), "drum"),
        PadSpec("Hi-Hat", lambda: hat(), (120, 235, 140), "hat"),
        PadSpec("Open Hat", lambda: hat(0.45, 9), (170, 235, 90), "hat"),
        PadSpec("Tom", lambda: tom(150), (255, 170, 90), "drum"),
        PadSpec("Floor Tom", lambda: tom(95), (255, 120, 120), "drum"),
        PadSpec("Crash", lambda: crash(), (255, 230, 120), "cymbal"),
        PadSpec("Ride", lambda: ride(), (230, 140, 255), "cymbal"),
    ]},
    "Electro": {"layout": "grid", "pads": [
        PadSpec("808 Kick", lambda: kick808(), (255, 90, 180), "drum"),
        PadSpec("E-Snare", lambda: e_snare(), (255, 190, 60), "drum"),
        PadSpec("Clap", lambda: clap(), (190, 120, 255), "pad"),
        PadSpec("E-Hat", lambda: hat(0.06, 70), (110, 240, 200), "hat"),
        PadSpec("Zap", lambda: zap(), (90, 200, 255), "wave"),
        PadSpec("Laser", lambda: laser(), (120, 140, 255), "wave"),
        PadSpec("Bass", lambda: synth_bass(), (255, 130, 90), "wave"),
        PadSpec("Click", lambda: click(), (160, 255, 130), "pad"),
    ]},
    "Piano": {"layout": "keys", "pads": [
        PadSpec(n, (lambda f=f: piano(f)), KEY_COLORS[i], "key")
        for i, (n, f) in enumerate(zip(NOTE_NAMES, NOTE_FREQS))
    ]},
    "Percussion": {"layout": "grid", "pads": [
        PadSpec("Conga Hi", lambda: conga(400), (70, 170, 255), "drum"),
        PadSpec("Conga Lo", lambda: conga(280), (80, 130, 255), "drum"),
        PadSpec("Bongo", lambda: conga(520, 0.28), (110, 220, 255), "drum"),
        PadSpec("Cowbell", lambda: cowbell(), (255, 200, 80), "bell"),
        PadSpec("Shaker", lambda: shaker(), (140, 240, 150), "pad"),
        PadSpec("Tambourine", lambda: tambourine(), (240, 150, 255), "cymbal"),
        PadSpec("Wood Block", lambda: woodblock(), (255, 150, 110), "pad"),
        PadSpec("Triangle", lambda: triangle(), (200, 230, 255), "bell"),
    ]},
}
KIT_NAMES = list(KITS)


def write_wav(path: str, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def load_sounds() -> Dict[str, List[pygame.mixer.Sound]]:
    """Synthesize any missing wav into sounds/<Kit>/ and load them.
    Drop your own .wav with the same filename to override a sound."""
    out: Dict[str, List[pygame.mixer.Sound]] = {}
    for kit, cfg in KITS.items():
        folder = os.path.join(SOUND_DIR, kit)
        os.makedirs(folder, exist_ok=True)
        sounds = []
        for i, pad in enumerate(cfg["pads"]):
            path = os.path.join(folder, f"{i + 1}_{safe(pad.name)}.wav")
            if not os.path.exists(path):
                write_wav(path, pad.synth())
            sounds.append(pygame.mixer.Sound(path))
        out[kit] = sounds
    return out


# --------------------------------------------------------------------------- #
# Hand tracking (supports both the legacy `solutions` API and the newer Tasks API)
# --------------------------------------------------------------------------- #
class HandTracker:
    def __init__(self) -> None:
        self.mode = None
        self.error = ""
        self._ts = 0
        if mp is None:
            self.error = "mediapipe not installed"
            return
        try:
            if hasattr(mp, "solutions"):
                self._hands = mp.solutions.hands.Hands(
                    max_num_hands=2, model_complexity=0,
                    min_detection_confidence=0.6, min_tracking_confidence=0.5)
                self.mode = "solutions"
            else:
                self._init_tasks()
                self.mode = "tasks"
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.mode = None

    def _init_tasks(self) -> None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "hand_landmarker.task")
        if not os.path.exists(path):
            import urllib.request
            print("Downloading hand model (one time, ~7 MB)...")
            urllib.request.urlretrieve(MODEL_URL, path)
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    @property
    def available(self) -> bool:
        return self.mode is not None

    def process(self, bgr) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Returns [(label, [(x, y) * 21])] with x, y normalised 0..1."""
        if not self.available:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = []
        if self.mode == "solutions":
            res = self._hands.process(rgb)
            if res.multi_hand_landmarks:
                for i, lm in enumerate(res.multi_hand_landmarks):
                    label = f"h{i}"
                    if res.multi_handedness:
                        label = res.multi_handedness[i].classification[0].label
                    hands.append((label, [(p.x, p.y) for p in lm.landmark]))
        else:
            self._ts = max(self._ts + 1, int(time.time() * 1000))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self._landmarker.detect_for_video(img, self._ts)
            for i, lm in enumerate(res.hand_landmarks):
                label = res.handedness[i][0].category_name if res.handedness else f"h{i}"
                hands.append((label, [(p.x, p.y) for p in lm]))
        return hands

    def close(self) -> None:
        try:
            (self._hands if self.mode == "solutions" else self._landmarker).close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def text(img, s, pos, scale=0.6, color=(240, 240, 240), thick=1, center=False, shadow=True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = pos
    if center:
        x -= tw // 2
        y += th // 2
    if shadow:
        cv2.putText(img, s, (x + 1, y + 2), FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def rrect(img, p1, p2, color, r=18, thick=-1):
    """Rounded rectangle; thick=-1 fills."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thick < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, cv2.LINE_AA)
        for (cx, cy), a in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                            ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thick, cv2.LINE_AA)


def mix(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


def draw_icon(img, kind, cx, cy, size, color):
    t = 2
    if kind == "drum":
        cv2.circle(img, (cx, cy), size, color, t, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(size * 0.62), color, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(size * 0.22), color, -1, cv2.LINE_AA)
    elif kind == "hat":
        for dy, k in ((-6, 1.0), (6, 0.9)):
            cv2.ellipse(img, (cx, cy + dy), (size, size // 3), 0, 0, 360, color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy + 10), (cx, cy + size + 8), color, t, cv2.LINE_AA)
    elif kind == "cymbal":
        cv2.ellipse(img, (cx, cy), (size + 8, size // 3), -12, 0, 360, color, t, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.line(img, (cx, cy + 4), (cx, cy + size + 8), color, t, cv2.LINE_AA)
    elif kind == "bell":
        pts = np.array([[cx - size // 2, cy + size // 2], [cx - size // 3, cy - size // 2],
                        [cx + size // 3, cy - size // 2], [cx + size // 2, cy + size // 2]], np.int32)
        cv2.polylines(img, [pts], True, color, t, cv2.LINE_AA)
    elif kind == "wave":
        pts = np.array([[cx - size + i * 4, cy + int(math.sin(i * 0.55) * size * 0.45)]
                        for i in range(int(size / 2) + 1)], np.int32)
        cv2.polylines(img, [pts], False, color, t, cv2.LINE_AA)
    elif kind == "pad":
        rrect(img, (cx - size, cy - size // 2), (cx + size, cy + size // 2), color, 8, t)
        for dx in (-size // 2, 0, size // 2):
            cv2.circle(img, (cx + dx, cy), 3, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #
class Pad:
    def __init__(self, spec: PadSpec, rect: Tuple[int, int, int, int], key: str):
        self.spec, self.rect, self.key = spec, rect, key
        self.energy = 0.0

    def contains(self, x, y) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @property
    def center(self):
        x1, y1, x2, y2 = self.rect
        return (x1 + x2) // 2, (y1 + y2) // 2


class App:
    def __init__(self, camera: int = 0, use_camera: bool = True, fullscreen: bool = False):
        os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0")
        pygame.mixer.pre_init(SR, -16, 1, 256)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(40)
        print("Preparing sounds (first run synthesizes wav files into ./sounds)...")
        self.sounds = load_sounds()
        self.metro_hi, self.metro_lo = self._metro_sounds()

        self.kit_idx = 0
        self.pads: List[Pad] = []
        self.tabs: List[Tuple[int, int, int, int]] = []
        self._layout()
        self.master = 0.9
        self._dt = 0.033
        self.hits = 0
        self.toast, self.toast_t = "", 0.0
        self.show_help = False
        self.fullscreen = fullscreen

        self.particles: List[list] = []   # x, y, vx, vy, life, color
        self.ripples: List[list] = []     # x, y, t0, color
        self.trails: Dict[Tuple[str, str], deque] = {}
        self.finger_state: Dict[Tuple[str, str], dict] = {}
        self.dwell: Dict[int, float] = {}
        self.clicks: List[Tuple[int, int]] = []
        self.hit_log: deque = deque(maxlen=6)

        # metronome / looper
        self.bpm, self.metro_on, self.last_beat, self.beat_no = 100, False, 0.0, 0
        self.rec_state = "idle"           # idle | rec | play
        self.loop_events: List[Tuple[float, int, int, float]] = []
        self.loop_t0, self.loop_len, self.loop_phase = 0.0, 0.0, 0.0

        self.tracker = HandTracker() if use_camera else None
        self.cap = None
        if use_camera:
            self.cap = cv2.VideoCapture(camera)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            if not self.cap.isOpened():
                print(f"Could not open camera {camera}; using keyboard/mouse mode.")
                self.cap = None
        if use_camera and self.tracker and not self.tracker.available:
            print(f"Hand tracking unavailable ({self.tracker.error}); using keyboard/mouse mode.")
        self.say("Welcome! Tap the air above a pad - or press A S D F J K L ;")

    # ----- setup ----------------------------------------------------------- #
    @staticmethod
    def _metro_sounds():
        hi, lo = click(0.05, 2400), click(0.05, 1500)
        mk = lambda a: pygame.mixer.Sound(buffer=(a * 0.6 * 32767).astype("<i2").tobytes())
        return mk(hi), mk(lo)

    def _layout(self) -> None:
        kit = KITS[KIT_NAMES[self.kit_idx]]
        specs = kit["pads"]
        top, bottom, margin, gap = TOP_H + 70, H - BOT_H - 18, 36, 16
        self.pads = []
        if kit["layout"] == "keys":
            kw = (W - 2 * margin - gap * 7) // 8
            for i, s in enumerate(specs):
                x = margin + i * (kw + gap)
                self.pads.append(Pad(s, (x, top + 20, x + kw, bottom), PAD_KEYS[i]))
        else:
            cw = (W - 2 * margin - gap * 3) // 4
            ch = (bottom - top - gap) // 2
            for i, s in enumerate(specs):
                r, c = divmod(i, 4)
                x, y = margin + c * (cw + gap), top + r * (ch + gap)
                self.pads.append(Pad(s, (x, y, x + cw, y + ch), PAD_KEYS[i]))
        self.tabs = []
        x = 300
        for _ in KIT_NAMES:
            self.tabs.append((x, 12, x + 150, 52))
            x += 160

    def say(self, msg: str, secs: float = 2.5) -> None:
        self.toast, self.toast_t = msg, time.time() + secs

    def set_kit(self, idx: int) -> None:
        if idx == self.kit_idx:
            return
        self.kit_idx = idx
        self._layout()
        self.say(f"Kit: {KIT_NAMES[idx]}")

    # ----- audio / feedback -------------------------------------------------- #
    def trigger(self, pad_i: int, vol: float, pos: Optional[Tuple[int, int]] = None,
                color=None, record: bool = True, kit_idx: Optional[int] = None) -> None:
        kit_idx = self.kit_idx if kit_idx is None else kit_idx
        kit_name = KIT_NAMES[kit_idx]
        ch = self.sounds[kit_name][pad_i].play()
        if ch:
            ch.set_volume(max(0.05, min(1.0, vol * self.master)))
        if kit_idx != self.kit_idx:
            return
        pad = self.pads[pad_i]
        pad.energy = 1.0
        px, py = pos or pad.center
        col = color or pad.spec.color
        self.ripples.append([px, py, time.time(), col])
        for _ in range(int(10 + 16 * vol)):
            a, s = np.random.uniform(0, 2 * math.pi), np.random.uniform(80, 260) * (0.5 + vol)
            self.particles.append([px, py, math.cos(a) * s, math.sin(a) * s - 60, 1.0, col])
        self.particles = self.particles[-500:]
        self.hits += 1
        self.hit_log.appendleft((pad.spec.name, time.time(), col))
        if record and self.rec_state == "rec":
            self.loop_events.append((time.time() - self.loop_t0, kit_idx, pad_i, vol))

    def toggle_record(self) -> None:
        now = time.time()
        if self.rec_state == "idle":
            self.loop_events, self.loop_t0, self.rec_state = [], now, "rec"
            self.say("Recording... play something, press R to loop it")
        elif self.rec_state == "rec":
            self.loop_len = max(1.0, now - self.loop_t0)
            if self.loop_events:
                self.rec_state, self.loop_t0, self.loop_phase = "play", now, 0.0
                self.say(f"Looping {len(self.loop_events)} hits - C clears")
            else:
                self.rec_state = "idle"
                self.say("Nothing recorded")
        else:
            self.rec_state = "idle"
            self.say("Loop stopped")

    def clear_loop(self) -> None:
        self.loop_events, self.rec_state = [], "idle"
        self.say("Loop cleared")

    def update_loop(self, now: float) -> None:
        if self.rec_state != "play":
            return
        phase = (now - self.loop_t0) % self.loop_len
        prev = self.loop_phase
        for t, k, p, v in self.loop_events:
            hit = (prev < t <= phase) if phase >= prev else (t > prev or t <= phase)
            if hit:
                self.trigger(p, v, record=False, kit_idx=k)
        self.loop_phase = phase

    def update_metronome(self, now: float) -> None:
        if not self.metro_on:
            return
        if now - self.last_beat >= 60.0 / self.bpm:
            self.last_beat = now
            (self.metro_hi if self.beat_no % 4 == 0 else self.metro_lo).set_volume(0.5)
            (self.metro_hi if self.beat_no % 4 == 0 else self.metro_lo).play()
            self.beat_no += 1

    # ----- input ------------------------------------------------------------- #
    def on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))

    def handle_clicks(self) -> None:
        for x, y in self.clicks:
            for i, (x1, y1, x2, y2) in enumerate(self.tabs):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.set_kit(i)
                    break
            else:
                for i, pad in enumerate(self.pads):
                    if pad.contains(x, y):
                        self.trigger(i, 0.85, (x, y))
                        break
        self.clicks.clear()

    def handle_key(self, k: int) -> bool:
        if k in (-1, 255):
            return True
        if k in (27, ord("q")):
            return False
        ch = chr(k).lower() if 0 <= k < 128 else ""
        if ch.isdigit() and 1 <= int(ch) <= len(KIT_NAMES):
            self.set_kit(int(ch) - 1)
        elif ch in PAD_KEYS:
            self.trigger(PAD_KEYS.index(ch), 0.85)
        elif ch == "m":
            self.metro_on = not self.metro_on
            self.beat_no = 0
            self.say("Metronome " + ("on" if self.metro_on else "off"))
        elif ch in "+=":
            self.bpm = min(220, self.bpm + 5)
            self.say(f"Tempo {self.bpm} BPM")
        elif ch in "-_":
            self.bpm = max(40, self.bpm - 5)
            self.say(f"Tempo {self.bpm} BPM")
        elif ch == "r":
            self.toggle_record()
        elif ch == "c":
            self.clear_loop()
        elif ch == "[":
            self.master = max(0.1, self.master - 0.1)
            self.say(f"Volume {int(self.master * 100)}%")
        elif ch == "]":
            self.master = min(1.0, self.master + 0.1)
            self.say(f"Volume {int(self.master * 100)}%")
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "f":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        return True

    # ----- hand logic -------------------------------------------------------- #
    def process_hands(self, hands, now: float, frame) -> None:
        seen = set()
        hover_tab = None
        for label, lms in hands:
            pts = [(int(x * W), int(y * H)) for x, y in lms]
            self.draw_hand(frame, label, pts)
            for finger, tid in FINGERS.items():
                key = (label, finger)
                seen.add(key)
                px, py = pts[tid]
                ny = lms[tid][1]
                st = self.finger_state.setdefault(key, {"y": ny, "t": now, "armed": True, "last": 0.0})
                dt = now - st["t"]
                vy = (ny - st["y"]) / dt if 0.001 < dt < 0.25 else 0.0
                st["y"], st["t"] = ny, now
                self.trails.setdefault(key, deque(maxlen=14)).append((px, py))

                if finger == "index":
                    for i, (x1, y1, x2, y2) in enumerate(self.tabs):
                        if x1 <= px <= x2 and y1 - 12 <= py <= y2 + 24:
                            hover_tab = i
                if vy < REARM_SPEED:
                    st["armed"] = True
                for i, pad in enumerate(self.pads):
                    if pad.contains(px, py):
                        pad.energy = max(pad.energy, 0.25)  # hover glow
                        if (vy > TAP_SPEED and st["armed"] and now - st["last"] > MIN_RETRIGGER):
                            vol = min(1.0, 0.4 + (vy - TAP_SPEED) / 3.0)
                            self.trigger(i, vol, (px, py), FINGER_COLORS[finger])
                            st["armed"], st["last"] = False, now
                        break
        for key in list(self.finger_state):
            if key not in seen:
                self.finger_state.pop(key)
                self.trails.pop(key, None)

        for i in range(len(self.tabs)):
            if i == hover_tab:
                self.dwell[i] = self.dwell.get(i, 0.0) + self._dt
                if self.dwell[i] >= DWELL_TIME:
                    self.set_kit(i)
                    self.dwell[i] = 0.0
            else:
                self.dwell[i] = 0.0

    def draw_hand(self, frame, label, pts) -> None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (30, 30, 30), -1, cv2.LINE_AA)
        for finger, tid in FINGERS.items():
            col = FINGER_COLORS[finger]
            trail = self.trails.get((label, finger), [])
            for j in range(1, len(trail)):
                cv2.line(frame, trail[j - 1], trail[j], mix((0, 0, 0), col, j / len(trail)),
                         max(1, j // 4), cv2.LINE_AA)
            cv2.circle(frame, pts[tid], 15, col, 2, cv2.LINE_AA)
            cv2.circle(frame, pts[tid], 8, col, -1, cv2.LINE_AA)

    # ----- rendering ---------------------------------------------------------- #
    def render(self, frame, now: float, dt: float, fps: float, hands_found: bool, mode_note: str) -> None:
        # pads (single alpha blend)
        overlay = frame.copy()
        for pad in self.pads:
            x1, y1, x2, y2 = pad.rect
            base = (28, 28, 36) if KITS[KIT_NAMES[self.kit_idx]]["layout"] != "keys" else (70, 70, 80)
            fill = mix(base, pad.spec.color, min(1.0, pad.energy) * 0.85)
            rrect(overlay, (x1, y1), (x2, y2), fill, 20)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        for pad in self.pads:
            x1, y1, x2, y2 = pad.rect
            e = max(0.0, pad.energy)
            col = pad.spec.color
            rrect(frame, (x1, y1), (x2, y2), mix(col, (255, 255, 255), min(1.0, e) * 0.6), 20, 2 + int(3 * e))
            cx, cy = pad.center
            keys = KITS[KIT_NAMES[self.kit_idx]]["layout"] == "keys"
            if keys:
                text(frame, pad.spec.name, (cx, y2 - 60), 1.0, (255, 255, 255), 2, True)
                text(frame, pad.key.upper(), (cx, y2 - 24), 0.6, col, 1, True)
            else:
                draw_icon(frame, pad.spec.icon, cx, cy - 12, min(46, (x2 - x1) // 5), mix(col, (255, 255, 255), e * 0.5))
                text(frame, pad.spec.name, (cx, y2 - 34), 0.7, (255, 255, 255), 1, True)
                rrect(frame, (x1 + 12, y1 + 12), (x1 + 40, y1 + 38), col, 7, 1)
                text(frame, pad.key.upper(), (x1 + 26, y1 + 25), 0.5, col, 1, True, False)
            pad.energy = max(0.0, pad.energy - dt * 3.2)

        # ripples + particles
        self.ripples = [r for r in self.ripples if now - r[2] < 0.5]
        for x, y, t0, col in self.ripples:
            k = (now - t0) / 0.5
            cv2.circle(frame, (x, y), int(18 + 90 * k), mix(col, (0, 0, 0), k), max(1, int(4 * (1 - k))), cv2.LINE_AA)
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 420 * dt
            p[4] -= dt * 1.8
            if p[4] > 0:
                alive.append(p)
                cv2.circle(frame, (int(p[0]), int(p[1])), max(1, int(4 * p[4])), mix((0, 0, 0), p[5], p[4]), -1, cv2.LINE_AA)
        self.particles = alive

        self.draw_top_bar(frame, fps, hands_found)
        self.draw_bottom_bar(frame, now)
        if now < self.toast_t:
            a = min(1.0, (self.toast_t - now) * 2)
            (tw, _), _ = cv2.getTextSize(self.toast, FONT, 0.75, 1)
            o = frame.copy()
            rrect(o, (W // 2 - tw // 2 - 24, TOP_H + 12), (W // 2 + tw // 2 + 24, TOP_H + 56), (20, 20, 20), 20)
            cv2.addWeighted(o, 0.7 * a, frame, 1 - 0.7 * a, 0, frame)
            text(frame, self.toast, (W // 2, TOP_H + 34), 0.75, (255, 255, 255), 1, True)
        if mode_note:
            text(frame, mode_note, (W // 2, TOP_H + 90), 0.6, (120, 200, 255), 1, True)
        if self.show_help:
            self.draw_help(frame)

    def draw_top_bar(self, frame, fps, hands_found) -> None:
        o = frame.copy()
        cv2.rectangle(o, (0, 0), (W, TOP_H), (12, 12, 18), -1)
        cv2.addWeighted(o, 0.8, frame, 0.2, 0, frame)
        text(frame, "AIR", (24, 34), 0.9, (255, 210, 90), 2, False)
        text(frame, "SYMPHONY", (82, 34), 0.9, (255, 255, 255), 2, False)
        for i, (x1, y1, x2, y2) in enumerate(self.tabs):
            sel = i == self.kit_idx
            rrect(frame, (x1, y1), (x2, y2), (255, 210, 90) if sel else (90, 90, 100), 14, -1 if sel else 1)
            text(frame, f"{i + 1}  {KIT_NAMES[i]}", ((x1 + x2) // 2, (y1 + y2) // 2 - 1), 0.55,
                 (20, 20, 20) if sel else (220, 220, 220), 1, True, False)
            d = self.dwell.get(i, 0.0)
            if d > 0:
                cv2.ellipse(frame, ((x1 + x2) // 2, y2 + 10), (9, 9), -90, 0, int(360 * d / DWELL_TIME),
                            (255, 210, 90), 3, cv2.LINE_AA)
        dot = (90, 230, 120) if hands_found else (110, 110, 120)
        cv2.circle(frame, (W - 190, 32), 7, dot, -1, cv2.LINE_AA)
        text(frame, "hand" if hands_found else "no hand", (W - 175, 32), 0.5, (200, 200, 200), 1, False)
        text(frame, f"{fps:4.0f} fps", (W - 90, 32), 0.5, (150, 150, 160), 1, False)

    def draw_bottom_bar(self, frame, now) -> None:
        y0 = H - BOT_H
        o = frame.copy()
        cv2.rectangle(o, (0, y0), (W, H), (12, 12, 18), -1)
        cv2.addWeighted(o, 0.8, frame, 0.2, 0, frame)
        cy = y0 + BOT_H // 2
        # metronome
        text(frame, f"{self.bpm} BPM", (28, cy - 12), 0.65, (255, 255, 255), 1, False)
        text(frame, "M metronome  +/- tempo", (28, cy + 16), 0.42, (150, 150, 160), 1, False)
        for i in range(4):
            lit = self.metro_on and (self.beat_no - 1) % 4 == i and now - self.last_beat < 0.15
            col = (255, 210, 90) if lit else ((90, 90, 100) if self.metro_on else (50, 50, 58))
            cv2.circle(frame, (230 + i * 26, cy), 8, col, -1, cv2.LINE_AA)
        # loop
        lx = 380
        if self.rec_state == "rec":
            cv2.circle(frame, (lx, cy), 10, (60, 60, 255) if int(now * 2) % 2 else (40, 40, 140), -1, cv2.LINE_AA)
            text(frame, f"REC {now - self.loop_t0:4.1f}s  ({len(self.loop_events)} hits)", (lx + 22, cy - 2), 0.6, (120, 120, 255), 1, False)
        elif self.rec_state == "play":
            cv2.ellipse(frame, (lx, cy), (11, 11), -90, 0, int(360 * self.loop_phase / self.loop_len), (90, 230, 120), 3, cv2.LINE_AA)
            text(frame, f"LOOP {self.loop_len:.1f}s", (lx + 22, cy - 2), 0.6, (110, 230, 140), 1, False)
        else:
            text(frame, "R  record loop", (lx, cy - 2), 0.6, (200, 200, 200), 1, False)
        # volume
        vx = 640
        text(frame, "VOL", (vx, cy - 2), 0.5, (170, 170, 180), 1, False)
        rrect(frame, (vx + 44, cy - 7), (vx + 164, cy + 7), (60, 60, 70), 6, -1)
        rrect(frame, (vx + 44, cy - 7), (vx + 44 + int(120 * self.master), cy + 7), (255, 210, 90), 6, -1)
        # hits
        text(frame, f"{self.hits}", (840, cy - 10), 0.8, (255, 255, 255), 1, False)
        text(frame, "hits", (840, cy + 16), 0.42, (150, 150, 160), 1, False)
        # recent hits
        x = 925
        for name, t, col in list(self.hit_log)[:3]:
            age = now - t
            if age < 3:
                text(frame, name, (x, cy), 0.5, mix((60, 60, 60), col, max(0.0, 1 - age / 3)), 1, False)
                x += 100
        text(frame, "H = help", (W - 110, cy), 0.5, (170, 170, 180), 1, False)

    def draw_help(self, frame) -> None:
        o = frame.copy()
        cv2.rectangle(o, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(o, 0.82, frame, 0.18, 0, frame)
        lines = [
            ("HOW TO PLAY", (255, 210, 90), 1.0),
            ("Hold your hand in view, then flick a fingertip DOWN over a pad - faster = louder.", (255, 255, 255), 0.6),
            ("Both hands work, every finger is its own drumstick.", (255, 255, 255), 0.6),
            ("Hover your index finger over a kit tab for 1 second to switch kits.", (255, 255, 255), 0.6),
            ("", None, 0), ("1-4  switch kit        A S D F J K L ;  play pads / click pads", (200, 220, 255), 0.6),
            ("M  metronome        + / -  tempo        [ / ]  volume", (200, 220, 255), 0.6),
            ("R  record, R again = loop it        C  clear loop", (200, 220, 255), 0.6),
            ("F  fullscreen        H  close help        Q / Esc  quit", (200, 220, 255), 0.6),
            ("", None, 0), ("Tip: drop your own .wav into sounds/<Kit>/ with the same filename to replace any sound.", (150, 150, 160), 0.5),
        ]
        y = 150
        for s, col, sc in lines:
            if s:
                text(frame, s, (W // 2, y), sc, col, 1 if sc < 1 else 2, True)
            y += 46

    # ----- main loop --------------------------------------------------------- #
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        if self.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        last, fps = time.time(), 30.0
        self._dt = 0.033
        try:
            while True:
                now = time.time()
                self._dt = dt = max(1e-3, min(0.1, now - last))
                last = now
                fps = fps * 0.9 + (1 / dt) * 0.1

                hands: list = []
                note = ""
                if self.cap is not None:
                    ok, cam = self.cap.read()
                    if not ok:
                        print("Camera feed lost.")
                        break
                    cam = cv2.flip(cam, 1)
                    if cam.shape[1] != W or cam.shape[0] != H:
                        cam = cv2.resize(cam, (W, H))
                    frame = cv2.convertScaleAbs(cam, alpha=0.7, beta=-5)
                    if self.tracker and self.tracker.available:
                        hands = self.tracker.process(cam)
                    else:
                        note = "Hand tracking unavailable - use the keyboard or click the pads"
                else:
                    frame = np.zeros((H, W, 3), np.uint8)
                    frame[:] = (24, 18, 18)
                    note = "No camera - use the keyboard (A S D F J K L ;) or click the pads"

                self.handle_clicks()
                self.process_hands(hands, now, frame)
                self.update_metronome(now)
                self.update_loop(now)
                self.render(frame, now, dt, fps, bool(hands), note)

                cv2.imshow(WINDOW, frame)
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.tracker:
                self.tracker.close()
            cv2.destroyAllWindows()
            pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="AirSymphony - play instruments in mid-air")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--no-camera", action="store_true", help="keyboard/mouse only")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()
    App(args.camera, not args.no_camera, args.fullscreen).run()


if __name__ == "__main__":
    main()
