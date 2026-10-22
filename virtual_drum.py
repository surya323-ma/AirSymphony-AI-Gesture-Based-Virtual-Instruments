import importlib
import math
import os
import struct
import subprocess
import sys
import time
import wave
from typing import Dict, List, Optional, Tuple


REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pygame": "pygame",
}


def ensure_dependencies() -> None:
    """Install any third-party dependencies that are missing."""
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing missing dependency: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


ensure_dependencies()

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import pygame  # type: ignore


SAMPLE_RATE = 44100
SOUND_DURATION = 0.35  # seconds
SOUND_FOLDER = os.path.join(os.path.dirname(__file__), "sounds")
FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
TAP_THRESHOLD = 0.045
COOLDOWN = 0.25

def ensure_sound_folder() -> None:
    os.makedirs(SOUND_FOLDER, exist_ok=True)


def generate_sine_wave(path: str, frequency: float) -> None:
    """Generate a simple sine wave and store it as a .wav file."""
    sample_count = int(SAMPLE_RATE * SOUND_DURATION)
    amplitude = 32767
    frames = []
    for n in range(sample_count):
        value = int(amplitude * 0.5 * math.sin(2 * math.pi * frequency * (n / SAMPLE_RATE)))
        frames.append(struct.pack('<h', value))
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(frames))


def prepare_sound_assets() -> Dict[str, str]:
    ensure_sound_folder()
    sound_plan = {
        "thumb": (180, "Kick"),
        "index": (320, "Snare"),
        "middle": (400, "HiHat"),
        "ring": (500, "Tom"),
        "pinky": (620, "Clap"),
    }
    finger_to_sound = {}
    for finger, (frequency, label) in sound_plan.items():
        filename = f"{finger}_{label}.wav"
        path = os.path.join(SOUND_FOLDER, filename)
        if not os.path.exists(path):
            generate_sine_wave(path, frequency)
        finger_to_sound[finger] = path
    return finger_to_sound


FINGER_COLORS = {
    "thumb": (255, 100, 100),
    "index": (255, 210, 30),
    "middle": (80, 220, 120),
    "ring": (100, 150, 255),
    "pinky": (200, 120, 255),
}

ZONE_ART_COLORS = {
    "Kick": (255, 140, 105),
    "Snare": (255, 230, 90),
    "HiHat": (120, 230, 150),
    "Tom": (120, 170, 255),
    "Clap": (230, 150, 255),
}


def init_audio(sound_files: Dict[str, str]) -> Dict[str, pygame.mixer.Sound]:
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.init()
    loaded = {}
    for finger, path in sound_files.items():
        loaded[finger] = pygame.mixer.Sound(path)
    return loaded


def draw_instrument_art(frame, zone_bounds: Tuple[int, int, int, int], name: str) -> None:
    """Render simple instrument silhouettes so lanes feel like real pads."""
    x1, x2, y1, y2 = zone_bounds
    color = ZONE_ART_COLORS.get(name, (220, 220, 220))
    center_x = (x1 + x2) // 2
    center_y = y1 + (y2 - y1) // 2
    width = x2 - x1
    height = y2 - y1
    thickness = 3

    if name == "Kick":
        radius = min(width, height) // 5
        cv2.circle(frame, (center_x, center_y + 40), radius + 30, color, thickness + 1)
        cv2.circle(frame, (center_x, center_y + 40), radius, color, thickness)
        cv2.line(frame, (center_x + radius + 10, center_y - 20), (center_x + radius + 60, center_y - 80), color, thickness)
    elif name == "Snare":
        drum_w = int(width * 0.5)
        drum_h = int(height * 0.25)
        top_left = (center_x - drum_w // 2, center_y - drum_h // 2)
        bottom_right = (center_x + drum_w // 2, center_y + drum_h // 2)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        cv2.line(frame, (top_left[0], top_left[1]), (top_left[0], top_left[1] - 40), color, thickness)
        cv2.line(frame, (bottom_right[0], top_left[1]), (bottom_right[0], top_left[1] - 40), color, thickness)
    elif name == "HiHat":
        top_radius = int(width * 0.25)
        cv2.line(frame, (center_x, y1 + 40), (center_x, center_y + 60), color, thickness)
        cv2.ellipse(frame, (center_x, center_y - 10), (top_radius, top_radius // 3), 0, 0, 360, color, thickness)
        cv2.ellipse(frame, (center_x, center_y + 20), (top_radius + 15, (top_radius + 15) // 3), 0, 0, 360, color, thickness)
    elif name == "Tom":
        radius = int(width * 0.25)
        cv2.circle(frame, (center_x, center_y), radius, color, thickness)
        cv2.circle(frame, (center_x, center_y), radius - 15, color, 1)
        cv2.line(frame, (center_x - radius, center_y - 30), (center_x - radius - 30, center_y - 90), color, thickness)
        cv2.line(frame, (center_x + radius, center_y - 30), (center_x + radius + 30, center_y - 90), color, thickness)
    elif name == "Clap":
        palm_width = int(width * 0.35)
        palm_height = int(height * 0.18)
        left_palm = [(center_x - palm_width, center_y - palm_height), (center_x - 20, center_y + palm_height)]
        right_palm = [(center_x + 20, center_y - palm_height), (center_x + palm_width, center_y + palm_height)]
        cv2.rectangle(frame, left_palm[0], left_palm[1], color, thickness)
        cv2.rectangle(frame, right_palm[0], right_palm[1], color, thickness)
        cv2.line(frame, left_palm[1], right_palm[0], color, thickness)


def draw_zones(frame, width: int, height: int, zone_names: List[str]) -> List[Tuple[int, int, int, int, str]]:
    zone_width = width // len(zone_names)
    overlay = frame.copy()
    zones: List[Tuple[int, int, int, int, str]] = []
    for index, name in enumerate(zone_names):
        x1 = index * zone_width
        x2 = x1 + zone_width
        cv2.rectangle(overlay, (x1, 0), (x2, height), (50, 50, 50), -1)
        zones.append((x1, x2, 0, height, name))

    alpha = 0.15
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for x1, x2, y1, y2, name in zones:
        draw_instrument_art(frame, (x1, x2, y1, y2), name)
    return zones


def find_zone(zones, x_pixel: int) -> Optional[str]:
    for x1, x2, _, _, name in zones:
        if x1 <= x_pixel <= x2:
            return name
    return None


def main() -> None:
    sound_files = prepare_sound_assets()
    sounds = init_audio(sound_files)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Track downward movement and cooldown per finger to infer taps.
    prev_finger_y: Dict[str, float] = {finger: 0.0 for finger in FINGER_TIP_IDS}
    last_trigger_time: Dict[str, float] = {finger: 0.0 for finger in FINGER_TIP_IDS}
    # Store transient visual pulses so taps feel responsive on screen.
    active_visuals: List[Tuple[int, int, Tuple[int, int, int], float]] = []
    zone_names = [
        "Kick",
        "Snare",
        "HiHat",
        "Tom",
        "Clap",
    ]

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not access webcam feed.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            zones = draw_zones(frame, width, height, zone_names)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for finger, tip_id in FINGER_TIP_IDS.items():
                        landmark = hand_landmarks.landmark[tip_id]
                        x_pixel = int(landmark.x * width)
                        y_pixel = int(landmark.y * height)

                        zone_name = find_zone(zones, x_pixel)
                        if zone_name:
                            cv2.circle(frame, (x_pixel, y_pixel), 10, FINGER_COLORS[finger], -1)

                        delta_y = landmark.y - prev_finger_y[finger]
                        prev_finger_y[finger] = landmark.y
                        recent = time.time() - last_trigger_time[finger]
                        # Downward speed + cooldown imply an intentional tap.
                        if zone_name and delta_y > TAP_THRESHOLD and recent > COOLDOWN:
                            sounds[finger].play()
                            last_trigger_time[finger] = time.time()
                            active_visuals.append((x_pixel, y_pixel, FINGER_COLORS[finger], time.time()))

            # Fade active tap visuals so they animate briefly.
            now = time.time()
            active_visuals = [v for v in active_visuals if now - v[3] < 0.3]
            for x, y, color, start_time in active_visuals:
                radius = int(30 * (1 - ((now - start_time) / 0.3))) + 1
                cv2.circle(frame, (x, y), radius, color, 2)

            cv2.putText(frame, "Virtual Drum/Piano - press 'q' to quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

            cv2.imshow("Virtual Drum Kit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
