"""
Realistic Synthetic Crowd Video Generator for Event Crowd Flow Tracker.

Generates a visually convincing 720p crowd entrance video where people
appear as proper human silhouettes that YOLOv8 (trained on COCO) can detect.

Strategy:
  - Use OpenCV to draw filled, correctly-proportioned human silhouettes with
    realistic skin tones, clothing textures, and motion blur.
  - Draw people at multiple scales to simulate depth.
  - Add camera noise, grain, and realistic lighting gradients.
  - People walk from the bottom edge upward (bottom-to-top crossing direction).
"""

from __future__ import annotations
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + t * (c2[i] - c1[i])) for i in range(3))


def draw_realistic_person(
    frame: np.ndarray,
    cx: int,
    cy_feet: int,
    person_height_px: int,
    shirt_color: Tuple[int, int, int],
    pants_color: Tuple[int, int, int],
    skin_color: Tuple[int, int, int],
    hair_color: Tuple[int, int, int],
    leg_phase: float,
    frame_idx: int,
):
    """Draw a realistic-looking human figure using geometric primitives."""
    h_frame, w_frame = frame.shape[:2]
    ph = person_height_px  # total height in pixels
    
    # Proportions (based on anatomical standards scaled to ph)
    head_h   = int(ph * 0.14)
    head_w   = int(ph * 0.10)
    neck_h   = int(ph * 0.04)
    torso_h  = int(ph * 0.32)
    torso_w  = int(ph * 0.17)
    arm_len  = int(ph * 0.32)
    arm_w    = max(2, int(ph * 0.06))
    leg_len  = int(ph * 0.47)
    leg_w    = max(2, int(ph * 0.07))

    # Key body Y positions
    feet_y   = cy_feet
    hip_y    = feet_y - leg_len
    shoulder_y = hip_y - torso_h
    neck_bot_y = shoulder_y + int(neck_h * 0.3)
    head_cy  = shoulder_y - neck_h - head_h // 2

    # ── Legs ──────────────────────────────────────────────────────────────────
    swing = int(math.sin(leg_phase) * leg_len * 0.25)
    # Left leg (back)
    lleg_pts = np.array([
        [cx - leg_w // 2, hip_y],
        [cx + leg_w // 2, hip_y],
        [cx + leg_w // 2 + swing // 2, feet_y],
        [cx - leg_w // 2 + swing // 2, feet_y],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [lleg_pts], lerp_color(pants_color, (0, 0, 0), 0.15))

    # Right leg (front)
    rleg_pts = np.array([
        [cx - leg_w // 2, hip_y],
        [cx + leg_w // 2, hip_y],
        [cx + leg_w // 2 - swing // 2, feet_y],
        [cx - leg_w // 2 - swing // 2, feet_y],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [rleg_pts], pants_color)

    # Shoes
    shoe_w = int(leg_w * 1.4)
    shoe_h = max(3, int(ph * 0.04))
    cv2.ellipse(frame, (cx + swing // 2, feet_y), (shoe_w, shoe_h), 0, 0, 180, (30, 25, 20), -1)
    cv2.ellipse(frame, (cx - swing // 2, feet_y), (shoe_w, shoe_h), 0, 0, 180, (35, 30, 25), -1)

    # ── Torso ─────────────────────────────────────────────────────────────────
    torso_pts = np.array([
        [cx - torso_w,     hip_y],
        [cx + torso_w,     hip_y],
        [cx + torso_w - 4, shoulder_y],
        [cx - torso_w + 4, shoulder_y],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [torso_pts], shirt_color)

    # Shirt crease/highlight line for depth
    mid_shirt_x = cx
    cv2.line(frame, (mid_shirt_x, shoulder_y + 4), (mid_shirt_x, hip_y - 4),
             lerp_color(shirt_color, (255, 255, 255), 0.12), 1)

    # ── Arms ──────────────────────────────────────────────────────────────────
    arm_swing = int(math.sin(leg_phase + math.pi) * arm_len * 0.18)
    # Left arm
    cv2.line(frame,
             (cx - torso_w + 4, shoulder_y + 6),
             (cx - torso_w - arm_swing - 6, hip_y - 4),
             lerp_color(shirt_color, (0, 0, 0), 0.2),
             arm_w, cv2.LINE_AA)
    # Right arm
    cv2.line(frame,
             (cx + torso_w - 4, shoulder_y + 6),
             (cx + torso_w + arm_swing + 6, hip_y - 4),
             shirt_color,
             arm_w, cv2.LINE_AA)

    # ── Neck ──────────────────────────────────────────────────────────────────
    neck_w = max(3, int(head_w * 0.55))
    cv2.rectangle(frame,
                  (cx - neck_w // 2, head_cy + head_h // 2),
                  (cx + neck_w // 2, neck_bot_y),
                  skin_color, -1)

    # ── Head ──────────────────────────────────────────────────────────────────
    cv2.ellipse(frame, (cx, head_cy), (head_w, head_h), 0, 0, 360, skin_color, -1)

    # Hair (top arc)
    hair_h = int(head_h * 0.55)
    cv2.ellipse(frame, (cx, head_cy - int(head_h * 0.1)),
                (head_w + 1, hair_h), 0, 180, 360, hair_color, -1)

    # Eyes
    eye_y = head_cy - int(head_h * 0.1)
    eye_sep = max(2, int(head_w * 0.35))
    eye_r = max(1, int(head_h * 0.12))
    cv2.circle(frame, (cx - eye_sep, eye_y), eye_r, (20, 20, 30), -1)
    cv2.circle(frame, (cx + eye_sep, eye_y), eye_r, (20, 20, 30), -1)
    # Whites / reflections
    cv2.circle(frame, (cx - eye_sep, eye_y), eye_r, (255, 255, 255), 1)
    cv2.circle(frame, (cx - eye_sep + 1, eye_y - 1), max(1, eye_r // 2), (230, 230, 230), -1)

    # Nose (simple vertical line suggestion)
    nose_y = head_cy + int(head_h * 0.1)
    cv2.line(frame, (cx, nose_y - 2), (cx, nose_y + 3), lerp_color(skin_color, (80, 60, 60), 0.5), 1)

    # Mouth (thin arc)
    mouth_y = head_cy + int(head_h * 0.3)
    mouth_w = max(2, int(head_w * 0.45))
    cv2.ellipse(frame, (cx, mouth_y), (mouth_w, max(1, int(head_h * 0.1))),
                0, 0, 180, lerp_color(skin_color, (80, 50, 50), 0.6), 1)


class AttendeeSprite:
    def __init__(self, width: int, height: int):
        self.cx    = random.randint(80, width - 80)
        self.cy    = random.randint(-50, height // 3)   # start near top
        self.speed = random.uniform(2.2, 4.5)           # px/frame downward
        self.drift = random.uniform(-0.5, 0.5)          # horizontal drift

        # Appearance
        skin_tones = [
            (200, 170, 150), (180, 140, 110), (220, 190, 165),
            (150, 110, 85),  (130, 90,  70),  (210, 180, 155),
        ]
        shirt_hues = [
            (40, 90, 200), (200, 40, 50), (30, 150, 80),
            (180, 140, 30), (100, 50, 180), (30, 180, 200),
            (200, 100, 40), (60, 60, 60), (220, 220, 220),
        ]
        pants_hues = [
            (30, 40, 80), (50, 50, 50), (60, 40, 20),
            (80, 70, 60), (20, 50, 40), (40, 30, 60),
        ]
        hair_choices = [(20, 15, 10), (40, 30, 15), (60, 45, 20), (15, 10, 10), (90, 50, 20)]

        self.skin   = random.choice(skin_tones)
        self.shirt  = random.choice(shirt_hues)
        self.pants  = random.choice(pants_hues)
        self.hair   = random.choice(hair_choices)

        # Size: farther away (smaller scale) or closer (larger scale)
        self.scale  = random.uniform(0.60, 1.15)
        self.leg_phase = random.uniform(0, math.pi * 2)

    @property
    def person_height(self) -> int:
        return int(180 * self.scale)

    def update(self):
        self.cy += self.speed
        self.cx += self.drift
        self.leg_phase += 0.20

    def feet_y(self) -> int:
        return int(self.cy + self.person_height // 2)

    def is_alive(self, frame_h: int) -> bool:
        return self.cy < frame_h + self.person_height

    def bbox_y1(self) -> int:
        return int(self.cy - self.person_height // 2)

    def draw(self, frame: np.ndarray):
        draw_realistic_person(
            frame,
            cx=int(self.cx),
            cy_feet=self.feet_y(),
            person_height_px=self.person_height,
            shirt_color=self.shirt,
            pants_color=self.pants,
            skin_color=self.skin,
            hair_color=self.hair,
            leg_phase=self.leg_phase,
            frame_idx=0,
        )


def generate_crowd_video(
    output_path: str = "sample_entrance.mp4",
    num_frames: int = 540,   # 18 seconds at 30 fps
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    max_onscreen: int = 20,
    spawn_every_n: int = 25,  # new person every N frames
) -> str:
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    sprites: List[AttendeeSprite] = []

    # Pre-built background: convention hall floor
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient from top (darker) to bottom (lighter) to give perspective feel
    for r in range(height):
        t = r / height
        col = lerp_color((28, 32, 40), (70, 75, 90), t)
        bg[r, :] = col

    # Tile grid (floor tiles in perspective)
    for x in range(0, width, 120):
        cv2.line(bg, (x, height // 2), (x + 60, height), (50, 55, 70), 1)
    for y in range(height // 2, height, 80):
        cv2.line(bg, (0, y), (width, y), (50, 55, 70), 1)

    # Ceiling lights (blobs near top)
    for lx in range(160, width, 200):
        cv2.ellipse(bg, (lx, 30), (30, 12), 0, 0, 360, (200, 210, 255), -1)
        cv2.ellipse(bg, (lx, 30), (80, 50), 0, 0, 360, (45, 48, 60), 1)

    # Entry gate arch at ~65% height
    gate_y = int(height * 0.65)
    cv2.rectangle(bg, (0, gate_y - 5), (width, gate_y + 5), (90, 100, 130), -1)
    for gx in range(0, width, width // 4):
        cv2.rectangle(bg, (gx, gate_y - 35), (gx + 20, gate_y + 5), (110, 120, 150), -1)

    # Spawn ahead of the first frame
    for _ in range(min(8, max_onscreen)):
        s = AttendeeSprite(width, height)
        s.cy = random.randint(50, gate_y)  # start in mid-frame immediately
        sprites.append(s)

    for f_idx in range(num_frames):
        frame = bg.copy()

        # Spawn new attendees
        if f_idx % spawn_every_n == 0 and len(sprites) < max_onscreen:
            sprites.append(AttendeeSprite(width, height))

        # Sort back-to-front (lower y_top = farther away = drawn first)
        sprites.sort(key=lambda s: s.bbox_y1())

        # Update & draw
        alive = []
        for sprite in sprites:
            sprite.update()
            sprite.draw(frame)
            if sprite.is_alive(height):
                alive.append(sprite)
        sprites = alive

        # Camera noise grain
        noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
        cv2.add(frame, noise, frame)

        # Vignette effect
        rows, cols = frame.shape[:2]
        k_row = cv2.getGaussianKernel(rows, rows * 0.7)
        k_col = cv2.getGaussianKernel(cols, cols * 0.7)
        kernel = k_row * k_col.T
        mask = kernel / kernel.max()
        vignette = np.dstack([mask] * 3)
        frame = (frame.astype(np.float32) * (0.75 + 0.25 * vignette)).clip(0, 255).astype(np.uint8)

        # Timestamp overlay
        sec = f_idx / fps
        cv2.putText(
            frame,
            f"SECURITY CAM 04  |  ENTRANCE FOYER  |  {sec:05.2f}s  |  REC",
            (18, height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 190, 210),
            1,
            cv2.LINE_AA,
        )
        # REC blink
        if (f_idx // 15) % 2 == 0:
            cv2.circle(frame, (width - 30, height - 28), 5, (0, 0, 220), -1)

        writer.write(frame)

    writer.release()
    return output_path


if __name__ == "__main__":
    out_file = generate_crowd_video("sample_entrance.mp4", num_frames=540, max_onscreen=20, spawn_every_n=20)
    print(f"[OK] Generated realistic synthetic crowd entrance video: {out_file}")
    print(f"     File size: {Path(out_file).stat().st_size / 1024 / 1024:.1f} MB")
