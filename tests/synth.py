"""A synthetic orbit for tests: a noise-textured ball on a textured floor, ray-traced in numpy.

It stands in for a head so the whole pipeline (COLMAP included) can run with no person in it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BALL_RADIUS = 1.0
FLOOR_Y = -1.25
FLOOR_RADIUS = 5.0
ORBIT_RADIUS = 4.2


def _noise(points, seed, octaves=(3, 7, 15)):
    """3-channel block noise in 0..1 at (N, 3) points: random-coloured cells at several sizes.
    Hard cell edges on purpose: they give a feature detector corners to hold on to, which
    smooth noise does not."""
    rng = np.random.default_rng(seed)
    n = 64
    total, weight = np.zeros((len(points), 3)), 0.0
    for i, freq in enumerate(octaves):
        grid = rng.random((n, n, n, 3))
        cell = np.floor(points * freq + 0.37 * (i + 1)).astype(int) % n
        amp = 0.7 ** i
        total += amp * grid[cell[:, 0], cell[:, 1], cell[:, 2]]
        weight += amp
    return total / weight


def ball_colour(points):
    """The ball's surface colour (sRGB 0..1) at points on it: reddish, well textured."""
    c = _noise(points, seed=1)
    return np.clip(np.array([0.25, 0.05, 0.05]) + c * np.array([0.75, 0.6, 0.5]), 0, 1)


def render(angle_deg, elevation_deg=12.0, width=800, height=600, focal=780.0):
    """One frame from a camera on the orbit, looking at the ball. Returns (H, W, 3) uint8."""
    a, e = np.radians(angle_deg), np.radians(elevation_deg)
    eye = ORBIT_RADIUS * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    u, v = np.meshgrid(np.arange(width) - width / 2 + 0.5, np.arange(height) - height / 2 + 0.5)
    d = (forward[None] * focal + u.reshape(-1, 1) * right[None] + v.reshape(-1, 1) * down[None])
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    img = np.tile(np.array([0.55, 0.62, 0.70]), (len(d), 1))          # plain wall
    # floor disc
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (FLOOR_Y - eye[1]) / d[:, 1]
    hit = eye[None] + t[:, None] * d
    on_floor = (t > 0) & np.isfinite(t) & (np.hypot(hit[:, 0], hit[:, 2]) < FLOOR_RADIUS)
    img[on_floor] = 0.15 + 0.8 * _noise(hit[on_floor] * 0.5, seed=2)
    # ball
    b = d @ eye
    disc = b * b - (eye @ eye - BALL_RADIUS ** 2)
    on_ball = disc > 0
    tb = -b[on_ball] - np.sqrt(disc[on_ball])
    p = eye[None] + tb[:, None] * d[on_ball]
    light = np.clip(p @ np.array([0.3, 0.8, 0.5]) / BALL_RADIUS, 0, 1)
    img[on_ball] = ball_colour(p) * (0.7 + 0.3 * light[:, None])
    return (np.clip(img, 0, 1).reshape(height, width, 3) * 255).astype(np.uint8)


def write_orbit(folder: Path, count=40, **kwargs) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = folder / f"{i + 1:04d}.jpg"
        # two heights, alternating, so the top of the ball is seen too
        img = render(360.0 * i / count, elevation_deg=8.0 if i % 2 else 32.0, **kwargs)
        Image.fromarray(img).save(path, quality=95)
        paths.append(path)
    return paths
