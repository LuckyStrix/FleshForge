"""Video -> a set of sharp, evenly spread still frames.

ffmpeg does the decoding (it applies the phone's rotation flag, so frames come out upright;
OpenCV's reader does not do that reliably). More candidates than needed are extracted, then the
sharpest one in each time window is kept: a hand-held orbit has motion blur in bursts, and one
blurred frame costs more matches than a slightly uneven spacing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CANDIDATES_PER_FRAME = 3      # candidates extracted per frame kept


def find_tool(name: str) -> str | None:
    """ffmpeg / ffprobe: $FLESHFORGE_FFMPEG / $FLESHFORGE_FFPROBE, else PATH."""
    return os.environ.get(f"FLESHFORGE_{name.upper()}") or shutil.which(name)


def video_duration(video: str) -> float:
    """Length in seconds, from ffprobe."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise SystemExit("ffprobe not found (it ships with ffmpeg); install ffmpeg or set FLESHFORGE_FFPROBE")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise SystemExit(f"{video}: ffprobe could not read a duration ({out.stderr.strip() or 'no output'})")


def extract_candidates(video: str, out_dir: Path, count: int, max_size: int) -> list[Path]:
    """Write about `count` evenly spaced frames, long side at most `max_size` px."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found; install it or set FLESHFORGE_FFMPEG")
    duration = video_duration(video)
    if duration <= 0:
        raise SystemExit(f"{video}: zero-length video")
    out_dir.mkdir(parents=True, exist_ok=True)
    # never upscale; -2 keeps the other side even, which the JPEG encoder wants
    scale = (f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'")
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf", f"fps={count / duration:.6f},{scale}",
           "-q:v", "2", str(out_dir / "%06d.jpg")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {video}:\n{done.stderr.strip()}")
    return sorted(out_dir.glob("*.jpg"))


def sharpness(path: Path) -> float:
    """Variance of the Laplacian: higher is sharper. Measured at a fixed width so it compares
    across videos of different resolution."""
    import cv2
    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return 0.0
    h, w = grey.shape
    if w > 640:
        grey = cv2.resize(grey, (640, max(1, round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_sharpest(scores, count: int) -> list[int]:
    """Indices of the sharpest entry in each of `count` equal windows, in order.

    With no more entries than `count`, everything is kept."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if count <= 0:
        return []
    if n <= count:
        return list(range(n))
    edges = np.linspace(0, n, count + 1).round().astype(int)
    return [int(lo + np.argmax(scores[lo:hi])) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def frames_from_video(video: str, frames_dir: Path, count: int, max_size: int, report=print) -> dict:
    """Fill `frames_dir` with the chosen frames. Returns what was done, for the report."""
    scratch = frames_dir.parent / "frames_all"
    if scratch.exists():
        shutil.rmtree(scratch)
    candidates = extract_candidates(video, scratch, count * CANDIDATES_PER_FRAME, max_size)
    if len(candidates) < 20:
        raise SystemExit(f"{video}: only {len(candidates)} frames could be read; a scan needs a "
                         "30-60 s orbit of the head")
    scores = [sharpness(p) for p in candidates]
    keep = select_sharpest(scores, count)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(keep, 1):
        shutil.copyfile(candidates[i], frames_dir / f"{n:06d}.jpg")
    shutil.rmtree(scratch)
    kept = [scores[i] for i in keep]
    report(f"frames: kept {len(keep)} of {len(candidates)} candidates "
           f"(sharpness median {np.median(kept):.0f}, lowest {min(kept):.0f})")
    return {"source": "video", "candidates": len(candidates), "kept": len(keep),
            "sharpness_median": float(np.median(kept)), "sharpness_min": float(min(kept))}


def frames_from_folder(folder: str, frames_dir: Path, max_size: int, report=print) -> dict:
    """Copy a folder of photos in, shrunk to `max_size` and renamed, as JPEG."""
    from PIL import Image, ImageOps
    images = list_images(Path(folder))
    if len(images) < 20:
        raise SystemExit(f"{folder}: {len(images)} images; a scan needs at least 20")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n, src in enumerate(images, 1):
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        im.thumbnail((max_size, max_size), Image.LANCZOS)
        im.save(frames_dir / f"{n:06d}.jpg", quality=95)
    report(f"frames: {len(images)} photos from {folder}")
    return {"source": "folder", "candidates": len(images), "kept": len(images)}
