"""Person masks for the dense stage.

The phone moves round a still person, so the room is useful for solving the camera poses and
is left in for that. It is only kept out of the dense cloud: COLMAP's fusion skips pixels
whose mask is black. Masks are made on the undistorted images fusion reads, so they line up
by construction.

COLMAP's naming: the mask for `000001.jpg` is `000001.jpg.png`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = "u2net_human_seg"
ERODE_PX_PER_1000 = 4       # pull the edge in: edge pixels are part background


def person_masks(images_dir: Path, mask_dir: Path, model: str = MODEL, report=print) -> dict:
    """Write one mask per image. Returns coverage numbers for the report."""
    import cv2
    from PIL import Image
    try:
        from rembg import new_session, remove
        session = new_session(model)
    except Exception as exc:      # model download blocked, most often
        raise SystemExit(
            f"could not load the '{model}' segmentation model: {exc}\n"
            "rembg downloads it once (to ~/.rembg/models). If a virus scanner intercepts HTTPS "
            "(Norton does), that download fails: fetch the .onnx by hand from "
            "https://github.com/danielgatis/rembg/releases/tag/v0.0.0, or run with --mask off.")
    mask_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    coverage = []
    for n, path in enumerate(images, 1):
        im = Image.open(path).convert("RGB")
        mask = np.asarray(remove(im, session=session, only_mask=True).convert("L")) > 127
        k = max(1, round(ERODE_PX_PER_1000 * max(im.size) / 1000))
        mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / (path.name + ".png"))
        coverage.append(float(mask.mean()))
        if n % 25 == 0 or n == len(images):
            report(f"  masks: {n}/{len(images)}")
    empty = sum(c < 0.01 for c in coverage)
    return {"model": model, "images": len(images), "coverage_median": float(np.median(coverage)),
            "empty": int(empty)}
