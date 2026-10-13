"""Where the subject is, which way is up, and which way the face points.

A COLMAP reconstruction sits in an arbitrary frame at an arbitrary scale. Everything here is
worked out from the camera poses: the cameras all look at the subject, a phone held upright
has its "up" along the world's, and the frames in which a face detector sees a frontal face
were taken from in front.

Output frame is glTF's: Y up, the face looking along +Z.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
NOMINAL_SIZE_M = 0.25        # true scale is unknown from video; printing rescales anyway


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def look_at_point(centres, forwards) -> np.ndarray:
    """The point closest (least squares) to every camera's optical axis."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(np.asarray(centres, float), np.asarray(forwards, float)):
        d = unit(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    if np.linalg.cond(A) > 1e8:
        raise ValueError("the cameras all look the same way; the video has to go round the head")
    return np.linalg.solve(A, b)


def horizontal(v, up) -> np.ndarray:
    """`v` with its component along `up` removed."""
    v, up = np.asarray(v, float), unit(up)
    return v - (v @ up) * up


def basis(up, front) -> np.ndarray:
    """Rotation whose rows are the new x, y, z axes: y = up, z = front made level."""
    y = unit(up)
    z = unit(horizontal(front, y))
    x = np.cross(y, z)
    return np.vstack([x, y, z])


def arc_degrees(centres, centre, up) -> float:
    """How much of the way round the subject the cameras went, 0..360."""
    up = unit(up)
    ref = unit(horizontal(centres[0] - centre, up))
    side = np.cross(up, ref)
    ang = np.sort(np.degrees([np.arctan2(horizontal(c - centre, up) @ side, horizontal(c - centre, up) @ ref)
                              for c in centres]) % 360.0)
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    return float(360.0 - gaps.max())


def frontalness(face) -> float:
    """0..1 from a YuNet detection row: 1 when the nose sits midway between the eyes."""
    (rx, ry), (lx, ly), nx = face[4:6], face[6:8], face[8]
    eye = float(np.hypot(lx - rx, ly - ry))
    if eye < 1e-6:
        return 0.0
    off = abs(nx - (rx + lx) / 2.0) / eye
    return float(max(0.0, 1.0 - off / 0.35))


def yunet_model() -> Path:
    cache = Path(os.environ.get("FLESHFORGE_CACHE") or Path.home() / ".cache" / "fleshforge")
    path = cache / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(YUNET_URL, timeout=30) as r:
            data = r.read()
        if len(data) < 100_000:
            raise OSError("the face model download was truncated")
        path.write_bytes(data)
    return path


def detect_faces(image_paths, report=print) -> dict:
    """{image name: (score, frontalness)} for images where one face is found."""
    import cv2
    try:
        detector = cv2.FaceDetectorYN.create(str(yunet_model()), "", (320, 320), 0.8)
    except Exception as exc:
        report(f"  warning: face detector unavailable ({exc}); cannot tell front from back")
        return {}
    found = {}
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        k = 640.0 / max(h, w)
        if k < 1:
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            continue
        face = max(faces, key=lambda f: f[2] * f[3])       # the subject is the biggest face
        found[Path(path).name] = (float(face[14]), frontalness(face))
    return found


def front_direction(poses, centre, up, faces: dict):
    """Level direction from the subject towards the cameras that saw the face head-on.
    None when no frame has a usable face."""
    total, weight = np.zeros(3), 0.0
    for p in poses:
        if p.name not in faces:
            continue
        score, frontal = faces[p.name]
        w = score * frontal
        total += w * unit(horizontal(p.centre - centre, up))
        weight += w
    if weight < 1e-6 or np.linalg.norm(total) < 0.2 * weight:
        return None
    return unit(total)


def to_gltf_frame(points, centre, R) -> np.ndarray:
    return (np.asarray(points, float) - centre) @ R.T
