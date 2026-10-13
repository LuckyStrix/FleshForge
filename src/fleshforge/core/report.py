"""Judge a scan and draw it, so a bad one is refused here and not found out on the printer."""
from __future__ import annotations

import numpy as np

MIN_REGISTERED = 0.80        # share of frames COLMAP must place
MIN_ARC_DEG = 270.0          # less than this and the back of the head is invented
MAX_REPROJ_PX = 1.5
MAX_EMPTY_MASKS = 0.20

# (name, column for u, sign, column for v, sign, column for depth, sign) in the glTF frame:
# x right, y up, z towards the viewer of the face
VIEWS = [("front", 0, 1, 1, -1, 2, 1), ("right side", 2, -1, 1, -1, 0, 1),
         ("back", 0, -1, 1, -1, 2, -1), ("top", 0, 1, 2, 1, 1, 1)]


def check_sparse(frames: int, registered: int, models: int) -> tuple[list, list]:
    """(errors, warnings) once the camera poses are known, before the slow dense stage."""
    errors, warnings = [], []
    share = registered / frames if frames else 0.0
    if share < MIN_REGISTERED:
        errors.append(
            f"only {registered} of {frames} frames could be placed ({share:.0%}; need "
            f"{MIN_REGISTERED:.0%}). Usual causes: motion blur, the subject moving, or a plain "
            "background with nothing to track")
    if models > 1:
        warnings.append(f"the frames fell into {models} separate reconstructions that could not be joined; "
                        "the largest is used")
    return errors, warnings


def check_scan(stats: dict) -> tuple[list, list]:
    """(errors, warnings) for a finished scan. `stats` is the report dictionary."""
    errors, warnings = check_sparse(stats["frames"], stats["registered"], stats.get("models", 1))
    if stats.get("arc_deg", 360.0) < MIN_ARC_DEG:
        warnings.append(f"the camera went {stats['arc_deg']:.0f} degrees round the subject; under "
                        f"{MIN_ARC_DEG:.0f} the unseen side is made up by the mesher")
    if stats.get("reproj_error", 0.0) > MAX_REPROJ_PX:
        warnings.append(f"reprojection error is {stats['reproj_error']:.2f} px (over {MAX_REPROJ_PX}); "
                        "the camera poses are loose, so expect soft detail")
    if not stats.get("face_found", True):
        warnings.append("no frontal face was found, so the model's front is a guess: check the preview")
    mesh = stats.get("mesh", {})
    if mesh and not mesh.get("closed", True):
        warnings.append("the mesh is not closed; a slicer may leave holes")
    masks = stats.get("masks") or {}
    if masks.get("images") and masks.get("empty", 0) / masks["images"] > MAX_EMPTY_MASKS:
        warnings.append(f"{masks['empty']} of {masks['images']} frames had no person found in them")
    return errors, warnings


def signed_volume(vertices, faces) -> float:
    """Positive when the triangles wind counter-clockwise seen from outside."""
    v = np.asarray(vertices, float)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def render_views(points, colors, normals, size=360) -> np.ndarray:
    """(2*size, 4*size, 3) uint8: front, side, back and top, in colour and as shaded shape.
    `points` are surface samples in the glTF frame; a z-buffered splat, no renderer needed."""
    points, colors, normals = (np.asarray(a, float) for a in (points, colors, normals))
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = float((points.max(axis=0) - points.min(axis=0)).max()) or 1.0
    p = (points - centre) / extent                       # within +-0.5
    sheet = np.full((2 * size, 4 * size, 3), 24, dtype=np.uint8)
    for col, (_, ui, us, vi, vs, di, ds) in enumerate(VIEWS):
        u = np.clip(((us * p[:, ui]) * 0.9 + 0.5) * (size - 2), 0, size - 2).astype(int)
        v = np.clip(((vs * p[:, vi]) * 0.9 + 0.5) * (size - 2), 0, size - 2).astype(int)
        order = np.argsort(ds * p[:, di])                # far first: the nearest is written last
        towards = np.zeros(3)
        towards[di] = ds
        shade = 0.25 + 0.75 * np.clip(normals @ towards, 0.0, 1.0)
        layers = (np.clip(colors, 0, 1) * 255, np.repeat((shade * 215)[:, None], 3, axis=1))
        for row, rgb in enumerate(layers):
            panel = sheet[row * size:(row + 1) * size, col * size:(col + 1) * size]
            for dv in (0, 1):
                for du in (0, 1):
                    panel[v[order] + dv, u[order] + du] = rgb[order].astype(np.uint8)
    return sheet


def save_preview(path, points, colors, normals, size=360) -> None:
    from PIL import Image, ImageDraw
    im = Image.fromarray(render_views(points, colors, normals, size))
    draw = ImageDraw.Draw(im)
    for col, view in enumerate(VIEWS):
        draw.text((col * size + 8, 6), view[0], fill=(235, 235, 235))
    im.save(path)
