"""fleshforge-check: is everything a scan needs installed and working?

Prints one line per requirement and exits non-zero if a required one is missing.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys

from fleshforge.core import colmap as cm
from fleshforge.core import frames as fr


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="fleshforge-check", description=__doc__.split("\n")[0])


def checks() -> list[tuple[str, bool, bool, str]]:
    """(name, ok, required, detail) per requirement."""
    out = []

    def add(name, ok, detail, required=True):
        out.append((name, bool(ok), required, detail))

    add("python", (3, 10) <= sys.version_info[:2] <= (3, 12),
        f"{sys.version_info.major}.{sys.version_info.minor} (need 3.10 to 3.12: open3d has no wheels for 3.13+)")
    for mod in ("numpy", "scipy", "PIL", "cv2", "open3d", "rembg", "onnxruntime"):
        try:
            m = importlib.import_module(mod)
            add(mod, True, getattr(m, "__version__", "installed"))
        except Exception as exc:
            add(mod, False, f"import failed: {exc}")
    for tool in ("ffmpeg", "ffprobe"):
        path = fr.find_tool(tool)
        add(tool, path, path or "not found; install ffmpeg (winget install Gyan.FFmpeg)")

    native, docker = cm.find_native(), cm.find_docker()
    for label, colmap in (("colmap (local)", native), ("colmap (Docker)", docker)):
        if colmap is None:
            continue
        version = colmap.version()
        add(label, bool(version) and colmap.has_cuda(),
            f"{version or 'did not start'}  [{colmap.where}]" + ("" if colmap.has_cuda() else "  NO CUDA"))
    if native is None and docker is None:
        add("colmap", False, f"none found; run scripts/setup_windows.ps1, or `docker pull {cm.DEFAULT_IMAGE}`")
    elif docker is None:
        add("colmap (Docker)", False, f"image {cm.docker_image()} not pulled; only needed if the "
                                      "local COLMAP's dense stereo crashes on your GPU", required=False)

    smi = shutil.which("nvidia-smi")
    gpu = ""
    if smi:
        done = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True)
        gpu = done.stdout.strip().splitlines()[0] if done.returncode == 0 and done.stdout.strip() else ""
    add("NVIDIA GPU", gpu, gpu or "nvidia-smi found no GPU; dense stereo needs one")

    stackforge = shutil.which("stackforge-paint")
    add("stackforge-paint", stackforge, stackforge or "not on PATH (only needed to print the result)",
        required=False)
    return out


def main(argv=None) -> int:
    build_parser().parse_args(argv)
    bad = 0
    for name, ok, required, detail in checks():
        tag = "ok" if ok else ("MISSING" if required else "optional")
        print(f"[{tag:>8}] {name}: {detail}")
        bad += (not ok) and required
    print("ready" if not bad else f"{bad} requirement(s) missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
