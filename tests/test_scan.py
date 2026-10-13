"""The scan command: its bookkeeping, the video path, and one full run through COLMAP.

The full run needs COLMAP with a CUDA GPU and takes a few minutes. It is skipped when no COLMAP
is found, and with FLESHFORGE_SKIP_COLMAP=1.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from fleshforge.core import colmap as cm
from fleshforge.core import frames
from fleshforge.tools import scan

from . import synth
from .test_core import edges_are_closed, read_glb


def test_parser_defaults():
    args = scan.build_parser().parse_args(["in.mp4", "-o", "out"])
    assert (args.mask, args.dense, args.colmap, args.frames) == ("person", "fast", "auto", 120)


def test_run_state_remembers_and_redoes(tmp_path):
    run = scan.Run(tmp_path)
    for stage in ("frames", "sfm", "undistort"):
        run.finish(stage)
    run.finish("masks", mask_mode="off")
    again = scan.Run(tmp_path)
    assert again.done("sfm") and again.state["mask_mode"] == "off"
    again.redo_from("undistort")
    assert again.done("sfm") and not again.done("undistort") and not again.done("masks")


def test_missing_input_is_reported(tmp_path):
    args = scan.build_parser().parse_args([str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as err:
        scan.scan(args)
    assert "no such file" in str(err.value)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_frames_from_a_video(tmp_path):
    src = tmp_path / "src"
    synth.write_orbit(src, count=24, width=160, height=120, focal=150.0)
    video = tmp_path / "orbit.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i", str(src / "%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    facts = frames.frames_from_video(str(video), tmp_path / "frames", 20, 120, report=lambda *_: None)
    kept = frames.list_images(tmp_path / "frames")
    assert facts["kept"] == len(kept) == 20
    assert not (tmp_path / "frames_all").exists()


def colmap_available():
    if os.environ.get("FLESHFORGE_SKIP_COLMAP") == "1":
        return False
    try:
        cm.choose("auto")
    except cm.ColmapError:
        return False
    return True


@pytest.mark.skipif(not colmap_available(), reason="no COLMAP (or FLESHFORGE_SKIP_COLMAP=1)")
def test_full_scan_of_a_synthetic_ball(tmp_path):
    synth.write_orbit(tmp_path / "in", count=32)
    args = scan.build_parser().parse_args(
        [str(tmp_path / "in"), "-o", str(tmp_path / "run"), "--mask", "off", "--dense-size", "640"])
    stats = scan.scan(args, report_line=lambda *_: None)

    assert stats["registered"] >= 30 and stats["errors"] == []
    assert stats["arc_deg"] > 300
    assert stats["mesh"]["closed"]
    doc, arrays = read_glb(tmp_path / "run" / "head.glb")
    v, f = arrays["POSITION"], arrays["indices"].reshape(-1, 3)
    assert edges_are_closed(f.astype(np.int64))
    assert np.isclose((v.max(axis=0) - v.min(axis=0)).max(), 0.25, atol=1e-4)
    # a ball: as wide as it is deep, and Y is up, so the cut at the floor is on the low side
    size = v.max(axis=0) - v.min(axis=0)
    assert abs(size[0] - size[2]) / size[0] < 0.08
    # the ball is reddish (see synth.ball_colour); linear red well above blue
    assert arrays["COLOR_0"][:, 0].mean() > 1.5 * arrays["COLOR_0"][:, 2].mean()
    assert (tmp_path / "run" / "preview.png").exists()
