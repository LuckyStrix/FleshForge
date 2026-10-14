"""The scan command: its bookkeeping, the video path, and one full run through COLMAP.

The full run needs COLMAP with a CUDA GPU and takes a few minutes. It is skipped when no COLMAP
is found, and with FLESHFORGE_SKIP_COLMAP=1.
"""
import os
import shutil
import subprocess
from pathlib import Path

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
    run.finish("masks", masks={"images": 3})
    again = scan.Run(tmp_path)
    assert again.done("sfm") and again.state["masks"] == {"images": 3}
    again.redo_from("undistort")
    assert again.done("sfm") and not again.done("undistort") and not again.done("masks")


def finished_run(tmp_path, args):
    run = scan.Run(tmp_path / "run")
    run.dir.mkdir(exist_ok=True)
    run.match_inputs(scan.stage_inputs(args, Path(args.input)))
    for stage in scan.STAGES[:-1]:
        run.finish(stage)
    run.state["model"] = {"folder": "colmap/sparse/0", "models": 1}
    run.save()
    return scan.Run(run.dir)


def test_new_inputs_redo_the_stages_that_depend_on_them(tmp_path):
    video = tmp_path / "alex.mp4"
    video.write_bytes(b"alex")
    parse = scan.build_parser().parse_args
    base = [str(video), "-o", str(tmp_path / "run")]

    run = finished_run(tmp_path, parse(base))
    run.match_inputs(scan.stage_inputs(parse(base), video))          # same again: nothing to redo
    assert run.done("stereo") and "model" in run.state

    run.match_inputs(scan.stage_inputs(parse(base + ["--dense-size", "1600"]), video))
    assert run.done("sfm") and not run.done("undistort") and "model" in run.state

    run = finished_run(tmp_path, parse(base))
    run.match_inputs(scan.stage_inputs(parse(base + ["--dense", "fine"]), video))
    assert run.done("masks") and not run.done("stereo") and not run.done("fuse")

    run = finished_run(tmp_path, parse(base))
    video.write_bytes(b"sam, a different person")                  # same name, new video
    run.match_inputs(scan.stage_inputs(parse(base), video))
    assert not run.done("frames") and not run.done("sfm") and "model" not in run.state

    sam = tmp_path / "sam.mp4"
    sam.write_bytes(b"sam")
    run = finished_run(tmp_path, parse(base))
    run.match_inputs(scan.stage_inputs(parse([str(sam), "-o", str(tmp_path / "run")]), sam))
    assert not run.done("frames")


def test_a_copied_or_moved_video_is_the_same_video(tmp_path):
    video = tmp_path / "orbit.mp4"
    video.write_bytes(bytes(range(256)) * 20_000)                 # 5 MB: three separate samples
    parse = scan.build_parser().parse_args
    run = finished_run(tmp_path, parse([str(video), "-o", str(tmp_path / "run")]))
    moved = tmp_path / "elsewhere" / "copy of orbit.mp4"
    moved.parent.mkdir()
    shutil.copy(video, moved)                                      # new path, new mtime
    assert run.match_inputs(scan.stage_inputs(parse([str(moved), "-o", str(run.dir)]), moved)) is None
    assert run.done("stereo")

    changed = bytearray(video.read_bytes())
    changed[len(changed) // 2] ^= 1                               # one byte in the middle
    moved.write_bytes(changed)
    assert run.match_inputs(scan.stage_inputs(parse([str(moved), "-o", str(run.dir)]), moved)) == "frames"


def test_a_photo_folder_is_its_images_only(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (photos / name).write_bytes(name.encode() * 100)
    before = scan.source_identity(photos)
    (photos / "a.AAE").write_text("sidecar")
    (photos / "desktop.ini").write_text("[.ShellClassInfo]")
    assert scan.source_identity(photos) == before
    (photos / "b.jpg").write_bytes(b"B.jpg" * 100)                # edited, same size
    assert scan.source_identity(photos) != before


def test_an_old_run_folder_keeps_its_stages(tmp_path):
    video = tmp_path / "orbit.mp4"
    video.write_bytes(b"orbit")
    run = scan.Run(tmp_path / "run")
    run.dir.mkdir()
    for stage in scan.STAGES[:-1]:
        run.finish(stage)                                          # no "inputs": made before tracking
    args = scan.build_parser().parse_args([str(video), "-o", str(run.dir), "--dense", "fine"])
    assert run.match_inputs(scan.stage_inputs(args, video)) is None
    assert scan.Run(run.dir).done("stereo") and scan.Run(run.dir).state["inputs"]["stereo"] == {"dense": "fine"}


def test_a_redo_is_on_disk_before_its_files_go(tmp_path):
    # the crash case: inputs changed, the frames are deleted, and the run dies before any stage
    # finishes. The next run must not think the old frames are still there
    video = tmp_path / "orbit.mp4"
    video.write_bytes(b"alex")
    parse = scan.build_parser().parse_args
    finished_run(tmp_path, parse([str(video), "-o", str(tmp_path / "run")]))
    video.write_bytes(b"sam")
    scan.Run(tmp_path / "run").match_inputs(scan.stage_inputs(parse([str(video), "-o", "x"]), video))
    assert not scan.Run(tmp_path / "run").done("frames")          # read back from state.json


def test_scan_log_records_settings_and_stage_times(tmp_path):
    shown = []
    log = scan.ScanLog(tmp_path / "scan.log", shown.append)
    args = scan.build_parser().parse_args(["in.mp4", "-o", "out", "--dense", "fine"])
    log.header(args, cm.Colmap(str(tmp_path / "no-colmap.exe")))
    with log.stage("frames"):
        log.line("frames: kept 3")
    with pytest.raises(RuntimeError), log.stage("sfm"):
        raise RuntimeError("mapper fell over")
    log.summary()
    text = (tmp_path / "scan.log").read_text(encoding="utf-8")
    assert '"dense": "fine"' in text and "frames: kept 3" in text and shown == ["frames: kept 3"]
    assert "stage frames:" in text and "stage sfm: FAILED" in text and "mapper fell over" in text
    assert "reused from an earlier run: sfm, undistort" in text and "frames" in log.timings


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

    # a rerun with the same options only redoes the mesh: the saved poses are read back
    # (model_converter), and no COLMAP step that computes anything runs again
    log = tmp_path / "run" / "colmap.log"
    before = len(log.read_text(encoding="utf-8", errors="replace"))
    again = scan.scan(args, report_line=lambda *_: None)
    started = [line for line in log.read_text(encoding="utf-8", errors="replace")[before:].splitlines()
               if line.startswith("===== ")]
    assert started and all(" model_converter " in line for line in started)
    assert again["registered"] == stats["registered"]
    assert set(stats["stage_seconds"]) == set(scan.STAGES) and set(again["stage_seconds"]) == {"mesh"}
    text = (tmp_path / "run" / "scan.log").read_text(encoding="utf-8")
    assert text.count("===== fleshforge-scan") == 2 and "stage stereo:" in text
    assert "reused from an earlier run: frames, sfm, undistort, masks, stereo, fuse" in text


@pytest.mark.skipif(not colmap_available(), reason="no COLMAP (or FLESHFORGE_SKIP_COLMAP=1)")
def test_sparse_from_a_pair_list(tmp_path, monkeypatch):
    # the path taken above EXHAUSTIVE_MAX_FRAMES, run on a small set by lowering the limit
    synth.write_orbit(tmp_path / "run" / "frames", count=32)
    monkeypatch.setattr(cm, "EXHAUSTIVE_MAX_FRAMES", 10)
    colmap, _ = cm.choose("auto")
    root = tmp_path / "run"
    sparse = cm.sparse(colmap, root, root / "frames", root / "colmap", 32, True, root / "colmap.log",
                       report=lambda *_: None)
    assert (root / "colmap" / "pairs.txt").exists()
    model, _ = cm.best_model(colmap, root, sparse, root / "colmap.log")
    assert len(model.poses) >= 30
