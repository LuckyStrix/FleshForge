"""fleshforge-scan: a phone video of a head -> a vertex-coloured GLB.

    fleshforge-scan orbit.mp4 -o runs/alex

Stages, each kept in the run folder so a rerun picks up where it stopped:
frames -> sfm -> undistort -> masks -> stereo -> fuse -> mesh. `--from STAGE` redoes that
stage and everything after it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from fleshforge.core import colmap as cm
from fleshforge.core import frames as fr
from fleshforge.core import glbwrite, mesh, orient, report

STAGES = ["frames", "sfm", "undistort", "masks", "stereo", "fuse", "mesh"]
SAMPLE_BYTES = 1 << 20      # a file is told apart by its size and three 1 MiB samples


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fleshforge-scan", description=__doc__.split("\n")[0])
    ap.add_argument("input", help="a video of the head (one slow orbit), or a folder of photos")
    ap.add_argument("-o", "--output", required=True, help="run folder; head.glb is written in it")
    ap.add_argument("--frames", type=int, default=120, help="frames to keep from a video")
    ap.add_argument("--max-size", type=int, default=1600, help="long side of the frames, px")
    ap.add_argument("--dense-size", type=int, default=1200,
                    help="long side used for the dense cloud, px; most of the run time scales with it")
    ap.add_argument("--mask", choices=["person", "off"], default="person",
                    help="person = keep only the person in the dense cloud; off = keep the scene")
    ap.add_argument("--crop", type=float, default=0.45,
                    help="keep the cloud within this fraction of the camera distance from the "
                         "point the cameras look at (0 = no crop)")
    ap.add_argument("--poisson-depth", type=int, default=9, help="mesh detail; 10 is finer and noisier")
    ap.add_argument("--smooth", type=float, default=1.5,
                    help="how much depth noise is flattened before meshing: 0 = off, 2 = twice "
                         "as far (loses fine detail)")
    ap.add_argument("--triangles", type=int, default=300_000, help="triangle budget of the GLB")
    ap.add_argument("--from", dest="from_stage", choices=STAGES,
                    help="redo this stage and everything after it")
    ap.add_argument("--dense", choices=sorted(cm.DENSE), default="fast",
                    help="fast = one stereo pass; fine = COLMAP's defaults with the geometric "
                         "consistency pass, several times slower")
    ap.add_argument("--colmap", choices=["auto", "native", "docker"], default="auto",
                    help="which COLMAP to run. auto = the local one, with Docker as the fallback "
                         "for dense stereo")
    ap.add_argument("--no-gpu", action="store_true",
                    help="extract and match features on the CPU (dense stereo still needs CUDA)")
    ap.add_argument("--force", action="store_true", help="carry on past a failed quality check")
    return ap


class Run:
    """The run folder: where each stage's files live, which stages are done, and what each was
    made from. A stage whose inputs differ from last time is redone, with everything after it.
    Every change to the state is written straight away, so a crash part-way through a redone
    stage cannot leave state.json claiming files that are already deleted."""

    def __init__(self, folder: Path):
        self.dir = folder
        self.frames = folder / "frames"
        self.colmap = folder / "colmap"
        self.dense = folder / "colmap" / "dense"
        self.masks = folder / "colmap" / "dense" / "masks"
        self.fused = folder / "fused.ply"
        self.log = folder / "colmap.log"
        self.state_file = folder / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def done(self, stage: str) -> bool:
        return stage in self.state.get("done", [])

    def save(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=1))

    def finish(self, stage: str, **facts) -> None:
        self.state.setdefault("done", [])
        if stage not in self.state["done"]:
            self.state["done"].append(stage)
        self.state.update(facts)
        self.save()

    def redo_from(self, stage: str) -> None:
        later = STAGES[STAGES.index(stage):]
        self.state["done"] = [s for s in self.state.get("done", []) if s not in later]
        if "sfm" in later:
            self.state.pop("model", None)
        if self.state_file.parent.is_dir():
            self.save()

    def match_inputs(self, inputs: dict) -> str | None:
        """Redo from the first stage whose `inputs` differ from the ones it was made with.
        Returns that stage, or None.

        A folder made before inputs were recorded has done stages and no inputs: those are
        taken as made with the current ones, since redoing them could cost an hour of stereo."""
        made_with = self.state.get("inputs")
        changed = None
        if made_with is not None:
            _migrate_source(made_with, inputs)
            for stage in STAGES:
                if stage in inputs and made_with.get(stage) != inputs[stage]:
                    changed = stage
                    self.redo_from(stage)
                    break
        self.state["inputs"] = inputs
        if self.state_file.parent.is_dir():
            self.save()
        return changed


    def missing_output(self, use_masks: bool) -> str | None:
        """The first done stage whose files are gone (deleted to free space, say), or None."""
        made = {"frames": self.frames, "sfm": self.colmap / "sparse", "undistort": self.dense / "images",
                "masks": self.masks if use_masks else None, "stereo": self.dense / "stereo" / "depth_maps",
                "fuse": self.fused}
        for stage in STAGES:
            if self.done(stage) and made.get(stage) is not None and not made[stage].exists():
                return stage
        return None


def _migrate_source(made_with: dict, inputs: dict) -> None:
    """Folders from 9f1826a stored the source as {path, bytes, modified}. If that still names
    a file of that size with today's identity, it is the same source: no hour of stereo again."""
    old = (made_with.get("frames") or {}).get("source")
    if not isinstance(old, dict):
        return
    src = Path(old.get("path", ""))
    try:
        same = src.is_file() and src.stat().st_size == old.get("bytes")
    except OSError:
        same = False
    if same and file_identity(src) == inputs["frames"]["source"]:
        made_with["frames"] = {**made_with["frames"], "source": inputs["frames"]["source"]}


def file_identity(path: Path, cache: dict | None = None) -> str:
    """Size plus a hash of the start, middle and end: the same video copied or moved keeps it,
    a different one does not. Reading three samples keeps this instant on a 4K video.

    `cache` maps a path to [size, mtime, identity]: an unchanged stat means no reading at all,
    which matters for a folder of a few hundred photos on every rerun."""
    st = path.stat()
    key, stamp = str(path.resolve()), [st.st_size, st.st_mtime_ns]
    if cache is not None and cache.get(key, [None, None])[:2] == stamp:
        return cache[key][2]
    size = st.st_size
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as fh:
        for at in sorted({0, max(0, size // 2 - SAMPLE_BYTES // 2), max(0, size - SAMPLE_BYTES)}):
            fh.seek(at)
            digest.update(fh.read(SAMPLE_BYTES))
    found = digest.hexdigest()[:20]
    if cache is not None:
        cache[key] = stamp + [found]
    return found


def source_identity(src: Path, cache: dict | None = None) -> str:
    """What a frames stage was made from, by content: a video, or the images in a folder (other
    files there, such as .AAE sidecars, are not read and do not count)."""
    if not src.is_dir():
        return file_identity(src, cache)
    digest = hashlib.sha256()
    for image in fr.list_images(src):
        digest.update(f"{image.name}:{file_identity(image, cache)};".encode())
    return digest.hexdigest()[:20]


def stage_inputs(args, src: Path, cache: dict | None = None) -> dict:
    """Per stage, the options its output depends on. Stages not listed depend only on the
    stages before them."""
    return {"frames": {"source": source_identity(src, cache), "max_size": args.max_size,
                       "count": None if src.is_dir() else args.frames},
            "undistort": {"dense_size": args.dense_size},
            "masks": {"mask": args.mask},
            "stereo": {"dense": args.dense}}


class ScanLog:
    """scan.log in the run folder: per run, the settings used, every progress line with the time
    since the start, and how long each stage took. Appended to, so earlier runs stay readable,
    and written line by line, so a run that dies still shows how far it got."""

    def __init__(self, path: Path, echo=print):
        self.path, self.echo = path, echo
        self.start = time.time()
        self.timings: dict = {}
        self.reused: list = []      # stages already done when this run began
        self.failed: str | None = None

    def write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    def line(self, text: str) -> None:
        self.echo(text)
        self.write(f"[{time.time() - self.start:8.1f} s] {text}")

    def header(self, args, colmap) -> None:
        settings = dict(vars(args))
        self.write(f"\n===== fleshforge-scan {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.write(f"settings: {json.dumps(settings)}")
        self.write(f"colmap: {colmap.version() or '?'} [{colmap.where}]")

    @contextlib.contextmanager
    def stage(self, name: str):
        began = time.time()
        try:
            yield
        except BaseException as exc:
            self.failed = name
            self.write(f"stage {name}: FAILED after {time.time() - began:.1f} s: "
                       f"{type(exc).__name__}: {exc}".splitlines()[0])
            raise
        self.timings[name] = round(time.time() - began, 1)
        self.write(f"stage {name}: {self.timings[name]:.1f} s")

    def summary(self) -> None:
        if self.reused:
            self.write(f"reused from an earlier run: {', '.join(self.reused)}")
        skipped = [s for s in STAGES if s not in self.timings and s not in self.reused and s != self.failed]
        if skipped:
            self.write(f"not run: {', '.join(skipped)}")
        self.write(f"total: {time.time() - self.start:.1f} s")


def fail(errors: list, force: bool) -> None:
    for e in errors:
        print(f"REFUSED: {e}", file=sys.stderr)
    if not force:
        raise SystemExit(2)
    print("continuing because of --force", file=sys.stderr)


def scan(args, report_line=print) -> dict:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"{src}: no such file or folder")
    run = Run(Path(args.output))
    run.dir.mkdir(parents=True, exist_ok=True)
    log = ScanLog(run.dir / "scan.log", report_line)
    colmap, fallback = cm.choose(args.colmap)
    log.header(args, colmap)
    try:
        stats = _scan(args, src, run, colmap, fallback, log)
    except BaseException as exc:
        log.write(f"stopped: {type(exc).__name__}: {exc}".splitlines()[0])
        log.summary()
        raise
    log.summary()
    return stats


def _scan(args, src: Path, run: Run, colmap, fallback, log: ScanLog) -> dict:
    say = log.line
    legacy = bool(run.state.get("done")) and "inputs" not in run.state
    if legacy:
        say("note: this run folder predates input tracking; its finished stages are kept as if "
            "made with these options (use --from to redo them)")
    changed = run.match_inputs(stage_inputs(args, src, run.state.setdefault("file_ids", {})))
    if changed:
        say(f"inputs changed since the last run: redoing from {changed}")
    gone = run.missing_output(args.mask == "person")
    if gone:
        say(f"the files of the {gone} stage are missing: redoing from {gone}")
        run.redo_from(gone)
    if args.from_stage:
        if STAGES.index(args.from_stage) <= STAGES.index("stereo"):
            # asked to redo stereo: give the local COLMAP another chance (it may have been
            # updated, or the earlier failure may have been a full disk or a busy GPU)
            run.state.pop("stereo_crashed", None)
        run.redo_from(args.from_stage)
    say(f"colmap: {colmap.where}")
    root = run.dir
    log.reused = [s for s in STAGES if run.done(s)]

    if not run.done("frames"):
        with log.stage("frames"):
            if run.frames.exists():
                shutil.rmtree(run.frames)
            facts = (fr.frames_from_folder(str(src), run.frames, args.max_size, say) if src.is_dir()
                     else fr.frames_from_video(str(src), run.frames, args.frames, args.max_size, say))
            run.finish("frames", frames_info=facts)
    n_frames = len(fr.list_images(run.frames))

    if not run.done("sfm"):
        with log.stage("sfm"):
            say(f"sfm: solving camera poses for {n_frames} frames")
            cm.sparse(colmap, root, run.frames, run.colmap, n_frames, not args.no_gpu, run.log, say)
            run.finish("sfm")
    chosen = run.state.get("model")
    if chosen and (run.dir / chosen["folder"]).is_dir():
        model, n_models = cm.read_model(colmap, root, run.dir / chosen["folder"], run.log), chosen["models"]
    elif legacy and run.done("undistort"):
        # made before the choice was saved: find the one it used, without merging again
        model, n_models = cm.earlier_choice(colmap, root, run.colmap / "sparse", run.log)
        run.state["model"] = {"folder": model.folder.relative_to(run.dir).as_posix(), "models": n_models}
        run.save()
    else:    # picking (and maybe merging) is done once, so every rerun uses the same poses
        if run.done("undistort"):
            # the saved model is gone, and a new pick or merge can move the poses a little:
            # everything made from the old ones has to be made again
            say("the saved camera solution is missing; picking it again and redoing from undistort")
            run.redo_from("undistort")
        model, n_models = cm.best_model(colmap, root, run.colmap / "sparse", run.log)
        run.state["model"] = {"folder": model.folder.relative_to(run.dir).as_posix(), "models": n_models}
        run.save()
    say(f"sfm: {len(model.poses)}/{n_frames} frames placed, {model.n_points:,} points, "
        f"reprojection error {model.reproj_error:.2f} px")
    errors, _ = report.check_sparse(n_frames, len(model.poses), n_models)
    if errors:
        fail(errors, args.force)

    if not run.done("undistort"):
        with log.stage("undistort"):
            cm.undistort(colmap, root, run.frames, model.folder, run.dense, args.dense_size, run.log, say)
            run.finish("undistort")

    use_masks = args.mask == "person"
    if not run.done("masks"):
        with log.stage("masks"):
            facts = None
            if use_masks:
                from fleshforge.core import masks as mk
                say("masks: finding the person in each frame")
                facts = mk.person_masks(run.dense / "images", run.masks, report=say)
            run.finish("masks", masks=facts)

    if not run.done("stereo"):
        with log.stage("stereo"):
            say(f"stereo: depth maps, {args.dense} (the slow part)")
            crashed = run.state.setdefault("stereo_crashed", [])
            stereo_colmap = colmap
            if fallback is not None and colmap.where in crashed:
                say(f"  the local COLMAP's dense stereo crashed here before; using {fallback.where}")
                stereo_colmap = fallback
            try:
                cm.stereo(stereo_colmap, root, run.dense, args.dense, run.log, say)
            except cm.ColmapError:
                if fallback is None or stereo_colmap is fallback:
                    raise
                crashed.append(colmap.where)
                run.save()
                say(f"  dense stereo failed in the local COLMAP; retrying in {fallback.where}")
                cm.stereo(fallback, root, run.dense, args.dense, run.log, say)
            run.finish("stereo")

    if not run.done("fuse"):
        with log.stage("fuse"):
            cm.fuse(colmap, root, run.dense, run.fused, run.masks if use_masks else None, args.dense,
                    run.log, say)
            run.finish("fuse")

    # mesh, orientation and export are quick, so they are always redone
    with log.stage("mesh"):
        centres = np.array([p.centre for p in model.poses])
        up = orient.unit(np.mean([p.up for p in model.poses], axis=0))
        centre = orient.look_at_point(centres, [p.forward for p in model.poses])
        distance = float(np.linalg.norm(centres - centre, axis=1).mean())
        say("mesh: building the surface")
        m = mesh.build(run.fused, centre, args.crop * distance if args.crop > 0 else None,
                       args.poisson_depth, args.triangles, say, args.smooth, distance)

        faces = orient.detect_faces([run.frames / p.name for p in model.poses], say)
        front = orient.front_direction(model.poses, centre, up, faces)
        face_found = front is not None
        if not face_found:
            front = model.poses[0].centre - centre      # a guess: the video usually starts in front
        R = orient.basis(up, front)
        vertices = orient.to_gltf_frame(m.vertices, centre, R)
        normals = m.normals @ R.T
        vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        vertices *= orient.NOMINAL_SIZE_M / float((vertices.max(axis=0) - vertices.min(axis=0)).max())
        tris = m.faces
        if report.signed_volume(vertices, tris) < 0:      # inside out: flip to outward-facing
            tris, normals = tris[:, ::-1], -normals

        glb = run.dir / "head.glb"
        glbwrite.write_glb(glb, vertices, tris, m.colors, normals)
        report.save_preview(run.dir / "preview.png", *mesh.sample_surface(vertices, tris, m.colors))

    stats = {
        "input": str(src), "colmap": colmap.version(), "settings": vars(args),
        "frames": n_frames,
        "registered": len(model.poses), "models": n_models, "sparse_points": model.n_points,
        "reproj_error": round(model.reproj_error, 3),
        "arc_deg": round(orient.arc_degrees(centres, centre, up), 1),
        "face_found": face_found, "faces_detected": len(faces),
        "frames_info": run.state.get("frames_info"), "masks": run.state.get("masks"),
        "mesh": m.stats, "size_m": orient.NOMINAL_SIZE_M,
        "seconds": round(time.time() - log.start),
        "stage_seconds": dict(log.timings),        # stages run this time; the rest were reused
    }
    errors, warnings = report.check_scan(stats)
    stats["errors"], stats["warnings"] = errors, warnings
    (run.dir / "report.json").write_text(json.dumps(stats, indent=1))
    for w in warnings:
        say(f"warning: {w}")
    if errors:
        fail(errors, args.force)
    say(f"wrote {glb} ({m.stats['triangles']:,} triangles); look at {run.dir / 'preview.png'}")
    say(f"print it:  stackforge-paint {glb} --scale-to 50 --filaments <four ids> "
        "--template <your slicer project.3mf> --preview paint.png -o head.3mf")
    return stats


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scan(args)
    except (cm.ColmapError, mesh.MeshError, ValueError) as exc:
        print(f"fleshforge-scan: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
