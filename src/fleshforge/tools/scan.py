"""fleshforge-scan: a phone video of a head -> a vertex-coloured GLB.

    fleshforge-scan orbit.mp4 -o runs/alex

Stages, each kept in the run folder so a rerun picks up where it stopped:
frames -> sfm -> undistort -> masks -> stereo -> fuse -> mesh. `--from STAGE` redoes that
stage and everything after it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fleshforge.core import colmap as cm
from fleshforge.core import frames as fr
from fleshforge.core import glbwrite, mesh, orient, report

STAGES = ["frames", "sfm", "undistort", "masks", "stereo", "fuse", "mesh"]


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
    """The run folder: where each stage's files live, and which stages are done."""

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

    def finish(self, stage: str, **facts) -> None:
        self.state.setdefault("done", [])
        if stage not in self.state["done"]:
            self.state["done"].append(stage)
        self.state.update(facts)
        self.state_file.write_text(json.dumps(self.state, indent=1))

    def redo_from(self, stage: str) -> None:
        later = STAGES[STAGES.index(stage):]
        self.state["done"] = [s for s in self.state.get("done", []) if s not in later]


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
    if args.from_stage:
        run.redo_from(args.from_stage)
    colmap, fallback = cm.choose(args.colmap)
    report_line(f"colmap: {colmap.where}")
    root = run.dir
    started = time.time()

    if not run.done("frames"):
        if run.frames.exists():
            import shutil
            shutil.rmtree(run.frames)
        facts = (fr.frames_from_folder(str(src), run.frames, args.max_size, report_line) if src.is_dir()
                 else fr.frames_from_video(str(src), run.frames, args.frames, args.max_size, report_line))
        run.redo_from("sfm")
        run.finish("frames", frames_info=facts)
    n_frames = len(fr.list_images(run.frames))

    if not run.done("sfm"):
        report_line(f"sfm: solving camera poses for {n_frames} frames")
        cm.sparse(colmap, root, run.frames, run.colmap, n_frames, not args.no_gpu, run.log, report_line)
        run.redo_from("undistort")
        run.finish("sfm")
    model, n_models = cm.best_model(colmap, root, run.colmap / "sparse", run.log)
    report_line(f"sfm: {len(model.poses)}/{n_frames} frames placed, {model.n_points:,} points, "
                f"reprojection error {model.reproj_error:.2f} px")
    errors, _ = report.check_sparse(n_frames, len(model.poses), n_models)
    if errors:
        fail(errors, args.force)

    if not run.done("undistort"):
        cm.undistort(colmap, root, run.frames, model.folder, run.dense, args.dense_size, run.log,
                     report_line)
        run.redo_from("masks")
        run.finish("undistort")

    use_masks = args.mask == "person"
    if not run.done("masks") or run.state.get("mask_mode") != args.mask:
        facts = None
        if use_masks:
            from fleshforge.core import masks as mk
            report_line("masks: finding the person in each frame")
            facts = mk.person_masks(run.dense / "images", run.masks, report=report_line)
        run.redo_from("fuse")
        run.finish("masks", mask_mode=args.mask, masks=facts)

    if not run.done("stereo") or run.state.get("dense") != args.dense:
        report_line(f"stereo: depth maps, {args.dense} (the slow part)")
        try:
            cm.stereo(colmap, root, run.dense, args.dense, run.log, report_line)
        except cm.ColmapError:
            if fallback is None:
                raise
            report_line(f"  dense stereo failed in the local COLMAP; retrying in {fallback.where}")
            cm.stereo(fallback, root, run.dense, args.dense, run.log, report_line)
        run.redo_from("fuse")
        run.finish("stereo", dense=args.dense)

    if not run.done("fuse"):
        cm.fuse(colmap, root, run.dense, run.fused, run.masks if use_masks else None, args.dense,
                run.log, report_line)
        run.finish("fuse")

    # mesh, orientation and export are quick, so they are always redone
    centres = np.array([p.centre for p in model.poses])
    up = orient.unit(np.mean([p.up for p in model.poses], axis=0))
    centre = orient.look_at_point(centres, [p.forward for p in model.poses])
    distance = float(np.linalg.norm(centres - centre, axis=1).mean())
    report_line("mesh: building the surface")
    m = mesh.build(run.fused, centre, args.crop * distance if args.crop > 0 else None,
                   args.poisson_depth, args.triangles, report_line, args.smooth, distance)

    faces = orient.detect_faces([run.frames / p.name for p in model.poses], report_line)
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
        "input": str(src), "colmap": colmap.version(), "dense": args.dense, "smooth": args.smooth,
        "frames": n_frames,
        "registered": len(model.poses), "models": n_models, "sparse_points": model.n_points,
        "reproj_error": round(model.reproj_error, 3),
        "arc_deg": round(orient.arc_degrees(centres, centre, up), 1),
        "face_found": face_found, "faces_detected": len(faces),
        "frames_info": run.state.get("frames_info"), "masks": run.state.get("masks"),
        "mesh": m.stats, "size_m": orient.NOMINAL_SIZE_M,
        "seconds": round(time.time() - started),
    }
    errors, warnings = report.check_scan(stats)
    stats["errors"], stats["warnings"] = errors, warnings
    (run.dir / "report.json").write_text(json.dumps(stats, indent=1))
    for w in warnings:
        report_line(f"warning: {w}")
    if errors:
        fail(errors, args.force)
    report_line(f"wrote {glb} ({m.stats['triangles']:,} triangles); look at {run.dir / 'preview.png'}")
    report_line(f"print it:  stackforge-paint {glb} --scale-to 50 --filaments <four ids> "
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
