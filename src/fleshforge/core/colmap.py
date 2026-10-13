"""Drive COLMAP: camera poses from the frames, then a dense point cloud.

COLMAP is run as a subprocess and never linked. Option names changed between releases
(`--SiftExtraction.use_gpu` became `--FeatureExtraction.use_gpu`), so names that differ are
looked up in the command's own `-h` output instead of being tied to one version.

Checked against COLMAP 4.2.1 (Windows CUDA build).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXHAUSTIVE_MAX_FRAMES = 200     # above this, match neighbours in time instead of every pair
DEFAULT_IMAGE = "colmap/colmap:latest"
DOCKER_MOUNT = "/work"


class ColmapError(Exception):
    pass


@dataclass
class Colmap:
    """A COLMAP to run: a local executable, or (`image` set) one inside a Docker image.

    Path arguments must be `Path` objects: under Docker they are rewritten to the run folder's
    mount point, and a plain string would be passed through as it is."""
    exe: str
    env: dict = field(default_factory=lambda: os.environ.copy())
    image: str | None = None
    _help: dict = field(default_factory=dict, repr=False)

    @property
    def where(self) -> str:
        return f"Docker image {self.image}" if self.image else self.exe

    def command(self, command: str, args: list, root: Path | None = None) -> list:
        if not self.image:
            return [self.exe, command] + [str(a) for a in args]
        cmd = ["docker", "run", "--rm", "--gpus", "all"]
        if hasattr(os, "getuid"):        # so the files are not left owned by root
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if root is not None:
            cmd += ["-v", f"{Path(root).resolve()}:{DOCKER_MOUNT}"]
        return cmd + ["--entrypoint", self.exe, self.image, command] + [_in_mount(a, root) for a in args]

    def help(self, command: str) -> str:
        if command not in self._help:
            done = subprocess.run(self.command(command, ["-h"]), capture_output=True, text=True, env=self.env)
            self._help[command] = done.stdout + done.stderr
        return self._help[command]

    def option(self, command: str, *names: str) -> str:
        """The first of `names` this COLMAP's `command` accepts."""
        text = self.help(command)
        for name in names:
            if name + " " in text or name + "\n" in text:
                return name
        raise ColmapError(f"colmap {command} accepts none of {', '.join(names)}; this COLMAP "
                          "version is not supported (checked against 4.2.1)")

    def version(self) -> str:
        lines = self.help("help").strip().splitlines()
        return lines[0] if lines else ""

    def has_cuda(self) -> bool:
        first = self.version()
        return "with CUDA" in first and "without CUDA" not in first

    def run(self, command: str, args: list, root: Path, log: Path, report=print) -> None:
        """`root` is the run folder: every path in `args` lies inside it."""
        cmd = self.command(command, args, root)
        start = time.time()
        with open(log, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n===== {' '.join(cmd)}\n")
            fh.flush()
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=self.env).returncode
        if code != 0:
            tail = "\n".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
            raise ColmapError(f"colmap {command} failed (exit {code:#x}) in {self.where}. "
                              f"End of {log}:\n{tail}")
        report(f"  colmap {command}: {time.time() - start:.0f} s")


def _in_mount(arg, root) -> str:
    if not isinstance(arg, Path):
        return str(arg)
    if root is None:
        raise ColmapError(f"no run folder given to place {arg} in the container")
    try:
        rel = arg.resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise ColmapError(f"{arg} is outside the run folder {root}, so Docker cannot see it")
    return f"{DOCKER_MOUNT}/{rel.as_posix()}".rstrip("/")


def find_native() -> Colmap | None:
    """$FLESHFORGE_COLMAP, then tools/colmap in the repo (the Windows zip), then PATH."""
    given = os.environ.get("FLESHFORGE_COLMAP")
    candidates = [given] if given else []
    candidates += [str(REPO_ROOT / "tools" / "colmap" / "COLMAP.bat"), shutil.which("colmap") or ""]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return _from_path(Path(cand))
    return None


def docker_image() -> str:
    return os.environ.get("FLESHFORGE_COLMAP_IMAGE") or DEFAULT_IMAGE


def find_docker() -> Colmap | None:
    """COLMAP in a Docker image that is already pulled. Nothing is downloaded here."""
    docker = shutil.which("docker")
    if not docker:
        return None
    image = docker_image()
    try:
        seen = subprocess.run([docker, "image", "inspect", image], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Colmap("colmap", image=image) if seen.returncode == 0 else None


def choose(backend: str = "auto") -> tuple[Colmap, Colmap | None]:
    """(the COLMAP to use, one to fall back to for dense stereo or None).

    auto: the local COLMAP when there is one, with Docker behind it for dense stereo, because
    the official Windows build's stereo crashes on some GPUs (seen: 4.2.1 on an RTX 5060 Ti)
    where the same version in a Linux container runs."""
    native = find_native() if backend in ("auto", "native") else None
    docker = find_docker() if backend in ("auto", "docker") else None
    if backend == "native" and native is None:
        raise ColmapError("no local COLMAP: run scripts/setup_windows.ps1, put colmap on PATH, "
                          "or set FLESHFORGE_COLMAP")
    if backend == "docker" and docker is None:
        raise ColmapError(f"Docker image {docker_image()} is not available: start Docker and run "
                          f"`docker pull {docker_image()}` (or set FLESHFORGE_COLMAP_IMAGE)")
    if native is None and docker is None:
        raise ColmapError("COLMAP not found. Run scripts/setup_windows.ps1 for a local copy, or "
                          f"`docker pull {DEFAULT_IMAGE}`")
    return (native, docker) if native is not None else (docker, None)


def _from_path(path: Path) -> Colmap:
    env = os.environ.copy()
    if path.suffix.lower() == ".bat":
        # The Windows release ships COLMAP.bat, which only puts bin\ on PATH and starts
        # bin\colmap.exe. Doing that here skips cmd.exe and its argument quoting.
        exe = path.parent / "bin" / "colmap.exe"
        if exe.is_file():
            env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
            env["QT_PLUGIN_PATH"] = str(path.parent / "plugins")
            return Colmap(str(exe), env)
    return Colmap(str(path), env)


def _flag(on: bool) -> str:
    return "1" if on else "0"


# Dense stereo settings. "fast" is COLMAP's own medium preset without the geometric-consistency
# second pass; "fine" is its defaults.
DENSE = {
    "fast": {"geometric": False, "args": ["--PatchMatchStereo.window_radius", "4",
                                          "--PatchMatchStereo.window_step", "2",
                                          "--PatchMatchStereo.num_samples", "10"]},
    "fine": {"geometric": True, "args": []},
}


def sparse(colmap: Colmap, root: Path, frames_dir: Path, work: Path, n_frames: int, gpu: bool,
           log: Path, report=print) -> Path:
    """Feature extraction, matching and mapping. Returns the folder holding the sparse models."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    for stale in work.glob("database.db*"):
        stale.unlink()
    out = work / "sparse"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    fx_gpu = colmap.option("feature_extractor", "--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu")
    # one phone, one lens, fixed zoom: every frame shares a camera, which steadies the focal length
    colmap.run("feature_extractor", [
        "--database_path", db, "--image_path", frames_dir,
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
        fx_gpu, _flag(gpu)], root, log, report)

    matcher = "exhaustive_matcher" if n_frames <= EXHAUSTIVE_MAX_FRAMES else "sequential_matcher"
    fm_gpu = colmap.option(matcher, "--FeatureMatching.use_gpu", "--SiftMatching.use_gpu")
    args = ["--database_path", db, fm_gpu, _flag(gpu)]
    if matcher == "sequential_matcher":
        args += ["--SequentialMatching.overlap", "20"]
    colmap.run(matcher, args, root, log, report)

    colmap.run("mapper", ["--database_path", db, "--image_path", frames_dir, "--output_path", out],
               root, log, report)
    return out


def undistort(colmap: Colmap, root: Path, frames_dir: Path, model_dir: Path, dense_dir: Path,
              max_size: int, log: Path, report=print) -> None:
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    colmap.run("image_undistorter", [
        "--image_path", frames_dir, "--input_path", model_dir, "--output_path", dense_dir,
        "--output_type", "COLMAP", "--max_image_size", max_size], root, log, report)


def stereo(colmap: Colmap, root: Path, dense_dir: Path, quality: str, log: Path, report=print) -> None:
    if not colmap.has_cuda():
        raise ColmapError(f"{colmap.where} was built without CUDA, and dense stereo needs it")
    q = DENSE[quality]
    # COLMAP skips an image whose depth map is already on disk, so maps from an earlier run
    # with other settings (or half of a crashed one) would be reused without a word
    for kind in ("depth_maps", "normal_maps"):
        for stale in (dense_dir / "stereo" / kind).glob("*.bin"):
            stale.unlink()
    colmap.run("patch_match_stereo", [
        "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", _flag(q["geometric"]), *q["args"]], root, log, report)


def fuse(colmap: Colmap, root: Path, dense_dir: Path, out_ply: Path, mask_dir: Path | None,
         quality: str, log: Path, report=print) -> None:
    args = ["--workspace_path", dense_dir, "--workspace_format", "COLMAP",
            "--input_type", "geometric" if DENSE[quality]["geometric"] else "photometric",
            "--output_path", out_ply]
    if mask_dir is not None:
        args += ["--StereoFusion.mask_path", mask_dir]
    colmap.run("stereo_fusion", args, root, log, report)


# ── reading a sparse model ────────────────────────────────────────────────────

@dataclass
class Pose:
    """One registered image. World -> camera is `x_cam = R @ x_world + t`; COLMAP's camera
    looks along +z with +x right and +y down."""
    name: str
    R: np.ndarray
    t: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    @property
    def up(self) -> np.ndarray:
        return -self.R[1]


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def parse_images_txt(text: str) -> list[Pose]:
    """images.txt: after the comments, lines come in pairs (pose, then that image's 2D points,
    which may be an empty line)."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    poses = []
    for header in lines[0::2]:
        tok = header.split()
        if len(tok) < 10:
            raise ColmapError(f"images.txt: unexpected line {header[:60]!r}")
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in tok[1:8])
        poses.append(Pose(" ".join(tok[9:]), quat_to_rot(qw, qx, qy, qz), np.array([tx, ty, tz])))
    return sorted(poses, key=lambda p: p.name)


def parse_points3d_txt(text: str) -> tuple[int, float]:
    """(number of sparse points, mean reprojection error in pixels)."""
    errors = [float(ln.split()[7]) for ln in text.splitlines() if ln and not ln.startswith("#")]
    return len(errors), (float(np.mean(errors)) if errors else float("nan"))


@dataclass
class SparseModel:
    folder: Path
    poses: list
    n_points: int
    reproj_error: float


def read_model(colmap: Colmap, root: Path, model_dir: Path, log: Path) -> SparseModel:
    txt = model_dir.parent / (model_dir.name + "_txt")
    if txt.exists():
        shutil.rmtree(txt)
    txt.mkdir()
    colmap.run("model_converter", ["--input_path", model_dir, "--output_path", txt, "--output_type", "TXT"],
               root, log, report=lambda *_: None)
    poses = parse_images_txt((txt / "images.txt").read_text(encoding="utf-8", errors="replace"))
    n, err = parse_points3d_txt((txt / "points3D.txt").read_text(encoding="utf-8", errors="replace"))
    shutil.rmtree(txt)
    return SparseModel(model_dir, poses, n, err)


def merge_models(colmap: Colmap, root: Path, models: list, work: Path, log: Path) -> tuple[SparseModel, int]:
    """Join the smaller models onto the largest where they share frames, then re-optimise.

    Returns (model, how many models were left out). The mapper starts a new model when it cannot
    carry on from the last frame (the back of a head against a plain wall is typical); the same
    frames often sit in both, and that overlap is what `model_merger` aligns them by."""
    models = sorted(models, key=lambda m: len(m.poses), reverse=True)
    best, left_out = models[0], 0
    for k, other in enumerate(models[1:]):
        out = work / f"merge{k}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        try:
            colmap.run("model_merger", ["--input_path1", best.folder, "--input_path2", other.folder,
                                        "--output_path", out], root, log, report=lambda *_: None)
            joined = read_model(colmap, root, out, log) if (out / "images.bin").exists() else None
        except ColmapError:
            joined = None
        if joined and len(joined.poses) > len(best.poses):
            best = joined
        else:
            left_out += 1
    if best is not models[0]:
        out = work / "adjusted"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        colmap.run("bundle_adjuster", ["--input_path", best.folder, "--output_path", out], root, log,
                   report=lambda *_: None)
        best = read_model(colmap, root, out, log)
    return best, left_out


def best_model(colmap: Colmap, root: Path, sparse_dir: Path, log: Path) -> tuple[SparseModel, int]:
    """The model to use, and how many separate models it stands for (1 = everything connected).
    When the mapper made several, the ones that share frames with the largest are merged into it."""
    folders = sorted(p for p in sparse_dir.iterdir() if p.is_dir() and not p.name.endswith("_txt"))
    if not folders:
        raise ColmapError("COLMAP could not build a model from these frames (too little overlap "
                          "or texture). Check the capture advice in the README and retake the video")
    models = [read_model(colmap, root, f, log) for f in folders]
    if len(models) == 1:
        return models[0], 1
    model, left_out = merge_models(colmap, root, models, sparse_dir.parent / "merged", log)
    return model, 1 + left_out
