# Working notes

**The rule:** `core/` holds the logic, `tools/` the CLIs (`build_parser()` + `main()`, like
stackforge). COLMAP is only ever a subprocess, through `core/colmap.py`. open3d, cv2 and rembg
are imported inside the functions that need them so the pure parts (and their tests) stay fast.
`pip install -e ".[dev]"`, then `pytest` and `ruff check src tests`; CI runs both.
The `.venv` is Python 3.12 (open3d has no 3.13+ wheels).

## Load-bearing

- **Scans are of real people. Nothing from `runs/` goes in git**, and no video, frame or model
  is added as a fixture. Test data is the synthetic ball in `tests/synth.py`.
- **The background is kept for pose solving and masked only for fusion.** The phone orbits a
  still person, so the room is what COLMAP tracks. (The Digital-Twins turntable rig is the
  opposite case: there the background must be masked for SfM.)
- **Masks are made on the undistorted images** (`colmap/dense/images`), which is what
  `stereo_fusion` reads, so sizes match. Naming is `<image name>.png`; black = left out.
- **Official Windows COLMAP 4.2.1: `patch_match_stereo` dies with 0xC0000409 on the RTX 5060 Ti**,
  no message, with or without geometric consistency. Same version in Docker works
  (`colmap/colmap:latest`, and the Digital-Twins `fip3d-pipeline` image). `cm.choose("auto")`
  returns the local COLMAP plus a Docker fallback, and `scan` retries only the stereo step,
  recording the crash in `state.json` (`stereo_crashed`) so reruns skip the local attempt.
- **COLMAP skips an image whose depth map exists.** `cm.stereo` deletes old `*.bin` first, or a
  rerun with other settings silently reuses the old maps.
- **Path arguments to `Colmap.run` must be `Path` objects inside the run folder.** Under Docker
  they are rewritten to `/work/...`; a string is passed through untouched.
- **Option names are looked up in `colmap <cmd> -h`** (`Colmap.option`): 4.x says
  `--FeatureExtraction.use_gpu`, 3.x `--SiftExtraction.use_gpu`.
- **COLOR_0 is linear light** (glTF's definition, and what `stackforge/core/glb.py` reads); the
  cloud's colours are sRGB and are decoded in `glbwrite.write_glb`.
- **GLB frame: Y up, face along +Z, 0.25 m across.** `orient.basis` is a proper rotation
  (det +1), so winding survives; `scan` flips the faces if the signed volume is negative.
- **"closed" means every edge has exactly two triangles.** Open3D's `is_watertight` also fails on
  self-intersection, which Poisson output often has.
- **Closing the mesh is a loop, after decimation** (`mesh.build`): drop triangles on edges with
  four triangles, drop folded slivers (`drop_folded`: two triangles running an edge the same
  way, which Open3D counts as fine), cap, repeat. `cap_holes` gives each rim loop its own centre;
  one fan for two openings that touch at a vertex puts four triangles on one edge.
- **The mapper splits a head orbit** at the back-of-head frames when the wall behind is plain.
  `cm.best_model` merges models that share frames (`model_merger`, then `bundle_adjuster`),
  retrying the leftovers until a round adds nothing; on the first real scan that took 122
  placed frames to 150 of 160. The result is saved as `state["model"]` and only re-read
  afterwards: re-merging on every run is slow and can shift the poses under undistort.
- **Stage caching is by inputs** (`scan.stage_inputs`, `Run.match_inputs`): a stage whose
  options or source changed is redone with everything after it. A new option that changes a
  stage's output must be added there, or reruns will reuse stale files. The source is known by
  content (`file_identity`: size plus three 1 MiB samples), never by path or mtime, so a moved
  video does not cost an hour of stereo. `state["file_ids"]` caches identities by path, size
  and mtime, so unchanged photos are not read again. `Run.redo_from` saves at once, so a crash
  mid-redo leaves no stale "done".
- **Older run folders are migrated, not redone.** No `inputs` (before 9f1826a): the current
  inputs are adopted, and the model is found on disk by `cm.earlier_choice` (largest model, or
  `colmap/merged/adjusted` if a merge happened) without merging again. A dict `source` (from
  9f1826a) is accepted if that path still holds a file of that size and today's identity.
- **A missing saved model means redo from undistort**: a new pick or merge can move the poses.
  A done stage whose files are gone (`Run.missing_output`) is redone from that stage.
- **`stereo_crashed` in state.json** sends stereo straight to the fallback; an explicit
  `--from` at or before stereo clears it, so the local COLMAP gets another chance.
- **`Colmap.help` treats a non-zero exit as "could not start"**: COLMAP's help exits 0, and
  otherwise Docker's own error (no NVIDIA container toolkit) would be read as help text. The
  failure is cached like the text (one launch), the message is the last line saying "error",
  and `cm.stereo` calls `help` before `has_cuda` so that error is what the user sees.
- **`scan.log`** is written by `ScanLog`: settings, timestamped progress, per-stage seconds,
  then which stages were reused (done when the run began) and which were not run (after a
  failure).
- **Over 200 frames, matching uses a pair list** (`cm.match_pairs`, via `matches_importer`):
  neighbours in time plus every 8th frame. `sequential_matcher` alone never closes the loop,
  and a video may go round the head several times.
- **Smoothing is sized from the camera distance**, the only yardstick in a scan with no scale:
  `SMOOTH_VOXEL` x distance x `--smooth` is the point spacing, and each point moves onto the
  plane through its 48 neighbours. Colours still come from the unsmoothed cloud.

## Bite

- **Open3D 0.20 `t.geometry.TriangleMesh.fill_holes` crashes (access violation) on Windows.**
  `mesh.cap_holes` is the numpy replacement: a fan to the centroid of each opening.
- **OpenCV 5 has no `CascadeClassifier`.** Faces come from `cv2.FaceDetectorYN` (YuNet), whose
  model is downloaded once to `~/.cache/fleshforge` with urllib (uses the Windows trust store, so
  Norton's HTTPS scanning does not break it). It prints a harmless "Targets are not supported"
  warning.
- **Smooth noise gives COLMAP nothing to match** (about 90 SIFT features a frame); the synthetic
  scene uses hard-edged cells for that reason.
- **One real head scanned** (720p compressed video): masks and front detection worked. Still
  unverified: a sharp original video, the `stackforge-paint` handoff of a real head, any print.
- **Commits here carry no Claude co-author line** (the user asked to be the listed contributor).
- `tools/` (COLMAP) and `runs/` are gitignored; `scripts/setup_windows.ps1` recreates `tools/`.
