# fleshforge

A phone video of someone's head in, a 3D model you can print in colour out. The name is the
joke: it sits between a FlashForge printer and [stackforge](https://github.com/LuckyStrix/stackforge),
which does the colour printing. The goal is bobbleheads.

```
phone video ──► sharp frames ──► camera poses ──► dense cloud ──► closed mesh ──► head.glb ──► stackforge-paint ──► 3MF
                 (ffmpeg)         (COLMAP)        (COLMAP, GPU)    (Open3D)      vertex colours
```

**Where this is:** the scan stage (video to GLB) is built. The neck cut, bobble socket, body and
automatic choice of filaments are not.

## Caveats up front

- **Run on one real head, never printed.** One 36 s phone video (720p, a compressed copy) gave
  a recognisable, closed, correctly oriented head and shoulders, with the person masks and the
  front-of-face detection working. The automated tests use a synthetic scene (a textured ball,
  ray-traced). Nothing has been through `stackforge-paint` to a printer yet.
- **Skin comes out pebbled unless it is smoothed, and smoothing eats the face.** Depth noise on
  skin is bumps about as big as an eyelid. `--smooth` (default 1.5) trades one for the other;
  a sharper video is the real fix.
- **Expect shape plus a few flat colours, not a photograph.** A four-toolhead printer gets four
  filaments, and `stackforge-paint` gives each voxel its nearest one.
- **The mesh invents what the camera did not see.** Under the chin, the top of the head and the
  cut at the shoulders are closed by the mesher. Hair comes out as a smooth blob.
- **True size is unknown from a video.** The GLB is 0.25 m across; set the print size with
  `stackforge-paint --scale-to`.
- Needs an NVIDIA GPU: COLMAP's dense stereo is CUDA only.

## Quick start (Windows)

```powershell
git clone https://github.com/LuckyStrix/FleshForge; cd FleshForge
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1   # venv + COLMAP into this folder, no admin
.venv\Scripts\fleshforge-scan orbit.mp4 -o runs\alex
```

It needs Python 3.10 to 3.12 (Open3D has no wheels for 3.13+) and ffmpeg on PATH
(`winget install Gyan.FFmpeg`). On Linux: `pip install -e ".[dev]"` and `apt install colmap ffmpeg`
(a CUDA build of COLMAP). `fleshforge-check` says what is missing.

Then look at `runs\alex\preview.png` (four views, in colour and as bare shape) before printing:

```sh
stackforge-paint runs/alex/head.glb --scale-to 50 --filaments <four ids> \
    --template my_project.3mf --preview paint.png -o head.3mf
```

Scans are of real people, so `runs/` and every video and model format are in `.gitignore`.

## Taking the video

- One slow walk all the way round the head, 45 to 60 seconds, phone upright, head filling
  about half the frame. A second pass from slightly above helps the top of the head.
- The subject keeps still and looks at one spot: no talking, eyes open, neutral face.
- Even, soft light. No direct sun, no flash; shiny skin reconstructs badly.
- Keep the room in view. The background is what the camera positions are solved from.
- Hair tied back or under a cap gives a cleaner shape.
- Lock exposure and focus if the phone allows it.
- Use the phone's original file (cable or AirDrop). A copy sent through a messaging app is
  shrunk and recompressed, and the lost skin texture is exactly what the depth step needs.
- Nothing on the neck or head that is not meant to be printed: it becomes part of the model.

A folder of photos works in place of a video: `fleshforge-scan photos/ -o runs/alex`.

## What the stages do

Each stage keeps its output in the run folder, so a rerun continues where it stopped.
`state.json` records what each stage was made from (the input file, `--frames`, `--max-size`,
`--dense-size`, `--mask`, `--dense`), and a stage whose inputs changed is redone with the ones
after it, so a new video in an old folder never reuses the last person's frames. The input is
recognised by its contents, so a moved or re-copied video counts as the same one; for a folder
of photos only the images count, not sidecar files. `--from STAGE` redoes that stage and the
ones after it regardless. A folder made by an earlier version keeps its finished stages and
camera solution. If a stage's files have been deleted (to free space, say), that stage is
redone.

`scan.log` in the run folder keeps, for every run, the settings used, each progress line with
the time since the start, how long each stage took, and which stages were reused. A run that
fails still logs how far it got, why, and which stages it did not reach. `report.json` also holds the settings and `stage_seconds`.

| stage | what happens |
|---|---|
| `frames` | ffmpeg extracts three times the frames needed, and the sharpest in each time window is kept (120 by default) |
| `sfm` | COLMAP finds features, matches them, and solves the camera positions. Up to 200 frames every pair is matched; above that, each frame is matched with its neighbours in time and with every 8th frame, which still links a video that goes round several times. If COLMAP splits the orbit into several reconstructions, the ones that share frames are merged. The chosen model is remembered, so later stages always use the same poses |
| `undistort` | frames are corrected for lens distortion and shrunk to `--dense-size` |
| `masks` | the person is found in each frame (rembg, `u2net_human_seg`), so the room stays out of the cloud |
| `stereo` | COLMAP computes a depth map per frame on the GPU (the slow part) |
| `fuse` | depth maps are merged into one coloured point cloud |
| `mesh` | crop to the subject, flatten depth noise, Poisson surface, 300k triangles, cap openings, colours from the cloud, orient (up from the phone, front from where a face was seen), write GLB, preview and report |

Useful options: `--dense fine` (COLMAP's full two-pass stereo, several times slower),
`--smooth` (0 = mesh the raw cloud, 1 = light, 2 = smooth skin but soft eyes and mouth; with
`--from mesh` a new value takes seconds to try),
`--mask off` (scan an object, not a person), `--crop` (how far from the point the cameras look
at is kept), `--colmap native|docker`.

### It refuses bad scans

`fleshforge-scan` stops with exit code 2 when fewer than 80% of the frames can be placed, before
spending time on the dense stage, and says why. It warns when the camera went less than 270
degrees round, when the camera solution is loose, when no frontal face was found (the front is
then a guess) and when the mesh is not closed. `report.json` holds all the numbers. `--force`
carries on regardless.

### COLMAP on Windows, and Docker

The official Windows build of COLMAP 4.2.1 crashes in dense stereo on an RTX 5060 Ti
(`0xC0000409`, no message), while the same version in a Linux container runs on the same card.
So with `--colmap auto` (the default) the local COLMAP does everything and, if its dense stereo
fails, that one step is retried in Docker, and the run folder remembers it so later reruns go
straight to Docker:

```sh
docker pull colmap/colmap:latest      # or set FLESHFORGE_COLMAP_IMAGE to another image
```

`--colmap docker` runs every COLMAP step in the container.

Timing on an RTX 5060 Ti, 40 frames at 800 px: poses 7 s, dense stereo 92 s with `--dense fast`
and 7.5 min with `--dense fine` (both in Docker). The real scan (150 frames at 1280 px): poses
under a minute, dense stereo 6 min with `--dense fast`; `--dense fine` was not timed. Dense
stereo is also the step that needs the most memory.

## Development

`pip install -e ".[dev]"`, then `pytest` and `ruff check src tests`. One test runs a whole scan
of the synthetic ball through COLMAP (about 90 s with a GPU); it is skipped when no COLMAP is
found and with `FLESHFORGE_SKIP_COLMAP=1`. The stackforge handoff test runs when stackforge is
installed in the same environment. `CLAUDE.md` has the things that bite.

## License

MIT. See `LICENSE`.
