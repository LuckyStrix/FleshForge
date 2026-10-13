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

- **Not yet run on a real head.** The whole pipeline is tested on a synthetic scene (a textured
  ball, ray-traced) and its GLB loads in `stackforge-paint`. The person masks and the
  front-of-face detection have not been through a real scan, and nothing has been printed.
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

A folder of photos works in place of a video: `fleshforge-scan photos/ -o runs/alex`.

## What the stages do

Each stage keeps its output in the run folder, so a rerun continues where it stopped.
`--from STAGE` redoes that stage and the ones after it.

| stage | what happens |
|---|---|
| `frames` | ffmpeg extracts three times the frames needed, and the sharpest in each time window is kept (120 by default) |
| `sfm` | COLMAP finds features, matches every pair, and solves the camera positions |
| `undistort` | frames are corrected for lens distortion and shrunk to `--dense-size` |
| `masks` | the person is found in each frame (rembg, `u2net_human_seg`), so the room stays out of the cloud |
| `stereo` | COLMAP computes a depth map per frame on the GPU (the slow part) |
| `fuse` | depth maps are merged into one coloured point cloud |
| `mesh` | crop to the subject, Poisson surface, cap openings, 300k triangles, colours from the cloud, orient (up from the phone, front from where a face was seen), write GLB, preview and report |

Useful options: `--dense fine` (COLMAP's full two-pass stereo, several times slower),
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
fails, that one step is retried in Docker:

```sh
docker pull colmap/colmap:latest      # or set FLESHFORGE_COLMAP_IMAGE to another image
```

`--colmap docker` runs every COLMAP step in the container.

Timing on an RTX 5060 Ti, 40 frames at 800 px: poses 7 s, dense stereo 92 s with `--dense fast`
and 7.5 min with `--dense fine` (both in Docker). A real scan (120 frames at 1200 px) has not
been timed.

## Development

`pip install -e ".[dev]"`, then `pytest` and `ruff check src tests`. One test runs a whole scan
of the synthetic ball through COLMAP (about 90 s with a GPU); it is skipped when no COLMAP is
found and with `FLESHFORGE_SKIP_COLMAP=1`. The stackforge handoff test runs when stackforge is
installed in the same environment. `CLAUDE.md` has the things that bite.

## License

MIT. See `LICENSE`.
