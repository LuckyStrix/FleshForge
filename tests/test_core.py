"""The pure parts: frame choice, GLB writing, orientation maths, hole capping, scan checks."""
import json
import struct
import sys

import numpy as np
import pytest

from fleshforge.core import colmap as cm
from fleshforge.core import frames, glbwrite, mesh, orient, report


# ── frames ────────────────────────────────────────────────────────────────────

def test_select_sharpest_takes_the_best_of_each_window():
    scores = [1, 9, 2, 3, 4, 8, 7, 1, 1]
    assert frames.select_sharpest(scores, 3) == [1, 5, 6]


def test_select_sharpest_keeps_everything_when_short():
    assert frames.select_sharpest([3, 1], 5) == [0, 1]
    assert frames.select_sharpest([3, 1], 0) == []


# ── GLB ───────────────────────────────────────────────────────────────────────

TETRA_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
TETRA_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])       # outward-facing


def read_glb(path):
    """Minimal reader for the files write_glb makes: (json document, attribute arrays)."""
    raw = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(raw))
    jlen, jtype = struct.unpack_from("<II", raw, 12)
    doc = json.loads(raw[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", raw, 20 + jlen)
    assert (jtype, btype) == (0x4E4F534A, 0x004E4942)
    binary = raw[28 + jlen:28 + jlen + blen]

    def accessor(i):
        acc = doc["accessors"][i]
        view = doc["bufferViews"][acc["bufferView"]]
        dtype = {5126: "<f4", 5125: "<u4"}[acc["componentType"]]
        width = {"VEC3": 3, "SCALAR": 1}[acc["type"]]
        data = np.frombuffer(binary, dtype, acc["count"] * width, view["byteOffset"])
        return data.reshape(acc["count"], width) if width > 1 else data

    prim = doc["meshes"][0]["primitives"][0]
    arrays = {k: accessor(i) for k, i in prim["attributes"].items()}
    arrays["indices"] = accessor(prim["indices"])
    return doc, arrays


def test_glb_round_trip(tmp_path):
    colours = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
    path = tmp_path / "t.glb"
    glbwrite.write_glb(path, TETRA_V, TETRA_F, colours, normals=np.tile([0, 0, 1.0], (4, 1)))
    doc, arrays = read_glb(path)
    assert np.allclose(arrays["POSITION"], TETRA_V)
    assert np.array_equal(arrays["indices"].reshape(-1, 3), TETRA_F)
    # COLOR_0 is linear light: sRGB 0.5 is about 0.214
    assert np.allclose(arrays["COLOR_0"][3], 0.2140, atol=1e-3)
    assert np.allclose(arrays["COLOR_0"][0], [1, 0, 0])
    assert doc["accessors"][0]["min"] == [0, 0, 0] and doc["accessors"][0]["max"] == [1, 1, 1]


def test_glb_rejects_bad_faces(tmp_path):
    with pytest.raises(ValueError):
        glbwrite.write_glb(tmp_path / "t.glb", TETRA_V, np.array([[0, 1, 9]]))


def test_glb_is_read_by_stackforge(tmp_path):
    """The handoff: stackforge's reader must see the mesh, in mm, with its colours."""
    glb = pytest.importorskip("stackforge.core.glb")
    colours = np.tile([0.8, 0.2, 0.1], (4, 1))
    path = tmp_path / "t.glb"
    glbwrite.write_glb(path, TETRA_V * 0.05, TETRA_F, colours)
    item = glb.read_glb(str(path), scale_to=50, report=lambda *_: None)
    assert item.appearance.corner_col is not None
    assert np.allclose(item.appearance.corner_col, glbwrite.srgb_to_linear([0.8, 0.2, 0.1]), atol=1e-4)


# ── orientation ───────────────────────────────────────────────────────────────

def ring_poses(n=12, radius=3.0, target=(0.5, 0.2, -0.3)):
    """Cameras on a level circle round `target`, all looking at it, world Y up."""
    target = np.array(target)
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        c = target + radius * np.array([np.sin(a), 0.0, np.cos(a)])
        forward = orient.unit(target - c)
        right = orient.unit(np.cross(forward, [0, 1.0, 0]))
        down = np.cross(forward, right)
        R = np.vstack([right, down, forward])
        poses.append(cm.Pose(f"{i:03d}.jpg", R, -R @ c))
    return poses, target


def test_pose_centre_and_axes():
    poses, target = ring_poses()
    p = poses[0]
    assert np.allclose(p.centre, target + [0, 0, 3.0])
    assert np.allclose(p.up, [0, 1, 0])
    assert np.allclose(p.forward, [0, 0, -1])


def test_look_at_point_finds_the_subject():
    poses, target = ring_poses()
    found = orient.look_at_point([p.centre for p in poses], [p.forward for p in poses])
    assert np.allclose(found, target, atol=1e-9)


def test_look_at_point_refuses_parallel_cameras():
    with pytest.raises(ValueError):
        orient.look_at_point([[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]])


def test_basis_is_a_rotation_with_y_up_and_z_front():
    R = orient.basis([0.1, 1.0, 0.0], [0.3, 0.5, 1.0])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ orient.unit([0.1, 1.0, 0.0]), [0, 1, 0])


def test_arc_degrees():
    poses, target = ring_poses(12)
    centres = np.array([p.centre for p in poses])
    assert np.isclose(orient.arc_degrees(centres, target, [0, 1, 0]), 330.0)
    assert np.isclose(orient.arc_degrees(centres[:4], target, [0, 1, 0]), 90.0)


def test_front_is_where_the_face_was_seen():
    poses, target = ring_poses(12)
    faces = {poses[2].name: (0.9, 0.9), poses[3].name: (0.9, 1.0), poses[4].name: (0.9, 0.9)}
    front = orient.front_direction(poses, target, [0, 1, 0], faces)
    assert np.allclose(front, orient.unit(poses[3].centre - target), atol=1e-6)
    assert orient.front_direction(poses, target, [0, 1, 0], {}) is None


def test_frontalness():
    face = np.zeros(15)
    face[4:10] = [40, 50, 60, 50, 50, 60]          # right eye, left eye, nose midway
    assert orient.frontalness(face) == 1.0
    face[8] = 60                                   # nose under the left eye: a profile
    assert orient.frontalness(face) == 0.0


# ── mesh ──────────────────────────────────────────────────────────────────────

def edges_are_closed(faces):
    a = faces[:, [0, 1, 2]].reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    n = faces.max() + 1
    return bool(np.isin(a * n + b, b * n + a).all())


def test_cap_holes_closes_an_open_tetrahedron():
    v, f, n = mesh.cap_holes(TETRA_V, TETRA_F[:3])          # one face missing
    assert n == 1 and len(v) == 5 and len(f) == 6
    assert edges_are_closed(f)
    assert report.signed_volume(v, f) > 0                   # still outward-facing


def test_cap_holes_leaves_a_closed_mesh_alone():
    v, f, n = mesh.cap_holes(TETRA_V, TETRA_F)
    assert n == 0 and len(f) == 4


def test_crop_sphere_and_colour_from_cloud():
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]], dtype=float)
    assert mesh.crop_sphere(pts, np.zeros(3), 1.0).tolist() == [True, True, False]
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    got = mesh.colour_from_cloud(np.array([[1.9, 0, 0]]), pts, cols, k=1)
    assert np.allclose(got, [[0, 0, 1]])


# ── report ────────────────────────────────────────────────────────────────────

def test_too_few_frames_placed_is_refused():
    errors, warnings = report.check_sparse(100, 60, 1)
    assert errors and "60 of 100" in errors[0]
    errors, warnings = report.check_sparse(100, 95, 2)
    assert not errors and "2 separate" in warnings[0]


def test_check_scan_warnings():
    stats = {"frames": 100, "registered": 99, "models": 1, "arc_deg": 180.0, "reproj_error": 2.0,
             "face_found": False, "mesh": {"closed": False}, "masks": {"images": 10, "empty": 5}}
    errors, warnings = report.check_scan(stats)
    assert not errors and len(warnings) == 5
    good = {"frames": 100, "registered": 99, "arc_deg": 350.0, "reproj_error": 0.6, "face_found": True,
            "mesh": {"closed": True}, "masks": None}
    assert report.check_scan(good) == ([], [])


def test_signed_volume_sign_follows_winding():
    assert np.isclose(report.signed_volume(TETRA_V, TETRA_F), 1 / 6)
    assert report.signed_volume(TETRA_V, TETRA_F[:, ::-1]) < 0


def test_render_views_draws_something():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(5000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    sheet = report.render_views(pts, np.full((5000, 3), 0.9), pts, size=64)
    assert sheet.shape == (128, 256, 3)
    assert (sheet > 100).any()


# ── COLMAP driver ─────────────────────────────────────────────────────────────

def test_parse_images_txt_handles_an_empty_points_line():
    text = ("# Image list\n# comment\n"
            "1 1 0 0 0 1 2 3 1 b.jpg\n10 20 -1 30 40 5\n"
            "2 1 0 0 0 0 0 0 1 a.jpg\n\n")
    poses = cm.parse_images_txt(text)
    assert [p.name for p in poses] == ["a.jpg", "b.jpg"]
    assert np.allclose(poses[1].centre, [-1, -2, -3])


def test_parse_points3d_txt():
    text = "# header\n1 0 0 0 255 0 0 0.5 1 2\n2 1 1 1 0 0 0 1.5 3 4\n"
    assert cm.parse_points3d_txt(text) == (2, 1.0)


def test_quat_to_rot_quarter_turn_about_z():
    s = np.sqrt(0.5)
    assert np.allclose(cm.quat_to_rot(s, 0, 0, s) @ [1, 0, 0], [0, 1, 0])


def test_option_picks_the_name_this_version_has():
    c = cm.Colmap("colmap")
    c._help["feature_extractor"] = "  --SiftExtraction.use_gpu arg (=1)\n"
    assert c.option("feature_extractor", "--FeatureExtraction.use_gpu",
                    "--SiftExtraction.use_gpu") == "--SiftExtraction.use_gpu"
    with pytest.raises(cm.ColmapError):
        c.option("feature_extractor", "--Nope.use_gpu")


def test_docker_command_maps_paths_into_the_mount(tmp_path):
    c = cm.Colmap("colmap", image="some/image:tag")
    cmd = c.command("mapper", ["--database_path", tmp_path / "colmap" / "database.db", "--n", 3], tmp_path)
    assert cmd[:2] == ["docker", "run"] and "some/image:tag" in cmd
    assert f"{tmp_path.resolve()}:/work" in cmd
    assert cmd[-5:] == ["mapper", "--database_path", "/work/colmap/database.db", "--n", "3"]
    with pytest.raises(cm.ColmapError):
        c.command("mapper", ["--image_path", tmp_path.parent / "elsewhere"], tmp_path)


def test_run_reports_a_failing_command(tmp_path):
    fake = cm.Colmap(sys.executable)      # "python -c ..." stands in for "colmap <command> ..."
    log = tmp_path / "log.txt"
    fake.run("-c", ["print('fine')"], tmp_path, log, report=lambda *_: None)
    with pytest.raises(cm.ColmapError) as err:
        fake.run("-c", ["import sys; print('went wrong'); sys.exit(3)"], tmp_path, log)
    assert "exit 0x3" in str(err.value) and "went wrong" in str(err.value)


def test_stereo_clears_stale_depth_maps(tmp_path):
    class Recorder(cm.Colmap):
        def has_cuda(self):
            return True

        def run(self, command, args, root, log, report=print):
            self.seen = [str(a) for a in args]

    stale = tmp_path / "stereo" / "depth_maps" / "0001.jpg.photometric.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    c = Recorder("colmap")
    cm.stereo(c, tmp_path, tmp_path, "fast", tmp_path / "log.txt")
    assert not stale.exists()
    assert c.seen[c.seen.index("--PatchMatchStereo.geom_consistency") + 1] == "0"
