"""Dense point cloud -> one closed, vertex-coloured mesh.

Depth noise is flattened in the cloud first (`smooth_points`): on skin it is bumps as large as
an eyelid, so the amount is a trade between a pebbled face and a featureless one.

Poisson reconstruction is used untrimmed on purpose: it bridges what the camera never saw
(under the chin, the top of the head), and a closed surface is what a slicer and stackforge's
voxeliser need. Where the subject runs out of the cropped cloud (the shoulders) the Poisson
surface is cut open at its bounding cube, so the remaining openings are capped afterwards.
The cost is that all of those closures are invented shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_POINTS = 2_000_000       # Poisson at depth 9 gains nothing from more
MIN_POINTS = 20_000          # below this the subject was not reconstructed
SMOOTH_VOXEL = 0.0025        # of the camera distance: about 1.5 mm on a head filmed from 0.6 m
CLOSE_PASSES = 5             # rounds of dropping pinched triangles and capping what that opens
SMOOTH_NEIGHBOURS = 48       # points per fitted plane: a patch about four voxels in radius


class MeshError(Exception):
    pass


@dataclass
class Mesh:
    vertices: np.ndarray     # (V, 3)
    faces: np.ndarray        # (F, 3)
    colors: np.ndarray       # (V, 3) sRGB-encoded 0..1
    normals: np.ndarray      # (V, 3)
    stats: dict


def crop_sphere(points, centre, radius) -> np.ndarray:
    """Boolean mask of the points within `radius` of `centre`."""
    return np.linalg.norm(np.asarray(points, float) - centre, axis=1) <= radius


def colour_from_cloud(vertices, cloud_points, cloud_colors, k=8) -> np.ndarray:
    """Each vertex takes the mean colour of its `k` nearest cloud points."""
    from scipy.spatial import cKDTree
    k = min(k, len(cloud_points))
    _, idx = cKDTree(cloud_points).query(vertices, k=k)
    if k == 1:
        return cloud_colors[idx]
    return cloud_colors[idx].mean(axis=1)


def smooth_points(points, k=SMOOTH_NEIGHBOURS, iterations=2):
    """Move each point onto the plane fitted through its `k` nearest neighbours.

    Returns (points, unit normals of those planes, unsigned). Depth noise on skin is bumps a few
    millimetres high and about a centimetre across; a surface wrapped round the raw cloud keeps
    every one of them. Needs evenly spaced points, so that k neighbours span a known patch."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    k = min(k, len(points))
    normals = np.zeros_like(points)
    for _ in range(iterations):
        _, idx = cKDTree(points).query(points, k=k, workers=-1)
        near = points[idx]
        centre = near.mean(axis=1)
        offset = near - centre[:, None]
        _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", offset, offset))
        normals = vectors[:, :, 0]                   # least spread = across the surface
        points = points - ((points - centre) * normals).sum(axis=1, keepdims=True) * normals
    return points, normals


def drop_folded(faces, n_vertices: int) -> np.ndarray:
    """`faces` without the triangles that run an edge the same way as another triangle does.

    Two neighbours should cross their shared edge in opposite directions; a sliver folded over
    by decimation does not, and capping would then stack two more triangles on that edge."""
    faces = np.asarray(faces, np.int64)
    key = (faces[:, [0, 1, 2]] * n_vertices + faces[:, [1, 2, 0]]).reshape(-1)
    values, counts = np.unique(key, return_counts=True)
    return faces[~np.isin(key, values[counts > 1]).reshape(-1, 3).any(axis=1)]


def open_loops(faces, n_vertices: int) -> list:
    """The openings of a mesh, each as a list of vertex indices walked along its rim.

    An edge with no triangle running it the other way is on an opening. Where two openings
    touch at one vertex the walk is split there, so no loop passes a vertex twice."""
    faces = np.asarray(faces, np.int64)
    a = faces[:, [0, 1, 2]].reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    is_open = ~np.isin(a * n_vertices + b, b * n_vertices + a)
    onward: dict = {}
    for start, end in zip(a[is_open].tolist(), b[is_open].tolist()):
        onward.setdefault(start, []).append(end)
    loops = []
    for start in list(onward):
        path, place = [start], {start: 0}
        while onward[path[-1]]:
            nxt = onward[path[-1]].pop()
            if nxt in place:                         # came back round: cut that loop off
                loops.append(path[place[nxt]:])
                for v in path[place[nxt] + 1:]:
                    del place[v]
                del path[place[nxt] + 1:]
            elif nxt in onward:
                place[nxt] = len(path)
                path.append(nxt)
            else:                                    # a rim that leads nowhere: not a loop
                break
    return loops


def cap_holes(vertices, faces):
    """Close every opening with a fan of triangles to the opening's centre.

    Returns (vertices, faces, number of openings capped); each opening gets one new vertex at
    the centroid of its rim. (Open3D's own fill_holes crashes with an access violation on
    Windows in 0.20.)"""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    loops = open_loops(faces, len(vertices))
    if not loops:
        return vertices, faces, 0
    centres, caps = [], []
    for k, loop in enumerate(loops):
        rim = np.array(loop)
        hub = np.full(len(rim), len(vertices) + k)
        centres.append(vertices[rim].mean(axis=0))
        caps.append(np.stack([np.roll(rim, -1), rim, hub], axis=1))   # reversed edge: same way out
    return np.vstack([vertices, centres]), np.vstack([faces] + caps), len(loops)


def sample_surface(vertices, faces, colors, count=1_500_000):
    """(points, colours, normals) spread evenly over the surface, for drawing it."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices, float)),
                                     o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(count, use_triangle_normal=True)
    return np.asarray(pcd.points), np.asarray(pcd.colors), np.asarray(pcd.normals)


def build(cloud_ply: Path, centre, radius, depth: int = 9, triangles: int = 300_000,
          report=print, smooth: float = 1.5, distance: float | None = None) -> Mesh:
    """`centre`/`radius` bound the subject (radius None or 0 keeps everything). `smooth` scales
    how much depth noise is flattened (0 = mesh the raw cloud); `distance` is how far the cameras
    were from the subject, the only yardstick there is in a scan without a real scale."""
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    pcd = o3d.io.read_point_cloud(str(cloud_ply))
    n_fused = len(pcd.points)
    if n_fused == 0:
        raise MeshError("the dense cloud is empty: nothing was reconstructed (with --mask person, "
                        "check that the masks actually cover the subject)")
    if radius:
        keep = np.flatnonzero(crop_sphere(np.asarray(pcd.points), centre, radius))
        pcd = pcd.select_by_index(keep)
    n_cropped = len(pcd.points)
    if n_cropped > MAX_POINTS:
        pcd = pcd.random_down_sample(MAX_POINTS / n_cropped)
    if len(pcd.points) >= 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    n_clean = len(pcd.points)
    report(f"  cloud: {n_fused:,} fused, {n_cropped:,} on the subject, {n_clean:,} after cleaning")
    if n_clean < MIN_POINTS:
        raise MeshError(f"only {n_clean:,} points on the subject (need {MIN_POINTS:,}). Too sparse "
                        "to mesh. If the masks in colmap/dense/masks miss the subject, use --mask off; "
                        "otherwise retake the video with more light and a slower orbit")

    cloud_points, cloud_colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    if smooth > 0:
        unit = distance or float(np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())) / 2
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(32))
            pcd.orient_normals_consistent_tangent_plane(24)
        pcd = pcd.voxel_down_sample(SMOOTH_VOXEL * unit * smooth)
        points, normals = smooth_points(np.asarray(pcd.points))
        # the fitted planes have no inside or outside; take that from the cloud's own normals
        normals[(normals * np.asarray(pcd.normals)).sum(axis=1) < 0] *= -1
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        report(f"  smoothed: {len(points):,} evenly spaced points")
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(32))
        pcd.orient_normals_consistent_tangent_plane(24)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, linear_fit=True)

    # keep the biggest piece: Poisson leaves small shells round stray points
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    if len(counts) > 1:
        mesh.remove_triangles_by_mask(clusters != counts.argmax())
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > triangles:
        mesh = mesh.simplify_quadric_decimation(triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    # decimation leaves a few edges shared by four triangles (two sheets pinched together) and
    # a few folded slivers. Dropping those triangles opens small holes, which are capped with the rest; a cap can
    # itself land on a pinch, so go round until every edge has two triangles
    n_capped = 0
    for _ in range(CLOSE_PASSES):
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        vertices = np.asarray(mesh.vertices)
        capped_v, capped_f, n = cap_holes(vertices, drop_folded(np.asarray(mesh.triangles), len(vertices)))
        n_capped += n
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(capped_v),
                                         o3d.utility.Vector3iVector(capped_f.astype(np.int32)))
        if mesh.is_edge_manifold(allow_boundary_edges=False):
            break
    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    colors = colour_from_cloud(vertices, cloud_points, cloud_colors)
    stats = {"points_fused": n_fused, "points_subject": n_cropped, "points_clean": n_clean,
             "vertices": len(vertices), "triangles": len(mesh.triangles),
             # closed = every edge shared by exactly two triangles. Open3D's is_watertight also
             # demands no self-intersection, which Poisson output rarely meets and slicers forgive
             "closed": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
             "pieces_removed": int(max(0, len(counts) - 1)),
             "openings_capped": n_capped}
    report(f"  mesh: {stats['triangles']:,} triangles, "
           f"{'closed' if stats['closed'] else 'NOT closed'}")
    return Mesh(vertices, np.asarray(mesh.triangles), colors, np.asarray(mesh.vertex_normals), stats)
