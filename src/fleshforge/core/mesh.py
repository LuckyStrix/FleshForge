"""Dense point cloud -> one closed, vertex-coloured mesh.

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


def cap_holes(vertices, faces):
    """Close every opening with a fan of triangles to the opening's centre.

    Returns (vertices, faces, number of openings capped). An edge used by one triangle only is
    on an opening; each connected run of such edges gets one new vertex at its centroid.
    (Open3D's own fill_holes crashes with an access violation on Windows in 0.20.)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    n = len(vertices)
    a = faces[:, [0, 1, 2]].reshape(-1)
    b = faces[:, [1, 2, 0]].reshape(-1)
    open_edge = ~np.isin(a * n + b, b * n + a)       # no triangle runs this edge the other way
    a, b = a[open_edge], b[open_edge]
    if len(a) == 0:
        return vertices, faces, 0
    rim = np.unique(np.concatenate([a, b]))
    local = np.searchsorted(rim, np.concatenate([a, b])).reshape(2, -1)
    graph = coo_matrix((np.ones(len(a)), (local[0], local[1])), shape=(len(rim), len(rim)))
    count, label = connected_components(graph, directed=False)
    centres = np.array([vertices[rim[label == k]].mean(axis=0) for k in range(count)])
    hub = n + label[local[0]]                        # the new vertex each open edge fans to
    caps = np.stack([b, a, hub], axis=1)             # reversed edge: faces the same way out
    return np.vstack([vertices, centres]), np.vstack([faces, caps]), int(count)


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
          report=print) -> Mesh:
    """`centre`/`radius` bound the subject (radius None or 0 keeps everything)."""
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
    capped_v, capped_f, n_capped = cap_holes(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    if n_capped:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(capped_v),
                                         o3d.utility.Vector3iVector(capped_f.astype(np.int32)))
    if len(mesh.triangles) > triangles:
        mesh = mesh.simplify_quadric_decimation(triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    colors = colour_from_cloud(vertices, np.asarray(pcd.points), np.asarray(pcd.colors))
    stats = {"points_fused": n_fused, "points_subject": n_cropped, "points_clean": n_clean,
             "vertices": len(vertices), "triangles": len(mesh.triangles),
             # closed = every edge shared by exactly two triangles. Open3D's is_watertight also
             # demands no self-intersection, which Poisson output rarely meets and slicers forgive
             "closed": bool(mesh.is_edge_manifold(allow_boundary_edges=False)), "pieces_removed": int(max(0, len(counts) - 1)),
             "openings_capped": n_capped}
    report(f"  mesh: {stats['triangles']:,} triangles, "
           f"{'closed' if stats['closed'] else 'NOT closed'}")
    return Mesh(vertices, np.asarray(mesh.triangles), colors, np.asarray(mesh.vertex_normals), stats)
