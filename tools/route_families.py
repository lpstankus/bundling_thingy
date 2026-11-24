# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Experimental shared-endpoint routing for exported bundling geometry."""

from pathlib import Path
import argparse
import heapq
import json
import subprocess
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import gaussian_filter1d

if __package__:
    from .family_lanes import FamilyLayout, apply_family_offsets, choose_family_offsets
else:
    from family_lanes import FamilyLayout, apply_family_offsets, choose_family_offsets

POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("edge", "<u4")])


def load_geometry(folder):
    for name, stride in [("points.bin", 12), ("paths.bin", 12), ("edges.bin", 16)]:
        size = (folder / name).stat().st_size
        if size == 0 or size % stride:
            raise ValueError(f"Invalid {name} record length.")
    raw = np.fromfile(folder / "points.bin", dtype=POINT_DTYPE)
    paths = np.fromfile(folder / "paths.bin", dtype="<u4").reshape(-1, 3)
    ends = np.fromfile(folder / "edges.bin", dtype="<f4").reshape(-1, 2, 2)
    if len(raw) > np.iinfo(np.uint32).max or len(paths) > np.iinfo(np.uint32).max:
        raise ValueError("Geometry indices exceed u32 storage.")
    if len(paths) != len(ends):
        raise ValueError("Path and edge counts differ.")
    if not np.isfinite(raw["x"]).all() or not np.isfinite(raw["y"]).all():
        raise ValueError("Non-finite geometry coordinates.")
    if np.any(np.abs(raw["x"]) > 1) or np.any(np.abs(raw["y"]) > 1):
        raise ValueError("Coordinates must be in clip space.")
    ranges = []
    for edge, (start, _, count) in enumerate(paths):
        start = int(start)
        count = int(count)
        if count == 0:
            continue
        if count < 2 or start + count > len(raw):
            raise ValueError(f"Invalid point range for edge {edge}.")
        points = raw[start : start + count]
        if not np.all(points["edge"] == edge):
            raise ValueError(f"Wrong edge IDs in path {edge}.")
        endpoint = np.column_stack([points["x"][[0, -1]], points["y"][[0, -1]]])
        if not np.array_equal(endpoint.view("<u4"), ends[edge].view("<u4")):
            raise ValueError(f"Edge {edge} endpoints differ from metadata.")
        ranges.append((start, start + count))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Path ranges overlap or leave gaps.")
        cursor = end
    if cursor != len(raw):
        raise ValueError("Point buffer has unassigned samples.")
    description = folder / "format.json"
    if description.exists():
        metadata = json.loads(description.read_text())
        if (
            metadata.get("version") != 1
            or metadata.get("width") != 1600
            or metadata.get("height") != 1600
            or metadata.get("coordinate_space") != "clip"
            or metadata.get("byte_order") != "little"
            or metadata.get("samples") != len(raw)
            or metadata.get("paths") != len(paths)
        ):
            raise ValueError(
                "This prototype expects version 1 geometry from the 1600 x 1600 headless renderer."
            )
    return raw, paths, ends


def write_geometry(folder, points, paths, ends):
    folder.mkdir(parents=True, exist_ok=True)
    points.tofile(folder / "points.bin")
    paths.tofile(folder / "paths.bin")
    ends.tofile(folder / "edges.bin")
    metadata = {
        "version": 1,
        "width": 1600,
        "height": 1600,
        "samples": len(points),
        "paths": len(paths),
        "coordinate_space": "clip",
        "byte_order": "little",
        "points.bin": "f32 x, f32 y, u32 edge_id",
        "paths.bin": "u32 point_offset, u32 segment_offset, u32 count",
        "edges.bin": "f32 source_x, f32 source_y, f32 destination_x, f32 destination_y",
    }
    (folder / "format.json").write_text(json.dumps(metadata, indent=2) + "\n")


def resample_for_consolidation(source, out):
    raw, paths, ends = load_geometry(source)
    xy = np.column_stack([raw["x"], raw["y"]]).astype(np.float64) * 799.5
    records = []
    values = []
    offset = 0
    segment = 0
    for edge, (start, _, count) in enumerate(paths):
        if count == 0:
            values.append(np.zeros(0, dtype=POINT_DTYPE))
            records.append([offset, segment, 0])
            continue
        v = xy[int(start) : int(start + count)]
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1))]
        keep = np.r_[True, np.diff(lengths) > 1e-6]
        v = v[keep]
        lengths = lengths[keep]
        along = np.r_[np.arange(0, lengths[-1], 4.0), lengths[-1]]
        sampled = np.column_stack(
            [np.interp(along, lengths, v[:, axis]) for axis in [0, 1]]
        )
        if len(sampled) < 2:
            sampled = np.repeat(sampled, 2, axis=0)
        record = np.zeros(len(sampled), dtype=POINT_DTYPE)
        record["x"] = sampled[:, 0] / 799.5
        record["y"] = sampled[:, 1] / 799.5
        record["edge"] = edge
        record["x"][[0, -1]] = ends[edge, :, 0]
        record["y"][[0, -1]] = ends[edge, :, 1]
        values.append(record)
        records.append([offset, segment, len(record)])
        offset += len(record)
        segment += len(record) - 1
    write_geometry(out, np.concatenate(values), np.asarray(records, dtype="<u4"), ends)


def run_renderer(command, log):
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=stream)
    if result.returncode:
        raise ValueError(f"Renderer failed. See {log}.")


def main():
    p = argparse.ArgumentParser(
        description="Refine exported geometry by routing paths with a shared endpoint neighborhood through common trees."
    )
    p.add_argument(
        "source", type=Path, help="Geometry directory produced by --export-geometry."
    )
    p.add_argument("out", type=Path, help="Directory for the refined geometry.")
    p.add_argument(
        "--endpoint-neighborhood",
        dest="radius",
        type=float,
        default=8,
        help="Maximum endpoint-family diameter in texels. Either end can establish membership. Default: 8.",
    )
    p.add_argument(
        "--lane-offset",
        dest="lane",
        type=float,
        default=2.5,
        help="Maximum sideways family offset in pixels. Default: 2.5.",
    )
    p.add_argument(
        "--lane-placement",
        choices=("coordinated", "stable"),
        default="coordinated",
        help="Choose offsets from sampled unrelated overlaps, or use the earlier stable offsets.",
    )
    p.add_argument(
        "--smoothing",
        dest="smooth",
        type=float,
        default=1.5,
        help="Route smoothing sigma in pixels. Default: 1.5.",
    )
    p.add_argument(
        "--consolidate-with",
        type=Path,
        help="Apply four consolidation updates and render the result with this bundling-wgpu executable.",
    )
    a = p.parse_args()
    a.grid = 1.0
    if (
        not np.isfinite([a.radius, a.lane, a.smooth]).all()
        or a.radius <= 0
        or a.lane < 0
        or a.smooth < 0
    ):
        p.error(
            "Neighborhood must be positive; lane offset and smoothing must be nonnegative and finite."
        )
    if a.source.resolve() == a.out.resolve():
        p.error("Source and output directories must differ.")
    initial_source = a.source
    if a.consolidate_with:
        a.out.mkdir(parents=True, exist_ok=True)
        resampled = a.out / "consolidation-input"
        resample_for_consolidation(a.source, resampled)
        consolidated = a.out / "consolidation"
        command = [
            str(a.consolidate_with.resolve()),
            "--headless",
            "--geometry-input",
            str(resampled),
            "--out",
            str(consolidated),
            "--neighbor-radius",
            "16",
            "--neighbor-radius-end",
            "10",
            "--iterations",
            "4",
            "--neighbor-weighting",
            "endpoint",
            "--neighbor-batch-size",
            "64",
            "--color-by",
            "edge-id",
            "--export-geometry",
            "--validate-gpu",
        ]
        print("Consolidating paths...", flush=True)
        run_renderer(command, a.out / "consolidation.log")
        a.source = consolidated / "geometry"

    raw, paths, ends = load_geometry(a.source)
    xy = np.column_stack([(raw["x"] + 1) * 800, (1 - raw["y"]) * 800]).astype(
        np.float64
    )
    endpoints = np.stack(
        [(ends[:, :, 0] + 1) * 800, (1 - ends[:, :, 1]) * 800], axis=2
    ).astype(np.float64)
    active = np.flatnonzero(paths[:, 2] > 0)
    print(f"Routing {len(active):,} drawable paths...", flush=True)
    flat = endpoints[active].reshape(-1, 2)
    n = len(paths)
    # Bounded families avoid merging an unrelated pair through a chain of shared endpoints.
    # Every member has one endpoint within half the neighborhood diameter of its anchor.
    neighbor = cKDTree(flat).query_ball_point(
        flat, a.radius * 0.5 * 1600 / 1599, workers=8
    )
    labels = np.full(len(flat), -1, np.int32)
    heap = [
        (-len(v), float(flat[i, 0]), float(flat[i, 1]), i)
        for i, v in enumerate(neighbor)
    ]
    heapq.heapify(heap)
    centers = []
    sizes = []
    while heap:
        old, x, y, center = heapq.heappop(heap)
        if labels[center] >= 0:
            continue
        members = np.array(neighbor[center])
        members = members[labels[members] < 0]
        if len(members) < -old:
            heapq.heappush(heap, (-len(members), x, y, center))
            continue
        labels[members] = len(centers)
        centers.append(flat[center])
        sizes.append(len(np.unique(members // 2)))
    centers = np.array(centers)
    sizes = np.array(sizes)
    groups = labels.reshape(len(active), 2)
    first_size = sizes[groups[:, 0]]
    last_size = sizes[groups[:, 1]]
    # Coordinate-based group order makes ties independent of path direction and input order.
    prefer_first = (first_size > last_size) | (
        (first_size == last_size) & (groups[:, 0] <= groups[:, 1])
    )
    active_side = np.where(prefer_first, 0, 1)
    side = np.zeros(n, dtype=np.int32)
    owner = np.full(n, -1, dtype=np.int32)
    side[active] = active_side
    owner[active] = groups[np.arange(len(active)), active_side]
    width = int(np.ceil(1600 / a.grid)) + 3

    def sample(v, spacing):
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1))]
        keep = np.r_[True, np.diff(lengths) > 1e-6]
        v = v[keep]
        lengths = lengths[keep]
        along = np.r_[np.arange(0, lengths[-1], spacing), lengths[-1]]
        return np.column_stack(
            [np.interp(along, lengths, v[:, axis]) for axis in [0, 1]]
        )

    def keys(v):
        q = np.rint(v / a.grid).astype(np.int64)
        result = q[:, 1] * width + q[:, 0]
        return result[np.r_[True, result[1:] != result[:-1]]]

    routes = [xy[int(start) : int(start + count)] for start, _, count in paths]
    preserved = np.ones(n, dtype=bool)
    families = 0
    accepted = 0
    fallback = 0
    ratios = []
    for group in np.unique(owner[active]):
        members = np.flatnonzero(owner == group)
        if len(members) < 2:
            edge = int(members[0])
            start, _, count = paths[edge]
            routes[edge] = xy[int(start) : int(start + count)]
            continue
        families += 1
        center = centers[group]
        root_key = int(keys(center[None])[0])
        sequences = []
        original = []
        for edge in members:
            start, _, count = paths[edge]
            v = xy[int(start) : int(start + count)]
            original.append(v)
            if side[edge]:
                v = v[::-1]
            sequence = keys(
                np.vstack(
                    (
                        sample(np.vstack((center, v[0])), a.grid * 0.75),
                        sample(v, a.grid * 0.75),
                    )
                )
            )
            sequences.append(sequence)
        all_keys = np.unique(np.concatenate(sequences))
        records = []
        for sequence in sequences:
            indices = np.searchsorted(all_keys, sequence)
            records.append(
                np.sort(np.column_stack((indices[:-1], indices[1:])), axis=1)
            )
        records = np.concatenate(records)
        encoded = records[:, 0] * len(all_keys) + records[:, 1]
        encoded = np.unique(encoded)
        source = encoded // len(all_keys)
        target = encoded % len(all_keys)
        locations = (
            np.column_stack((all_keys % width, all_keys // width)).astype(np.float64)
            * a.grid
        )
        cost = np.linalg.norm(locations[source] - locations[target], axis=1)
        graph = csr_matrix(
            (np.r_[cost, cost], (np.r_[source, target], np.r_[target, source])),
            shape=(len(all_keys), len(all_keys)),
        )
        root = int(np.searchsorted(all_keys, root_key))
        distances, predecessor = dijkstra(
            graph, directed=False, indices=root, return_predecessors=True
        )
        # Stable placement preserves the original family offset rule.
        lane = (
            a.lane * ((int(group) * 0.6180339887498949) % 1 * 2 - 1)
            if a.lane_placement == "stable"
            else 0.0
        )
        for edge, sequence, old in zip(members, sequences, original):
            if np.linalg.norm(endpoints[edge, 0] - endpoints[edge, 1]) < 16:
                routes[edge] = old
                fallback += 1
                continue
            cursor = int(np.searchsorted(all_keys, sequence[-1]))
            chain = []
            while cursor != root and cursor >= 0:
                chain.append(locations[cursor])
                cursor = int(predecessor[cursor])
            if cursor < 0:
                routes[edge] = old
                fallback += 1
                continue
            chain.append(locations[root])
            chain.reverse()
            v = sample(np.asarray(chain), 1.0)
            if len(v) > 2:
                if a.smooth > 0:
                    v = gaussian_filter1d(v, a.smooth, axis=0, mode="nearest")
                v[0] = center
                if lane != 0:
                    tangent = np.gradient(v, axis=0)
                    tangent /= np.maximum(
                        np.linalg.norm(tangent, axis=1)[:, None], 1e-8
                    )
                    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
                    along = np.r_[
                        0, np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1))
                    ]
                    fade = np.clip(np.minimum(along, along[-1] - along) / 24, 0, 1)
                    v += lane * fade[:, None] * normal
            v = np.vstack(
                (endpoints[edge, side[edge]], v, endpoints[edge, 1 - side[edge]])
            )
            if side[edge]:
                v = v[::-1]
            v[0] = endpoints[edge, 0]
            v[-1] = endpoints[edge, 1]
            old_length = np.linalg.norm(np.diff(old, axis=0), axis=1).sum()
            new_length = np.linalg.norm(np.diff(v, axis=0), axis=1).sum()
            if new_length > 1.15 * old_length + 8:
                routes[edge] = old
                fallback += 1
                continue
            routes[edge] = np.clip(v, 0, 1600)
            preserved[edge] = False
            accepted += 1
            ratios.append(new_length / max(old_length, 1e-6))
    values = []
    records = []
    offset = 0
    segment = 0
    for edge, v in enumerate(routes):
        if preserved[edge]:
            start, _, count = paths[edge]
            data = raw[int(start) : int(start + count)].copy()
        else:
            data = np.zeros(len(v), dtype=POINT_DTYPE)
            data["x"] = v[:, 0] / 800 - 1
            data["y"] = 1 - v[:, 1] / 800
            data["edge"] = edge
        if len(data):
            data["x"][[0, -1]] = ends[edge, :, 0]
            data["y"][[0, -1]] = ends[edge, :, 1]
        values.append(data)
        records.append([offset, segment, len(data)])
        offset += len(data)
        segment += max(len(data) - 1, 0)
    points = np.concatenate(values)
    path_records = np.asarray(records, dtype="<u4")
    optimizer = None
    offsets = np.zeros(len(centers), dtype=np.float32)
    if a.lane_placement == "coordinated":
        layout = FamilyLayout(
            points=points,
            paths=path_records,
            endpoints=ends,
            families=owner,
            root_sides=side,
            movable=~preserved,
            family_count=len(centers),
        )
        print("Coordinating family offsets...", flush=True)

        def report_progress(done, total):
            print(f"Checked {done:,} of {total:,} path positions.", flush=True)

        plan = choose_family_offsets(
            layout, neighborhood=a.radius, limit=a.lane, progress=report_progress
        )
        shifted = apply_family_offsets(layout, raw, paths, plan.offsets)
        points = shifted.points
        path_records = shifted.paths
        preserved = shifted.preserved
        ratios = shifted.length_ratios
        fallback += shifted.fallbacks
        accepted = int(np.count_nonzero(~preserved))
        offsets = plan.offsets
        optimizer = {
            "sampled_positions": plan.sampled_positions,
            "family_pairs": plan.family_pairs,
            "offset_fallbacks": shifted.fallbacks,
        }
    else:
        for group in range(len(centers)):
            offsets[group] = a.lane * ((group * 0.6180339887498949) % 1 * 2 - 1)

    a.out.mkdir(parents=True, exist_ok=True)
    write_geometry(a.out, points, path_records, ends)
    owner.astype("<u4").tofile(a.out / "families.bin")
    side.astype("<u4").tofile(a.out / "family-root-side.bin")
    centers.astype("<f4").tofile(a.out / "family-centers.bin")
    offsets.astype("<f4").tofile(a.out / "family-offsets.bin")
    metadata = {
        "algorithm_version": 3,
        "algorithm": "shared-endpoint routing on existing paths",
        "source": str(a.source),
        "initial_source": str(initial_source),
        "consolidation_updates": 4 if a.consolidate_with else 0,
        "consolidation": (
            {
                "iterations": 4,
                "initial_radius_texels": 16,
                "final_radius_texels": 10,
                "sample_spacing_texels": 4,
            }
            if a.consolidate_with
            else None
        ),
        "relationship_radius_texels": a.radius,
        "grid_px": a.grid,
        "lane_offset_px": a.lane,
        "lane_placement": a.lane_placement,
        "smoothing_sigma_px": a.smooth,
        "offset_optimizer": optimizer,
        "family_files": {
            "families.bin": "u32 family_id per edge; 4294967295 means inactive",
            "family-root-side.bin": "u32 per edge, 0 for source and 1 for destination",
            "family-centers.bin": "f32 x, f32 y per family in image pixels",
            "family-offsets.bin": "f32 signed offset per family in image pixels",
        },
        "families_routed": families,
        "accepted_edges": accepted,
        "fallback_edges": fallback,
        "length_ratio_median": float(np.median(ratios)) if ratios else 1.0,
        "length_ratio_p95": float(np.quantile(ratios, 0.95)) if ratios else 1.0,
    }
    (a.out / "algorithm.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    if a.consolidate_with:
        print("Rendering family routes...", flush=True)
        run_renderer(
            [
                str(a.consolidate_with.resolve()),
                "--headless",
                "--geometry-input",
                str(a.out),
                "--out",
                str(a.out / "render"),
                "--color-by",
                "edge-id",
                "--validate-gpu",
            ],
            a.out / "render.log",
        )
        (a.out / "family-routes.png").write_bytes(
            (a.out / "render" / "original.png").read_bytes()
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
