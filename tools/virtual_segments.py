# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Bundle cell-local path groups around the mean of their member endpoints.

Groups are rebuilt each pass. A candidate joins if either of its endpoints is
near the group's current identity; the mean does not guarantee that every pair
of members has nearby endpoints.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .route_families import POINT_DTYPE, load_geometry, write_geometry
else:
    from route_families import POINT_DTYPE, load_geometry, write_geometry


@dataclass(frozen=True)
class Geometry:
    points: np.ndarray
    paths: np.ndarray
    endpoints: np.ndarray
    positions: np.ndarray
    endpoint_pixels: np.ndarray
    movable: np.ndarray


@dataclass
class SegmentGroup:
    first: int
    first_endpoints: np.ndarray
    size: int = 1
    endpoint_sum: tuple[float, float] = (0.0, 0.0)
    anchor: tuple[float, float] = (0.0, 0.0)

    def identities(self, mode: str) -> tuple[tuple[float, float], ...]:
        if self.size == 1:
            return tuple(tuple(float(v) for v in end) for end in self.first_endpoints)
        if mode == "anchor":
            return (self.anchor,)
        return ((self.endpoint_sum[0] / self.size, self.endpoint_sum[1] / self.size),)


@dataclass(frozen=True)
class Segments:
    record_for_sample: np.ndarray
    routes: np.ndarray
    positions: np.ndarray
    cells: np.ndarray


@dataclass(frozen=True)
class Grouping:
    group_for_record: np.ndarray
    side_for_record: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray


def resample(source: Path, spacing: float) -> Geometry:
    raw, paths, endpoints = load_geometry(source)
    values = []
    records = []
    movable = []
    offset = 0
    segments = 0
    for edge, (start, _, count) in enumerate(paths):
        data = raw[int(start) : int(start + count)]
        if count == 0:
            points = np.zeros(0, dtype=POINT_DTYPE)
        else:
            pixels = np.column_stack(
                [(data["x"] + 1) * 800, (1 - data["y"]) * 800]
            ).astype(np.float64)
            lengths = np.r_[
                0, np.cumsum(np.linalg.norm(np.diff(pixels, axis=0), axis=1))
            ]
            keep = np.r_[True, np.diff(lengths) > 1e-6]
            pixels = pixels[keep]
            lengths = lengths[keep]
            if len(pixels) == 1:
                sampled = np.repeat(pixels, 2, axis=0)
            else:
                sample_count = math.ceil(lengths[-1] / spacing) + 1
                along = np.linspace(0, lengths[-1], sample_count)
                sampled = np.column_stack(
                    [np.interp(along, lengths, pixels[:, axis]) for axis in (0, 1)]
                )
            points = np.zeros(len(sampled), dtype=POINT_DTYPE)
            points["x"] = sampled[:, 0] / 800 - 1
            points["y"] = 1 - sampled[:, 1] / 800
            points["edge"] = edge
            points["x"][[0, -1]] = endpoints[edge, :, 0]
            points["y"][[0, -1]] = endpoints[edge, :, 1]
        active = np.ones(len(points), dtype=bool)
        if len(active):
            active[[0, -1]] = False
        values.append(points)
        movable.append(active)
        records.append([offset, segments, len(points)])
        offset += len(points)
        segments += max(len(points) - 1, 0)
    points = np.concatenate(values)
    positions = np.column_stack(
        [(points["x"] + 1) * 800, (1 - points["y"]) * 800]
    ).astype(np.float32)
    endpoint_pixels = np.stack(
        [(endpoints[:, :, 0] + 1) * 800, (1 - endpoints[:, :, 1]) * 800],
        axis=2,
    ).astype(np.float32)
    return Geometry(
        points,
        np.asarray(records, dtype="<u4"),
        endpoints,
        positions,
        endpoint_pixels,
        np.concatenate(movable),
    )


def canonical_endpoints(endpoints: np.ndarray) -> np.ndarray:
    first, last = endpoints[:, 0], endpoints[:, 1]
    reverse = (first[:, 0] > last[:, 0]) | (
        (first[:, 0] == last[:, 0]) & (first[:, 1] > last[:, 1])
    )
    return np.where(reverse[:, None, None], endpoints[:, ::-1], endpoints)


def local_segments(
    geometry: Geometry, positions: np.ndarray, radius: float, iteration: int
) -> Segments:
    # Every pair in a cell is at most radius apart. Alternate the cell origin.
    cell_size = radius / math.sqrt(2)
    shifts = ((0, 0), (0.5, 0.5), (0.5, 0), (0, 0.5))
    shift = np.asarray(shifts[iteration % len(shifts)]) * cell_size
    grid = np.floor((positions + shift) / cell_size).astype(np.int64)
    width = math.ceil(1600 / cell_size) + 2
    cells = grid[:, 1] * width + grid[:, 0]
    edge_ids = geometry.points["edge"]
    order = np.lexsort((edge_ids, cells))
    ordered_cells = cells[order]
    ordered_edges = edge_ids[order]
    starts = np.flatnonzero(
        np.r_[
            True,
            (np.diff(ordered_cells) != 0) | (np.diff(ordered_edges) != 0),
        ]
    )
    # Each route contributes once per cell, even when it has multiple samples there.
    counts = np.diff(np.r_[starts, len(order)])
    sums = np.add.reduceat(positions[order].astype(np.float64), starts, axis=0)
    record_for_sample = np.empty(len(positions), dtype=np.int64)
    record_for_sample[order] = np.repeat(np.arange(len(starts)), counts)
    return Segments(
        record_for_sample,
        ordered_edges[starts],
        (sums / counts[:, None]).astype(np.float32),
        ordered_cells[starts],
    )


def bucket(point: tuple[float, float], radius: float) -> tuple[int, int]:
    return (math.floor(point[0] / radius), math.floor(point[1] / radius))


def update_buckets(
    bins: dict[tuple[int, int], set[int]],
    group_id: int,
    identities: tuple[tuple[float, float], ...],
    radius: float,
    *,
    add: bool,
) -> None:
    for key in {bucket(point, radius) for point in identities}:
        if add:
            bins.setdefault(key, set()).add(group_id)
        else:
            bins[key].remove(group_id)
            if not bins[key]:
                del bins[key]


def best_group(
    ends: np.ndarray,
    groups: list[SegmentGroup],
    bins: dict[tuple[int, int], set[int]],
    radius: float,
    mode: str,
) -> tuple[int, int, int] | None:
    candidates = set()
    for end in ends:
        x, y = bucket(end, radius)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidates.update(bins.get((x + dx, y + dy), ()))
    best = None
    for group_id in candidates:
        for group_side, identity in enumerate(groups[group_id].identities(mode)):
            for side, end in enumerate(ends):
                distance = (float(end[0]) - identity[0]) ** 2 + (
                    float(end[1]) - identity[1]
                ) ** 2
                if distance <= radius * radius:
                    candidate = (distance, group_id, side, group_side)
                    if best is None or candidate < best:
                        best = candidate
    if best is None:
        return None
    return best[1:]


def group_segments(
    segments: Segments, endpoints: np.ndarray, radius: float, mode: str
) -> Grouping:
    ends = endpoints[segments.routes]
    first, last = ends[:, 0], ends[:, 1]
    order = np.lexsort(
        (
            segments.positions[:, 1],
            segments.positions[:, 0],
            last[:, 1],
            last[:, 0],
            first[:, 1],
            first[:, 0],
            segments.cells,
        )
    )
    groups: list[SegmentGroup] = []
    bins: dict[tuple[int, int], set[int]] = {}
    group_for_record = np.empty(len(order), dtype=np.int64)
    side_for_record = np.full(len(order), -1, dtype=np.int8)
    previous_cell = None
    for processed, record in enumerate(order, 1):
        cell = int(segments.cells[record])
        if cell != previous_cell:
            bins = {}
            previous_cell = cell
        match = best_group(ends[record], groups, bins, radius, mode)
        if match is None:
            group_id = len(groups)
            group = SegmentGroup(int(record), ends[record])
            groups.append(group)
        else:
            group_id, side, old_side = match
            group = groups[group_id]
            update_buckets(bins, group_id, group.identities(mode), radius, add=False)
            selected = ends[record, side]
            if group.size == 1:
                anchor = group.first_endpoints[old_side]
                side_for_record[group.first] = old_side
                group.anchor = (float(anchor[0]), float(anchor[1]))
                group.endpoint_sum = group.anchor
            group.endpoint_sum = (
                group.endpoint_sum[0] + float(selected[0]),
                group.endpoint_sum[1] + float(selected[1]),
            )
            # In virtual mode the next candidate is compared with the updated mean.
            group.size += 1
            side_for_record[record] = side
        group_for_record[record] = group_id
        update_buckets(bins, group_id, group.identities(mode), radius, add=True)
        if processed % 100_000 == 0 or processed == len(order):
            print(f"  Grouped {processed:,}/{len(order):,} local routes.", flush=True)
    centers = np.asarray(
        [group.identities(mode)[0] for group in groups], dtype=np.float32
    )
    sizes = np.asarray([group.size for group in groups], dtype=np.uint32)
    side_for_record[side_for_record < 0] = 0
    return Grouping(group_for_record, side_for_record, centers, sizes)


def move_segments(
    geometry: Geometry,
    positions: np.ndarray,
    segments: Segments,
    grouping: Grouping,
    endpoints: np.ndarray,
    step: float,
) -> np.ndarray:
    group = grouping.group_for_record
    ends = endpoints[segments.routes, grouping.side_for_record]
    distance = np.linalg.norm(ends - grouping.centers[group], axis=1)
    weight = 1 / (1 + distance)
    total_weight = np.bincount(group, weights=weight, minlength=len(grouping.sizes))
    total = np.column_stack(
        [
            np.bincount(
                group,
                weights=weight * segments.positions[:, axis],
                minlength=len(grouping.sizes),
            )
            for axis in (0, 1)
        ]
    )
    remaining = total_weight[group] - weight
    target = segments.positions.astype(np.float64)
    shared = remaining > 0
    target[shared] = (
        total[group[shared]] - weight[shared, None] * segments.positions[shared]
    ) / remaining[shared, None]
    sample_target = target[segments.record_for_sample]
    result = positions + step * (sample_target - positions)
    result[~geometry.movable] = positions[~geometry.movable]
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exported 1600 x 1600 geometry.")
    parser.add_argument("out", type=Path, help="Output geometry directory.")
    parser.add_argument(
        "--identity",
        choices=("virtual", "anchor"),
        default="virtual",
        help="Compare with the group mean or its first endpoint. Default: virtual.",
    )
    parser.add_argument(
        "--endpoint-radius",
        type=float,
        default=8,
        help="Maximum distance to the group identity in pixels. Default: 8.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=16,
        help="Initial spatial radius in pixels. Default: 16.",
    )
    parser.add_argument(
        "--radius-end",
        type=float,
        default=8,
        help="Final spatial radius in pixels. Default: 8.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=6,
        help="Number of movement passes. Default: 6.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=12,
        help="Initial sample spacing in pixels. Default: 12.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1,
        help="Fraction of the way to the mean. Default: 1.",
    )
    args = parser.parse_args()
    numbers = [
        args.endpoint_radius,
        args.radius,
        args.radius_end,
        args.spacing,
        args.step,
    ]
    if (
        not np.isfinite(numbers).all()
        or min(numbers) <= 0
        or args.step > 1
        or args.iterations <= 0
    ):
        parser.error(
            "Radii, spacing, and iterations must be positive; step must be in (0, 1]."
        )
    if args.source.resolve() == args.out.resolve():
        parser.error("Source and output directories must differ.")
    geometry = resample(args.source, args.spacing)
    args.out.mkdir(parents=True, exist_ok=True)
    write_geometry(
        args.out / "initial", geometry.points, geometry.paths, geometry.endpoints
    )
    endpoints = canonical_endpoints(geometry.endpoint_pixels)
    positions = geometry.positions.copy()
    radii = np.linspace(args.radius, args.radius_end, args.iterations)
    print(
        f"Bundling {len(geometry.paths):,} routes and {len(positions):,} samples.",
        flush=True,
    )
    for iteration, radius in enumerate(radii):
        print(
            f"Iteration {iteration + 1}/{args.iterations}, radius {radius:.2f} px.",
            flush=True,
        )
        segments = local_segments(geometry, positions, radius, iteration)
        grouping = group_segments(
            segments, endpoints, args.endpoint_radius, args.identity
        )
        positions = move_segments(
            geometry, positions, segments, grouping, endpoints, args.step
        )
        print(
            f"  {np.count_nonzero(grouping.sizes > 1):,} shared local groups.",
            flush=True,
        )
    result = geometry.points.copy()
    result["x"] = np.clip(positions[:, 0] / 800 - 1, -1, 1)
    result["y"] = np.clip(1 - positions[:, 1] / 800, -1, 1)
    for edge, (start, _, count) in enumerate(geometry.paths):
        if count:
            indices = [int(start), int(start + count - 1)]
            result["x"][indices] = geometry.endpoints[edge, :, 0]
            result["y"][indices] = geometry.endpoints[edge, :, 1]
    write_geometry(args.out, result, geometry.paths, geometry.endpoints)
    record_group = grouping.group_for_record[segments.record_for_sample]
    grouping.centers[record_group].astype("<f4").tofile(
        args.out / "virtual-endpoints.bin"
    )
    grouping.sizes[record_group].astype("<u4").tofile(args.out / "virtual-support.bin")
    record_group.astype("<u4").tofile(args.out / "segment-groups.bin")
    sample_records = segments.record_for_sample
    chosen = endpoints[
        segments.routes[sample_records], grouping.side_for_record[sample_records]
    ]
    chosen.astype("<f4").tofile(args.out / "member-endpoints.bin")
    metadata = {
        "algorithm": "shared local segment with endpoint identity",
        "identity": args.identity,
        "source": str(args.source),
        "iterations": args.iterations,
        "radii_px": radii.tolist(),
        "endpoint_radius_px": args.endpoint_radius,
        "sample_spacing_px": args.spacing,
        "step": args.step,
        "identity_scope": "one shared endpoint identity per spatial cell and route group",
        "virtual-endpoints.bin": "f32 x, f32 y in image pixels per sample before the final movement",
        "virtual-support.bin": "u32 distinct routes in the selected group per sample",
        "segment-groups.bin": "u32 last-iteration local group ID per sample",
        "member-endpoints.bin": "f32 x, f32 y of the member endpoint chosen for the group, in image pixels per sample",
    }
    (args.out / "algorithm.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
