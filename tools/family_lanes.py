"""Choose compact lateral offsets for shared-endpoint route families."""

from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FamilyLayout:
    # Arrays use the exported GPU geometry layout. Endpoints remain in clip coordinates.
    points: np.ndarray
    paths: np.ndarray
    endpoints: np.ndarray
    families: np.ndarray
    root_sides: np.ndarray
    movable: np.ndarray
    family_count: int


@dataclass(frozen=True)
class LanePlan:
    offsets: np.ndarray
    sampled_positions: int
    family_pairs: int


@dataclass(frozen=True)
class ShiftedGeometry:
    points: np.ndarray
    paths: np.ndarray
    preserved: np.ndarray
    length_ratios: list[float]
    fallbacks: int


@dataclass(frozen=True)
class Samples:
    points: np.ndarray
    normals: np.ndarray
    fades: np.ndarray
    edges: np.ndarray
    valid: np.ndarray
    queries: np.ndarray
    query_weights: np.ndarray


def _canonical_reverse(points: np.ndarray) -> bool:
    """Orient a fixed path consistently, including a closed path."""
    different = np.flatnonzero(np.any(points != points[::-1], axis=1))
    if len(different) == 0:
        return False
    index = int(different[0])
    return tuple(points[index]) > tuple(points[-1 - index])


def _sample_paths(layout: FamilyLayout) -> Samples | None:
    positions = []
    normals = []
    fades = []
    edge_ids = []
    valid = []
    queries = []
    query_weights = []
    offset = 0
    for edge, (start, _, count) in enumerate(layout.paths):
        if count < 2:
            continue
        start = int(start)
        count = int(count)
        data = layout.points[start : start + count]
        vertices = np.column_stack(
            [(data["x"] + 1) * 800, (1 - data["y"]) * 800]
        ).astype(np.float64)
        reverse = (
            bool(layout.root_sides[edge])
            if layout.movable[edge]
            else _canonical_reverse(vertices)
        )
        if reverse:
            vertices = vertices[::-1]
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
        keep = np.r_[True, np.diff(lengths) > 1e-6]
        vertices = vertices[keep]
        lengths = lengths[keep]
        if len(vertices) < 2:
            continue
        along = np.arange(0, lengths[-1], 2.0)
        if len(along) < 12:
            continue
        points = np.column_stack(
            [np.interp(along, lengths, vertices[:, axis]) for axis in (0, 1)]
        ).astype(np.float32)
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-10)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        fade = np.clip(np.minimum(along, lengths[-1] - along) / 24, 0, 1).astype(
            np.float32
        )
        local_valid = np.zeros(len(points), dtype=bool)
        local_valid[5:-5] = True
        # Two deterministic positions per path estimate the conflicts without using edge order.
        local = np.unique(
            np.clip([len(points) // 3, 2 * len(points) // 3], 5, len(points) - 6)
        )
        queries.extend(offset + local)
        query_weights.extend([max(1, round(lengths[-1] / len(local)))] * len(local))
        positions.append(points)
        normals.append(normal)
        fades.append(fade)
        edge_ids.append(np.full(len(points), edge, dtype=np.int32))
        valid.append(local_valid)
        offset += len(points)
    if not positions:
        return None
    return Samples(
        points=np.concatenate(positions),
        normals=np.concatenate(normals),
        fades=np.concatenate(fades),
        edges=np.concatenate(edge_ids),
        valid=np.concatenate(valid),
        queries=np.asarray(queries, dtype=np.int64),
        query_weights=np.asarray(query_weights, dtype=np.int64),
    )


def _conflict_costs(
    layout: FamilyLayout,
    samples: Samples,
    neighborhood: float,
    limit: float,
    progress: Callable[[int, int], None] | None,
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    endpoints = np.stack(
        [(layout.endpoints[:, :, 0] + 1) * 800, (1 - layout.endpoints[:, :, 1]) * 800],
        axis=2,
    )
    tree = cKDTree(samples.points)
    states = np.asarray([-limit, 0, limit], dtype=np.float32)
    steps = np.arange(-3, 4)
    cosine = np.cos(np.deg2rad(15))
    search = 4 + 2 * limit
    costs_by_pair: dict[tuple[int, int], np.ndarray] = {}
    edge_pairs = 0
    for start in range(0, len(samples.queries), 256):
        batch = samples.queries[start : start + 256]
        neighbors = tree.query_ball_point(
            samples.points[batch], r=search, workers=8, return_sorted=False
        )
        for query, nearby, importance in zip(
            batch, neighbors, samples.query_weights[start : start + 256]
        ):
            candidates = np.asarray(nearby, dtype=np.int64)
            candidates = candidates[
                (samples.edges[candidates] != samples.edges[query])
                & samples.valid[candidates]
            ]
            if not len(candidates):
                continue
            alignment = np.sum(
                samples.normals[candidates] * samples.normals[query], axis=1
            )
            keep = np.abs(alignment) >= cosine
            candidates, alignment = candidates[keep], alignment[keep]
            if not len(candidates):
                continue
            tangent = np.array([samples.normals[query, 1], -samples.normals[query, 0]])
            along = np.sum(
                (samples.points[query] - samples.points[candidates]) * tangent, axis=1
            )
            keep = np.abs(along) <= 4
            candidates, alignment = candidates[keep], alignment[keep]
            if not len(candidates):
                continue
            edge = samples.edges[query]
            others = samples.edges[candidates]
            family = int(layout.families[edge])
            other_families = layout.families[others]
            delta = endpoints[edge][None, :, None, :] - endpoints[others][:, None, :, :]
            closest = (
                np.sqrt(np.sum(delta * delta, axis=-1).min(axis=(1, 2))) * 1599 / 1600
            )
            keep = (
                (closest > neighborhood)
                & (other_families != family)
                & (layout.movable[edge] | layout.movable[others])
            )
            candidates, alignment, others, other_families = (
                candidates[keep],
                alignment[keep],
                others[keep],
                other_families[keep],
            )
            if not len(candidates):
                continue
            direction = np.where(alignment >= 0, 1, -1)
            other_steps = candidates[:, None] + direction[:, None] * steps[None, :]
            delta = (
                samples.points[query + steps][None, :, :] - samples.points[other_steps]
            )
            keep = np.max(np.sum(delta * delta, axis=2), axis=1) <= search * search
            candidates, others, other_families, other_steps, delta = (
                candidates[keep],
                others[keep],
                other_families[keep],
                other_steps[keep],
                delta[keep],
            )
            if not len(candidates):
                continue
            query_shift = (
                samples.normals[query + steps][None, :, :]
                * samples.fades[query + steps][None, :, None]
                * float(layout.movable[edge])
            )
            other_shift = (
                samples.normals[other_steps]
                * samples.fades[other_steps][:, :, None]
                * layout.movable[others][:, None, None]
            )
            cost = np.empty((len(candidates), 3, 3), dtype=np.uint8)
            for i, first in enumerate(states):
                for j, second in enumerate(states):
                    shifted = delta + first * query_shift - second * other_shift
                    farthest = np.max(np.sum(shifted * shifted, axis=2), axis=1)
                    # Penalize co-runs within 4 px, with five extra points within 2 px.
                    cost[:, i, j] = (farthest <= 16) + 5 * (farthest <= 4)
            # A path contributes at most once per sampled position and lane pair.
            order = np.argsort(others, kind="stable")
            others, other_families, cost = (
                others[order],
                other_families[order],
                cost[order],
            )
            starts = np.r_[0, np.flatnonzero(np.diff(others) != 0) + 1]
            cost = np.maximum.reduceat(cost, starts, axis=0)
            other_families = other_families[starts]
            unique, inverse = np.unique(other_families, return_inverse=True)
            aggregate = np.zeros((len(unique), 3, 3), dtype=np.int64)
            for i in range(3):
                for j in range(3):
                    aggregate[:, i, j] = (
                        np.bincount(
                            inverse, weights=cost[:, i, j], minlength=len(unique)
                        ).astype(np.int64)
                        * importance
                    )
            for other, pair_cost in zip(unique, aggregate):
                if not pair_cost.any():
                    continue
                other = int(other)
                low, high = min(family, other), max(family, other)
                key = (low, high)
                if key not in costs_by_pair:
                    costs_by_pair[key] = np.zeros((3, 3), dtype=np.int64)
                costs_by_pair[key] += pair_cost if low == family else pair_cost.T
            edge_pairs += len(starts)
        done = start + len(batch)
        if progress is not None and (done == len(samples.queries) or done % 4096 == 0):
            progress(done, len(samples.queries))
    return costs_by_pair, edge_pairs


def _choose_states(
    family_count: int, pairs: dict[tuple[int, int], np.ndarray], active: np.ndarray
) -> np.ndarray:
    adjacent: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(family_count)]
    for (first, second), cost in pairs.items():
        adjacent[first].append((second, cost))
        adjacent[second].append((first, cost.T))
    degree = np.asarray(
        [sum(int(cost.max()) for _, cost in row) for row in adjacent], dtype=np.int64
    )
    order = np.argsort(-degree, kind="stable")
    best_score = None
    best = np.ones(family_count, dtype=np.int8)
    for initial in range(4):
        labels = np.ones(family_count, dtype=np.int8)
        if initial == 1:
            labels[active] = 0
        elif initial == 2:
            labels[active] = 2
        elif initial == 3:
            labels[active] = (np.arange(family_count) * 2654435761 % 3).astype(np.int8)[
                active
            ]
        for _ in range(40):
            changes = 0
            for family in order:
                if not active[family] or not adjacent[family]:
                    continue
                values = np.zeros(3, dtype=np.int64)
                for other, cost in adjacent[family]:
                    values += cost[:, labels[other]]
                current = int(labels[family])
                choice = min(
                    range(3), key=lambda index: (values[index], abs(index - 1), index)
                )
                if values[choice] < values[current] or (
                    values[choice] == values[current]
                    and (abs(choice - 1), choice) < (abs(current - 1), current)
                ):
                    labels[family] = choice
                    changes += 1
            if changes == 0:
                break
        score = sum(
            int(cost[labels[first], labels[second]])
            for (first, second), cost in pairs.items()
        )
        if best_score is None or score < best_score:
            best_score = score
            best = labels.copy()
    return best


def choose_family_offsets(
    layout: FamilyLayout,
    *,
    neighborhood: float,
    limit: float,
    progress: Callable[[int, int], None] | None = None,
) -> LanePlan:
    offsets = np.zeros(layout.family_count, dtype=np.float32)
    if limit == 0 or not layout.movable.any():
        return LanePlan(offsets, 0, 0)
    samples = _sample_paths(layout)
    if samples is None:
        return LanePlan(offsets, 0, 0)
    pairs, _ = _conflict_costs(layout, samples, neighborhood, limit, progress)
    active = (
        np.bincount(
            layout.families[layout.movable].astype(np.int64),
            minlength=layout.family_count,
        )
        > 0
    )
    labels = _choose_states(layout.family_count, pairs, active)
    offsets = np.asarray([-limit, 0, limit], dtype=np.float32)[labels]
    offsets[~active] = 0
    return LanePlan(offsets, len(samples.queries), len(pairs))


def apply_family_offsets(
    layout: FamilyLayout,
    input_points: np.ndarray,
    input_paths: np.ndarray,
    offsets: np.ndarray,
) -> ShiftedGeometry:
    endpoint_pixels = np.stack(
        [(layout.endpoints[:, :, 0] + 1) * 800, (1 - layout.endpoints[:, :, 1]) * 800],
        axis=2,
    ).astype(np.float64)
    values = []
    records = []
    preserved = ~layout.movable.copy()
    ratios = []
    fallbacks = 0
    offset = 0
    segment = 0
    for edge, ((start, _, count), (old_start, _, old_count)) in enumerate(
        zip(layout.paths, input_paths)
    ):
        data = layout.points[int(start) : int(start + count)].copy()
        old = input_points[int(old_start) : int(old_start + old_count)]
        if layout.movable[edge]:
            lane = offsets[layout.families[edge]]
            if lane != 0:
                vertices = np.column_stack(
                    [(data["x"] + 1) * 800, (1 - data["y"]) * 800]
                ).astype(np.float64)
                if layout.root_sides[edge]:
                    vertices = vertices[::-1]
                interior = vertices[1:-1].copy()
                if len(interior) > 2:
                    tangent = np.gradient(interior, axis=0)
                    tangent /= np.maximum(
                        np.linalg.norm(tangent, axis=1)[:, None], 1e-8
                    )
                    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
                    along = np.r_[
                        0, np.cumsum(np.linalg.norm(np.diff(interior, axis=0), axis=1))
                    ]
                    fade = np.clip(np.minimum(along, along[-1] - along) / 24, 0, 1)
                    interior += lane * fade[:, None] * normal
                vertices = np.vstack((vertices[0], interior, vertices[-1]))
                if layout.root_sides[edge]:
                    vertices = vertices[::-1]
                vertices[0], vertices[-1] = (
                    endpoint_pixels[edge, 0],
                    endpoint_pixels[edge, 1],
                )
                before = np.column_stack(
                    [(old["x"] + 1) * 800, (1 - old["y"]) * 800]
                ).astype(np.float64)
                old_length = np.linalg.norm(np.diff(before, axis=0), axis=1).sum()
                new_length = np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
                if new_length > 1.15 * old_length + 8:
                    data = old.copy()
                    preserved[edge] = True
                    fallbacks += 1
                else:
                    vertices = np.clip(vertices, 0, 1600)
                    data["x"] = vertices[:, 0] / 800 - 1
                    data["y"] = 1 - vertices[:, 1] / 800
            if not preserved[edge]:
                current = np.column_stack(
                    [(data["x"] + 1) * 800, (1 - data["y"]) * 800]
                ).astype(np.float64)
                before = np.column_stack(
                    [(old["x"] + 1) * 800, (1 - old["y"]) * 800]
                ).astype(np.float64)
                new_length = np.linalg.norm(np.diff(current, axis=0), axis=1).sum()
                old_length = np.linalg.norm(np.diff(before, axis=0), axis=1).sum()
                ratios.append(float(new_length / max(old_length, 1e-6)))
        if len(data):
            data["x"][[0, -1]] = layout.endpoints[edge, :, 0]
            data["y"][[0, -1]] = layout.endpoints[edge, :, 1]
        values.append(data)
        records.append([offset, segment, len(data)])
        offset += len(data)
        segment += max(len(data) - 1, 0)
    return ShiftedGeometry(
        np.concatenate(values),
        np.asarray(records, dtype="<u4"),
        preserved,
        ratios,
        fallbacks,
    )
