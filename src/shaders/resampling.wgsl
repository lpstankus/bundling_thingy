override block_size: u32;
override accumulate_density: bool;

struct Parameters {
    minimum: vec2<f32>,
    scale: f32,
    path_count: u32,
    translation: vec2<f32>,
    capacity: u32,
    padding: u32,
}

struct Draw {
    vertices: u32,
    instances: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct State {
    points: atomic<u32>,
    segments: atomic<u32>,
    endpoints: atomic<u32>,
    overflow: atomic<u32>,
    path_draw: Draw,
    endpoint_draw: Draw,
}

struct Cursor {
    count: u32,
    last: vec2<f32>,
    edge_id: u32,
}

@group(0) @binding(0) var<storage, read_write> control_points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> paths: array<Path>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<uniform> parameters: Parameters;
@group(0) @binding(5) var<storage, read_write> points: array<Point>;
@group(0) @binding(6) var<storage, read_write> segments: array<u32>;
@group(0) @binding(7) var<storage, read_write> edges: array<Edge>;
@group(0) @binding(8) var<storage, read_write> density: array<atomic<u32>>;

fn normalized(index: u32) -> vec2<f32> {
    return ((control_points[index] - parameters.minimum) * parameters.scale
        + parameters.translation - vec2<f32>(0.5)) * 2.0;
}

fn push_sample(cursor: ptr<function, Cursor>, point: vec2<f32>, path: Path, write: bool) {
    if (*cursor).count > 0u && all(point == (*cursor).last) {
        return;
    }
    if !write && (*cursor).count >= parameters.capacity {
        (*cursor).count = parameters.capacity + 1u;
        return;
    }
    if write {
        if (*cursor).count >= path.count {
            atomicStore(&state.overflow, 2u);
            (*cursor).count = parameters.capacity + 1u;
            return;
        }
        let index = path.point_offset + (*cursor).count;
        points[index] = Point(point.x, point.y, (*cursor).edge_id);
        if (*cursor).count > 0u {
            segments[path.segment_offset + (*cursor).count - 1u] = index - 1u;
        }
        if accumulate_density {
            // Normalization keeps every sample inside the map.
            let uv = vec2<f32>(point.x * 0.5 + 0.5, 0.5 - point.y * 0.5);
            let last = vec2<f32>(map.size - vec2<u32>(1u));
            let cell = vec2<u32>(round(uv * last));
            atomicAdd(&density[cell.y * map.size.x + cell.x], 1u);
        }
    }
    (*cursor).last = point;
    (*cursor).count += 1u;
}

// Counting and writing share the same spacing, duplicate, and endpoint rules.
fn resample(path_index: u32, write: bool) -> u32 {
    let start = starts[path_index];
    let end = starts[path_index + 1u];
    let path = paths[path_index];
    let first = normalized(start);
    let texel_scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
    var cursor = Cursor(0u, first, path_index);
    push_sample(&cursor, first, path, write);
    var previous = first;
    var remaining = 1.0;
    for (var index = start + 1u; index < end; index += 1u) {
        let current = normalized(index);
        let delta = current - previous;
        let distance = length(delta * texel_scale);
        if distance > 0.0 {
            let steps = u32(max(floor(distance - remaining) + 1.0, 0.0));
            if write {
                for (var step = 0u; step < steps; step += 1u) {
                    let along = remaining + f32(step);
                    push_sample(&cursor, previous + delta * (along / distance), path, write);
                    if cursor.count > parameters.capacity {
                        return cursor.count;
                    }
                }
            } else if steps > 0u {
                let first_sample = previous + delta * (remaining / distance);
                let last_sample = previous + delta * ((remaining + f32(steps - 1u)) / distance);
                // Samples are a texel apart. Only the first can repeat the prior sample.
                cursor.count += steps - select(0u, 1u, all(first_sample == cursor.last));
                cursor.last = last_sample;
                if cursor.count > parameters.capacity {
                    return parameters.capacity + 1u;
                }
            }
            remaining = remaining + f32(steps) - distance;
        }
        previous = current;
    }
    if !write {
        edges[path_index] = Edge(first, previous);
    }
    push_sample(&cursor, previous, path, write);
    // A short closed path still needs its excursion and returning endpoint.
    if cursor.count == 1u {
        for (var index = start + 1u; index < end; index += 1u) {
            let next = normalized(index);
            if any(next != first) {
                push_sample(&cursor, next, path, write);
                push_sample(&cursor, previous, path, write);
                break;
            }
        }
    }
    return cursor.count;
}

@compute @workgroup_size(block_size)
fn count_paths(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index >= parameters.path_count {
        return;
    }
    let count = resample(index, false);
    if count < 2u {
        return;
    }
    let offset = atomicAdd(&state.points, count);
    // Any capacity failure disables the entire emit pass, even if totals wrap.
    if offset >= parameters.capacity || count > parameters.capacity - offset {
        atomicStore(&state.overflow, 1u);
        return;
    }
    paths[index] = Path(offset, atomicAdd(&state.segments, count - 1u), count);
    atomicAdd(&state.endpoints, 2u);
}

@compute @workgroup_size(block_size)
fn emit_paths(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index >= parameters.path_count {
        return;
    }
    let overflow = atomicLoad(&state.overflow) != 0u;
    if index == 0u {
        state.path_draw = Draw(4u, select(atomicLoad(&state.segments), 0u, overflow), 0u, 0u);
        state.endpoint_draw = Draw(4u, select(atomicLoad(&state.endpoints), 0u, overflow), 0u, 0u);
    }
    let path = paths[index];
    if overflow || path.count < 2u {
        return;
    }
    let written = resample(index, true);
    if written != path.count {
        atomicStore(&state.overflow, 2u);
    }
}
