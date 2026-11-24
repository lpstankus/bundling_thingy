override block_size: u32;
override accumulate_density: bool;

struct Vertex {
    x: f32,
    y: f32,
    weight: u32,
    fixed: u32,
    origin: u32,
}

struct GroupParameters {
    attraction: u32,
    merge: f32,
    samples: u32,
    segments: u32,
    base: u32,
    cell_mode: u32,
    padding0: u32,
    padding1: u32,
}

@group(0) @binding(0) var<storage, read> original: array<Point>;
@group(0) @binding(1) var<storage, read> paths: array<Path>;
@group(0) @binding(2) var<storage, read_write> input: array<Vertex>;
@group(0) @binding(3) var<storage, read_write> output: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> sample_vertex: array<u32>;
@group(0) @binding(5) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> links: array<u32>;
@group(0) @binding(7) var<storage, read_write> partner: array<u32>;
@group(0) @binding(8) var<storage, read_write> remap: array<u32>;
@group(0) @binding(10) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> segments: array<u32>;
@group(0) @binding(12) var<storage, read_write> segment_path: array<u32>;
@group(0) @binding(13) var<storage, read_write> path_counts: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(15) var<uniform> params: GroupParameters;
@group(0) @binding(16) var<storage, read_write> resolved: array<Point>;
@group(0) @binding(17) var<storage, read_write> membership_heads: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> membership_next: array<u32>;

const NONE: u32 = 0xffffffffu;

fn index_of(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return params.base + id.y * groups.x * block_size + id.x;
}

fn vertex_position(vertex: Vertex) -> vec2<f32> {
    return vec2<f32>(vertex.x, vertex.y);
}

fn cell_of(position: vec2<f32>) -> vec2<i32> {
    let pixels = (position * 0.5 + 0.5) * vec2<f32>(map.size - vec2<u32>(1u));
    let width = select(f32(params.attraction), max(params.merge, 1.0), params.cell_mode == 1u);
    return vec2<i32>(floor(pixels / width));
}

fn hash_pair(pair: vec2<u32>) -> u32 {
    var hash = pair.x * 0x9e3779b9u ^ pair.y * 0x85ebca6bu;
    hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
    hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
    return (hash ^ (hash >> 16u)) % arrayLength(&heads);
}

fn segment_vertices(index: u32) -> vec2<u32> {
    let first = segments[index];
    let a = sample_vertex[first];
    let b = sample_vertex[first + 1u];
    return vec2<u32>(min(a, b), max(a, b));
}

@compute @workgroup_size(block_size)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index < params.samples {
        let point = original[index];
        let path = paths[point.edge_id];
        let fixed = index == path.point_offset || index == path.point_offset + path.count - 1u;
        input[index] = Vertex(point.x, point.y, 1u, select(0u, 1u, fixed), index);
        sample_vertex[index] = index;
    }
    if index < params.segments {
        segment_path[index] = index;
        atomicStore(&path_counts[index], 1u);
        atomicStore(&membership_heads[index], index + 1u);
        membership_next[index] = 0u;
    }
}

@compute @workgroup_size(block_size)
fn index_vertices(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= atomicLoad(&state[0]) {
        return;
    }
    let cell = cell_of(vertex_position(input[index]));
    let bucket = hash_pair(vec2<u32>(cell));
    links[index] = atomicExchange(&heads[bucket], index + 1u);
}

@compute @workgroup_size(block_size)
fn average(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= atomicLoad(&state[0]) {
        return;
    }
    var vertex = input[index];
    if vertex.fixed == 0u {
        let position = vertex_position(vertex);
        let center = cell_of(position);
        let scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
        let radius_squared = f32(params.attraction) * f32(params.attraction);
        var displacement = vec2<f32>(0.0);
        var weight = 0.0;
        for (var y = -1; y <= 1; y += 1) {
            for (var x = -1; x <= 1; x += 1) {
                let cell = center + vec2<i32>(x, y);
                if any(cell < vec2<i32>(0)) {
                    continue;
                }
                var node = atomicLoad(&heads[hash_pair(vec2<u32>(cell))]);
                while node != 0u {
                    let other = node - 1u;
                    node = links[other];
                    let neighbor = input[other];
                    let neighbor_position = vertex_position(neighbor);
                    if any(cell_of(neighbor_position) != cell) {
                        continue;
                    }
                    let delta = neighbor_position - position;
                    let texels = delta * scale;
                    if dot(texels, texels) <= radius_squared {
                        displacement += delta * f32(neighbor.weight);
                        weight += f32(neighbor.weight);
                    }
                }
            }
        }
        let mean = position + displacement / weight;
        vertex.x = mean.x;
        vertex.y = mean.y;
    }
    output[index] = vertex;
}

@compute @workgroup_size(block_size)
fn choose_partner(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= atomicLoad(&state[0]) {
        return;
    }
    let vertex = input[index];
    var best = NONE;
    var best_origin = NONE;
    var best_distance = params.merge * params.merge;
    if vertex.fixed == 0u && params.merge > 0.0 {
        let position = vertex_position(vertex);
        let center = cell_of(position);
        let scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
        for (var y = -1; y <= 1; y += 1) {
            for (var x = -1; x <= 1; x += 1) {
                let cell = center + vec2<i32>(x, y);
                if any(cell < vec2<i32>(0)) {
                    continue;
                }
                var node = atomicLoad(&heads[hash_pair(vec2<u32>(cell))]);
                while node != 0u {
                    let other = node - 1u;
                    node = links[other];
                    let candidate = input[other];
                    let candidate_position = vertex_position(candidate);
                    if other == index || candidate.fixed != 0u || any(cell_of(candidate_position) != cell) {
                        continue;
                    }
                    let delta = (candidate_position - position) * scale;
                    let distance = dot(delta, delta);
                    if distance < best_distance || (distance == best_distance && candidate.origin < best_origin) {
                        best = other;
                        best_origin = candidate.origin;
                        best_distance = distance;
                    }
                }
            }
        }
    }
    partner[index] = best;
}

@compute @workgroup_size(block_size)
fn compact_vertices(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= atomicLoad(&state[0]) {
        return;
    }
    var vertex = input[index];
    let other = partner[index];
    var paired = false;
    if other != NONE && partner[other] == index {
        let candidate = input[other];
        if candidate.origin < vertex.origin {
            return;
        }
        paired = true;
        let weight = vertex.weight + candidate.weight;
        let position = vertex_position(vertex)
            + (vertex_position(candidate) - vertex_position(vertex)) * (f32(candidate.weight) / f32(weight));
        vertex.x = position.x;
        vertex.y = position.y;
        vertex.weight = weight;
    }
    let next = atomicAdd(&state[1], 1u);
    output[next] = vertex;
    remap[index] = next;
    if paired {
        remap[other] = next;
    }
    if accumulate_density {
        let position = vertex_position(vertex);
        let uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
        let cell = vec2<u32>(round(uv * vec2<f32>(map.size - vec2<u32>(1u))));
        atomicAdd(&density[cell.y * map.size.x + cell.x], vertex.weight);
    }
}

@compute @workgroup_size(block_size)
fn resolve_samples(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= params.samples {
        return;
    }
    let vertex = remap[sample_vertex[index]];
    sample_vertex[index] = vertex;
    let point = input[vertex];
    resolved[index] = Point(point.x, point.y, original[index].edge_id);
}

@compute @workgroup_size(block_size)
fn index_segments(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= params.segments {
        return;
    }
    let endpoints = segment_vertices(index);
    if endpoints.x == endpoints.y {
        segment_path[index] = NONE;
        return;
    }
    let bucket = hash_pair(endpoints);
    links[index] = atomicExchange(&heads[bucket], index + 1u);
}

@compute @workgroup_size(block_size)
fn count_paths(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= params.segments {
        return;
    }
    let endpoints = segment_vertices(index);
    if endpoints.x == endpoints.y {
        return;
    }
    let edge = original[segments[index]].edge_id;
    var path = index;
    var edge_occurrence = index;
    var node = atomicLoad(&heads[hash_pair(endpoints)]);
    while node != 0u {
        let other = node - 1u;
        node = links[other];
        if all(segment_vertices(other) == endpoints) {
            path = min(path, other);
            if original[segments[other]].edge_id == edge {
                edge_occurrence = min(edge_occurrence, other);
            }
        }
    }
    segment_path[index] = path;
    membership_next[index] = select(NONE, 0u, edge_occurrence == index);
}

@compute @workgroup_size(block_size)
fn build_memberships(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= params.segments || segment_path[index] == NONE || membership_next[index] == NONE {
        return;
    }
    let path = segment_path[index];
    // Each node is an original segment occurrence. Its edge ID remains in the original samples.
    membership_next[index] = atomicExchange(&membership_heads[path], index + 1u);
    atomicAdd(&path_counts[path], 1u);
}

@compute @workgroup_size(block_size)
fn summarize_paths(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = index_of(id, groups);
    if index >= params.segments || segment_path[index] != index {
        return;
    }
    let edges = atomicLoad(&path_counts[index]);
    atomicAdd(&state[2], 1u);
    if edges > 1u {
        atomicAdd(&state[3], 1u);
    }
    atomicMax(&state[4], edges);
}

@compute @workgroup_size(1)
fn finish_iteration() {
    atomicStore(&state[0], atomicLoad(&state[1]));
}
