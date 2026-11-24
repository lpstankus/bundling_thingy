override block_size: u32;
override accumulate_density: bool;
override weight_by_endpoints: bool;

@group(0) @binding(0) var<storage, read> input_points: array<Point>;
@group(0) @binding(1) var<storage, read_write> output_points: array<Point>;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> paths: array<Path>;
@group(0) @binding(10) var<uniform> query_offset: vec4<u32>;

struct Summary {
    start: u32,
    end: u32,
    relative_sum: vec2<f32>,
    minimum: vec2<f32>,
    maximum: vec2<f32>,
    edge_id: u32,
}

struct Neighbor {
    point: Point,
    endpoints: Edge,
}

var<workgroup> cell_summary: Summary;
var<workgroup> scan_needed: atomic<u32>;
var<workgroup> scan_flag: u32;
var<workgroup> tile: array<Neighbor, block_size>;

fn edge_endpoints(edge_id: u32) -> Edge {
    if !weight_by_endpoints {
        return Edge(vec2<f32>(0.0), vec2<f32>(0.0));
    }
    // Endpoints are pinned, including after swapping the point buffers.
    let path = paths[edge_id];
    return Edge(
        point_position(input_points[path.point_offset]),
        point_position(input_points[path.point_offset + path.count - 1u]),
    );
}

fn read_neighbor(slot: u32) -> Neighbor {
    let point = input_points[positions[slot].index];
    return Neighbor(point, edge_endpoints(point.edge_id));
}

fn endpoint_similarity(a: Edge, b: Edge, texel_scale: vec2<f32>) -> f32 {
    if !weight_by_endpoints {
        return 1.0;
    }
    let ss = (a.source - b.source) * texel_scale;
    let sd = (a.source - b.destination) * texel_scale;
    let ds = (a.destination - b.source) * texel_scale;
    let dd = (a.destination - b.destination) * texel_scale;
    let closest_squared = min(min(dot(ss, ss), dot(sd, sd)), min(dot(ds, ds), dot(dd, dd)));
    // Inverse distance, with a one-texel offset to keep shared endpoints finite.
    return 1.0 / (1.0 + sqrt(closest_squared));
}

fn moved_position(position: vec2<f32>, displacement: vec2<f32>, total_weight: f32) -> vec2<f32> {
    if total_weight == 0.0 {
        return position;
    }
    // Relative displacements give the weighted mean without summing absolute coordinates.
    return position + displacement / total_weight;
}

fn read_summary(cell: u32) -> Summary {
    return Summary(
        spatial.cells[cell].start,
        atomicLoad(&spatial.cells[cell].end),
        summaries[cell].relative_sum,
        summaries[cell].minimum,
        summaries[cell].maximum,
        summaries[cell].edge_id,
    );
}

fn load_summary(cell: u32, lane: u32) -> Summary {
    if lane == 0u {
        cell_summary = read_summary(cell);
    }
    return workgroupUniformLoad(&cell_summary);
}

// Zero means outside, one means fully contained, and two needs individual distance checks.
fn classify_cell(source: Summary, position: vec2<f32>, texel_scale: vec2<f32>) -> u32 {
    if source.start == source.end {
        return 0u;
    }
    let padding = vec2<f32>(map.size) * 0.00000024;
    let radius_squared = f32(grid.radius) * f32(grid.radius);
    let nearest = max(max(source.minimum - position, position - source.maximum), vec2<f32>(0.0));
    let near_texels = max(nearest * texel_scale - padding, vec2<f32>(0.0));
    if dot(near_texels, near_texels) > radius_squared {
        return 0u;
    }
    let farthest = max(abs(source.minimum - position), abs(source.maximum - position));
    let far_texels = farthest * texel_scale + padding;
    return select(2u, 1u, dot(far_texels, far_texels) < radius_squared);
}

fn cell_displacement(source: Summary, position: vec2<f32>) -> vec2<f32> {
    // Relative sums avoid subtracting two large absolute position sums.
    return source.relative_sum + (indexed_position(source.start) - position) * f32(source.end - source.start);
}

fn movable_point(index: u32, current: Point) -> bool {
    let path = paths[current.edge_id];
    return index != path.point_offset && index != path.point_offset + path.count - 1u;
}

fn write_point(index: u32, mean: vec2<f32>, edge_id: u32) {
    output_points[index] = Point(mean.x, mean.y, edge_id);
    if accumulate_density {
        let uv = vec2<f32>(mean.x * 0.5 + 0.5, 0.5 - mean.y * 0.5);
        let cell = vec2<u32>(round(uv * vec2<f32>(map.size - vec2<u32>(1u))));
        atomicAdd(&density[cell.y * map.size.x + cell.x], 1u);
    }
}

@compute @workgroup_size(block_size)
fn neighbor_mean(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let task = load_task(query_offset.x + group.y * groups.x + group.x, lane);
    if task.cell >= arrayLength(&spatial.cells) {
        return;
    }
    let query_cell = load_summary(task.cell, lane);
    let slot = task.start + lane;
    let valid = slot < query_cell.end;
    var index = 0u;
    var current = Point(0.0, 0.0, 0u);
    var endpoints = Edge(vec2<f32>(0.0), vec2<f32>(0.0));
    var movable = false;
    if valid {
        index = positions[slot].index;
        current = input_points[index];
        endpoints = edge_endpoints(current.edge_id);
        movable = movable_point(index, current);
    }
    let position = point_position(current);
    let texel_scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
    let radius_squared = f32(grid.radius) * f32(grid.radius);
    // Conservative bounds leave exact-radius decisions to the individual distance check.
    let padding = vec2<f32>(map.size) * 0.00000024;
    let reach = vec2<f32>(f32(grid.radius)) + padding;
    let first = grid_cell(texel_position(query_cell.minimum) - reach);
    let last = grid_cell(texel_position(query_cell.maximum) + reach);
    var displacement = vec2<f32>(0.0);
    var total_weight = 0.0;
    for (var y = first.y; y <= last.y; y += 1u) {
        for (var x = first.x; x <= last.x; x += 1u) {
            let source = load_summary(y * grid.size.x + x, lane);
            if source.start == source.end {
                continue;
            }
            let classification = classify_cell(source, position, texel_scale);
            let foreign_edge = source.edge_id != current.edge_id;
            let summarize = classification == 1u && source.edge_id != MIXED_EDGES;
            let scan = movable && foreign_edge && classification != 0u && !summarize;
            if movable && foreign_edge && summarize {
                let count = source.end - source.start;
                let weight = endpoint_similarity(endpoints, edge_endpoints(source.edge_id), texel_scale);
                displacement += weight * cell_displacement(source, position);
                total_weight += weight * f32(count);
            }

            if lane == 0u {
                atomicStore(&scan_needed, 0u);
            }
            workgroupBarrier();
            if scan {
                atomicStore(&scan_needed, 1u);
            }
            workgroupBarrier();
            if lane == 0u {
                scan_flag = atomicLoad(&scan_needed);
            }
            if workgroupUniformLoad(&scan_flag) != 0u {
                for (var start = source.start; start < source.end; start += block_size) {
                    let count = min(block_size, source.end - start);
                    if lane < count {
                        tile[lane] = read_neighbor(start + lane);
                    }
                    workgroupBarrier();
                    if scan {
                        for (var neighbor = 0u; neighbor < count; neighbor += 1u) {
                            let candidate = tile[neighbor];
                            if candidate.point.edge_id == current.edge_id {
                                continue;
                            }
                            let delta = point_position(candidate.point) - position;
                            let texel_delta = delta * texel_scale;
                            if dot(texel_delta, texel_delta) <= radius_squared {
                                let weight = endpoint_similarity(endpoints, candidate.endpoints, texel_scale);
                                displacement += weight * delta;
                                total_weight += weight;
                            }
                        }
                    }
                    workgroupBarrier();
                }
            }
        }
    }
    // Inactive lanes and fixed endpoints still participate in every workgroup barrier.
    if valid {
        var mean = position;
        if movable {
            mean = moved_position(position, displacement, total_weight);
        }
        write_point(index, mean, current.edge_id);
    }
}

@compute @workgroup_size(block_size)
fn neighbor_mean_sparse(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index >= arrayLength(&input_points) {
        return;
    }
    let current = input_points[index];
    let position = point_position(current);
    if !movable_point(index, current) {
        write_point(index, position, current.edge_id);
        return;
    }
    let endpoints = edge_endpoints(current.edge_id);
    let texel_scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
    let radius_squared = f32(grid.radius) * f32(grid.radius);
    let padding = vec2<f32>(map.size) * 0.00000024;
    let reach = vec2<f32>(f32(grid.radius)) + padding;
    let texels = texel_position(position);
    let first = grid_cell(texels - reach);
    let last = grid_cell(texels + reach);
    var displacement = vec2<f32>(0.0);
    var total_weight = 0.0;
    for (var y = first.y; y <= last.y; y += 1u) {
        for (var x = first.x; x <= last.x; x += 1u) {
            // Small cells are cheaper to scan directly, without building summaries.
            let cell = y * grid.size.x + x;
            let end = atomicLoad(&spatial.cells[cell].end);
            for (var neighbor = spatial.cells[cell].start; neighbor < end; neighbor += 1u) {
                let candidate = input_points[positions[neighbor].index];
                if candidate.edge_id == current.edge_id {
                    continue;
                }
                let delta = point_position(candidate) - position;
                let texel_delta = delta * texel_scale;
                if dot(texel_delta, texel_delta) <= radius_squared {
                    let weight = endpoint_similarity(endpoints, edge_endpoints(candidate.edge_id), texel_scale);
                    displacement += weight * delta;
                    total_weight += weight;
                }
            }
        }
    }
    write_point(index, moved_position(position, displacement, total_weight), current.edge_id);
}
