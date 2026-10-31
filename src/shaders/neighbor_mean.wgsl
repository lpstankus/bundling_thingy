override block_size: u32;
override accumulate_density: bool;

@group(0) @binding(0) var<storage, read> input_points: array<Point>;
@group(0) @binding(1) var<storage, read_write> output_points: array<Point>;
@group(0) @binding(2) var<uniform> radius: u32;
@group(0) @binding(3) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> paths: array<Path>;

@compute @workgroup_size(block_size)
fn neighbor_mean(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    let count = arrayLength(&input_points);
    if index >= count {
        return;
    }

    let current = input_points[index];
    let position = point_position(current);
    let path = paths[current.edge_id];
    let is_endpoint = index == path.point_offset || index == path.point_offset + path.count - 1u;
    var mean = position;
    if !is_endpoint {
        let texel_scale = vec2<f32>(map.size - vec2<u32>(1u)) * 0.5;
        let radius_squared = f32(radius) * f32(radius);
        var displacement = vec2<f32>(0.0);
        var neighbors = 0u;
        // Read positions from the start of this iteration, including endpoints and self.
        for (var neighbor = 0u; neighbor < count; neighbor += 1u) {
            let delta = point_position(input_points[neighbor]) - position;
            let texel_delta = delta * texel_scale;
            if dot(texel_delta, texel_delta) <= radius_squared {
                displacement += delta;
                neighbors += 1u;
            }
        }
        mean += displacement / f32(neighbors);
    }
    output_points[index] = Point(mean.x, mean.y, current.edge_id);
    if accumulate_density {
        let uv = vec2<f32>(mean.x * 0.5 + 0.5, 0.5 - mean.y * 0.5);
        let cell = vec2<u32>(round(uv * vec2<f32>(map.size - vec2<u32>(1u))));
        atomicAdd(&density[cell.y * map.size.x + cell.x], 1u);
    }
}
