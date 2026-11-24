override map_width: u32;
override map_height: u32;
override step_size: f32;

@group(0) @binding(0) var<storage, read_write> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> segments: array<u32>;
@group(0) @binding(2) var<storage, read> density: array<f32>;

fn density_at(position: vec2<f32>) -> f32 {
    let last = vec2<u32>(map_width - 1u, map_height - 1u);
    let cell = clamp(position, vec2<f32>(0.0), vec2<f32>(last));
    let first = vec2<u32>(floor(cell));
    let second = min(first + vec2<u32>(1u), last);
    let fraction = fract(cell);
    let top = mix(
        density[first.y * map_width + first.x],
        density[first.y * map_width + second.x],
        fraction.x,
    );
    let bottom = mix(
        density[second.y * map_width + first.x],
        density[second.y * map_width + second.x],
        fraction.x,
    );
    return mix(top, bottom, fraction.y);
}

@compute @workgroup_size(256)
fn advect(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * 256u + id.x;
    if index + 1u >= arrayLength(&segments) {
        return;
    }

    // Consecutive segments share an interior point. Gaps mark fixed endpoints.
    let point_index = segments[index] + 1u;
    if segments[index + 1u] != point_index {
        return;
    }

    let point = points[point_index];
    let last = vec2<f32>(f32(map_width - 1u), f32(map_height - 1u));
    let position = vec2<f32>(point.x * 0.5 + 0.5, 0.5 - point.y * 0.5) * last;
    let gradient = 0.5 * vec2<f32>(
        density_at(position + vec2<f32>(1.0, 0.0)) - density_at(position - vec2<f32>(1.0, 0.0)),
        density_at(position + vec2<f32>(0.0, 1.0)) - density_at(position - vec2<f32>(0.0, 1.0)),
    );
    let movement = step_size * gradient / max(length(gradient), 1e-6);
    let next = clamp(position + movement, vec2<f32>(0.0), last);
    points[point_index] = point + (next - position) * vec2<f32>(2.0, -2.0) / max(last, vec2<f32>(1.0));
}
