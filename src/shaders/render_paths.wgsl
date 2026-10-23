struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Palette {
    colors: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var density: texture_2d<f32>;
@group(0) @binding(1) var density_sampler: sampler;
@group(0) @binding(2) var<uniform> palette: Palette;
@group(1) @binding(0) var<storage, read> points: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read> segments: array<u32>;

const PATH_OPACITY: f32 = 0.1;
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, -1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let index = segments[instance];
    let first = points[index];
    let second = points[index + 1u];
    let direction = normalize(second - first);
    let normal = vec2<f32>(-direction.y, direction.x) * 0.001;
    let corner = CORNERS[vertex];
    let position = mix(first, second, corner.x) + normal * corner.y;
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Grid coordinates span [0, size - 1]. Align them with texel centers.
    let size = vec2<f32>(textureDimensions(density));
    let uv = (in.uv * (size - 1.0) + 0.5) / size;
    let value = clamp(textureSample(density, density_sampler, uv).r, 0.0, 1.0);
    let stop = value * 15.0;
    let index = min(u32(stop), 14u);
    let color = mix(palette.colors[index].rgb, palette.colors[index + 1u].rgb, stop - f32(index));
    return vec4<f32>(color, PATH_OPACITY);
}
