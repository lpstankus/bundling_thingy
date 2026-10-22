struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var density: texture_2d<f32>;
@group(0) @binding(1) var density_sampler: sampler;

const PATH_OPACITY: f32 = 0.1;

// Sixteen evenly spaced samples of Matplotlib's cividis colormap.
// Nuñez, Anderton, and Renslow, https://doi.org/10.1371/journal.pone.0199239
const CIVIDIS = array<vec3<f32>, 16>(
    vec3<f32>(0.000000, 0.135112, 0.304751),
    vec3<f32>(0.000000, 0.181610, 0.421859),
    vec3<f32>(0.117612, 0.225935, 0.434308),
    vec3<f32>(0.208926, 0.272546, 0.424809),
    vec3<f32>(0.279411, 0.318677, 0.423031),
    vec3<f32>(0.342246, 0.364939, 0.428559),
    vec3<f32>(0.401418, 0.411790, 0.440708),
    vec3<f32>(0.458366, 0.459552, 0.460457),
    vec3<f32>(0.517920, 0.508454, 0.472707),
    vec3<f32>(0.582087, 0.558670, 0.468118),
    vec3<f32>(0.648222, 0.610553, 0.454801),
    vec3<f32>(0.716177, 0.664384, 0.432386),
    vec3<f32>(0.785965, 0.720438, 0.399613),
    vec3<f32>(0.857809, 0.778969, 0.353259),
    vec3<f32>(0.932180, 0.840159, 0.285880),
    vec3<f32>(0.995737, 0.909344, 0.217772),
);

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
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
    return vec4<f32>(mix(CIVIDIS[index], CIVIDIS[index + 1u], stop - f32(index)), PATH_OPACITY);
}
