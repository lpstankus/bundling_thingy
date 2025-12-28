#version 450

layout(location = 0) in vec2 in_pos;

layout(set = 1, binding = 0) uniform ColorUniform {
    vec4 u_color;
};

layout(location = 0) out vec4  out_color;

void main() {
    float scale = 1.0;
    vec2 offset = vec2(0, 0);

    gl_Position = vec4((in_pos * scale) + offset, 0.0, 1.0);
    out_color = u_color;
}
