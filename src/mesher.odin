package main

import "core:fmt"
import "core:math"

GraphVertex :: distinct Vec2

build_mesh :: proc { graphdrawing_build_mesh }

graphdrawing_build_mesh :: proc(
    gd: GraphDrawing,
    line_width: f32 = 0.0025,
) -> []GraphVertex
{
    if len(gd.points) < 2 do return {}

    mesh := make([dynamic]GraphVertex, 0, (len(gd.points) - 1) * 2)

    half_width := line_width / 2

    it := EdgeIt{ gd = gd }
    for edge in next_edge(&it)
    {
        if len(edge) <= 1 do continue

        for i := 0; i < len(edge) - 1; i += 1
        {
            p0 := edge[i]
            p1 := edge[i + 1]

            dir := normalize(p1 - p0) or_continue
            nor := Vec2{ -dir.y, dir.x } * half_width

            v0 := GraphVertex(p0 + nor)
            v1 := GraphVertex(p0 - nor)
            v2 := GraphVertex(p1 + nor)
            v3 := GraphVertex(p1 - nor)

            append(&mesh, v0, v1, v2)
            append(&mesh, v3, v2, v1)
        }
    }

    return mesh[:]
}
