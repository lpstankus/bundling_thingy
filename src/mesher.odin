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
    mesh := make([dynamic]GraphVertex, 0, gd.total_points * 3, context.temp_allocator)

    half_width := line_width / 2

    for edge, edge_num in gd.edges
    {
        if len(edge) <= 1 do continue

        prev_point := edge[0]
        prev_upper, prev_lower, ok := calc_bounds(edge[0], edge[1] - edge[0], half_width)
        if !ok
        {
            fmt.println("Degenerate segment at beggining of edge", edge_num, "skipping edge...")
            continue
        }

        for i := 1; i < len(edge) - 1; i += 1
        {
            prev_dir := edge[i] - prev_point
            next_dir := edge[i + 1] - edge[i]

            next_upper, next_lower := calc_bounds(edge[i], next_dir, half_width) or_continue

            curr_upper := calc_intersection(prev_upper, prev_dir, next_upper, next_dir) or_continue
            curr_lower := calc_intersection(prev_lower, prev_dir, next_lower, next_dir) or_continue

            append(&mesh, GraphVertex(prev_upper))
            append(&mesh, GraphVertex(prev_lower))
            append(&mesh, GraphVertex(curr_upper))

            append(&mesh, GraphVertex(curr_lower))
            append(&mesh, GraphVertex(curr_upper))
            append(&mesh, GraphVertex(prev_lower))

            prev_point = edge[i]
            prev_upper = curr_upper
            prev_lower = curr_lower
        }
        {
            dir := edge[len(edge) - 1] - prev_point
            curr_upper, curr_lower, ok := calc_bounds(edge[len(edge) - 1], dir, half_width)
            if ok
            {
                append(&mesh, GraphVertex(prev_upper))
                append(&mesh, GraphVertex(prev_lower))
                append(&mesh, GraphVertex(curr_upper))

                append(&mesh, GraphVertex(curr_lower))
                append(&mesh, GraphVertex(curr_upper))
                append(&mesh, GraphVertex(prev_lower))
            }
            else
            {
                fmt.println("Failed to generate last line segment for edge", edge_num)
            }
        }
    }

    return mesh[:]
}

@(private="file")
calc_bounds :: #force_inline proc(anchor, dir: Vec2, half_width: f32) -> (Vec2, Vec2, bool)
{
    dir, ok := normalize(dir)
    if !ok do return {}, {}, false

    nor := Vec2{ -dir.y, dir.x }
    return anchor + nor * half_width, anchor - nor * half_width, true
}

@(private="file")
calc_intersection :: #force_inline proc(p0, dir0, p1, dir1: Vec2) -> (Vec2, bool)
{
    dir_cross := cross(dir0, dir1)
    if math.abs(dir_cross) < 1e-6 do return {}, false

    dir := p1 - p0
    t := cross(dir, dir1) / dir_cross

    return p0 + t * dir0, true
}
