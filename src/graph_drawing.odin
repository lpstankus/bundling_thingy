package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:strings"
import "core:strconv"
import mem "core:mem/virtual"

Point2D :: distinct [2]f32
Edge :: [dynamic]Point2D

GraphDrawing :: struct
{
    min: Point2D,
    max: Point2D,
    total_points: int,
    edges: [dynamic]Edge,
    endpoints: [dynamic][2]Point2D,
}

graphdrawing_from_trl :: proc(filename: string) -> (gd: GraphDrawing, ok: bool = true)
{
    data: []byte
    data, ok = os.read_entire_file(filename, allocator=context.temp_allocator)
    if !ok
    {
        fmt.println("Failed to read file `", filename, "`: allocation error", sep="")
        return {}, false
    }

    gd.edges = make([dynamic]Edge)
    gd.min = Point2D{ max(f32), max(f32) }
    gd.max = Point2D{ min(f32), min(f32) }

    str := string(data)
    for line in strings.split_lines_iterator(&str)
    {
        line := string(line)

        edge := make(Edge)

        _, _ = strings.split_iterator(&line, " ") // skip idx
        for token in strings.split_iterator(&line, " ")
        {
            if len(token) == 0 do continue
            if token[0] == '(' || token[0] == '<' do continue

            first := token
            secnd, ok := strings.split_iterator(&line, " ")
            if !ok
            {
                fmt.println("Failed to read file: unexpected EOL")
                return {}, false
            }

            x: f32
            x, ok = strconv.parse_f32(first)
            if !ok
            {
                fmt.println("Failed to read file: failed to parse float:", first)
                return {}, false
            }

            y: f32
            y, ok = strconv.parse_f32(secnd)
            if !ok
            {
                fmt.println("Failed to read file: failed to parse float:", secnd)
                return {}, false
            }

            point := Point2D{ x, y }

            gd.min.x = min(gd.min.x, point.x)
            gd.min.y = min(gd.min.y, point.y)

            gd.max.x = max(gd.max.x, point.x)
            gd.max.y = max(gd.max.y, point.y)

            append(&edge, point)
            gd.total_points += 1
        }

        if len(edge) == 0
        {
            delete(edge)
            continue
        }

        append(&gd.edges, edge)
        append(&gd.endpoints, [?]Point2D{ edge[0], edge[len(edge) - 1] })
    }

    fmt.println("Successfully read file:", filename)
    fmt.println(">> total edges:", len(gd.edges))
    fmt.println(">> total control points:", gd.total_points)

    return
}

graphdrawing_normalize :: proc(gd: ^GraphDrawing, border: f32 = 0.05)
{
    range := Point2D{ gd.max.x - gd.min.x, gd.max.y - gd.min.y }
    scale := (1 - border) / max(range.x, range.y)
    translation := Point2D{ (1 - scale * range.x) / 2, (1 - scale * range.y) / 2 }

    for edge in gd.edges
    {
        for &point in edge
        {
            point = (point - gd.min) * scale + translation
            point = (point - 0.5) * 2
        }
    }

    for &points in gd.endpoints
    {
        points[0] = (points[0] - gd.min) * scale + translation
        points[0] = (points[0] - 0.5) * 2

        points[1] = (points[1] - gd.min) * scale + translation
        points[1] = (points[1] - 0.5) * 2
    }

    gd.min, gd.max = Point2D{ -1, -1 }, Point2D{ 1, 1 }
}

graphdrawing_resample :: proc(gd: ^GraphDrawing, spacing: f32 = 0.02)
{
    new_total_points: int
    new_edges := make([dynamic]Edge, 0, len(gd.edges))
    for edge, i in gd.edges
    {
        new_edge := make(Edge, 0, len(edge))
        defer append(&new_edges, new_edge)

        cur_space := spacing
        prv_pos := edge[0]

        append(&new_edge, prv_pos)
        new_total_points += 1

        for i := 1; i < len(edge);
        {
            cur_pos := edge[i]
            new_space := distance(cur_pos, prv_pos)

            if new_space < cur_space
            {
                cur_space -= new_space
                prv_pos = cur_pos
                i += 1
                continue
            }

            t := cur_space / new_space
            prv_pos = prv_pos * (1 - t) + cur_pos * t
            append(&new_edge, prv_pos)
            new_total_points += 1

            cur_space = spacing
        }

        // If endpoint was not included during the resample pass
        if cur_space < spacing
        {
            append(&new_edge, edge[len(edge) - 1])
            new_total_points += 1
        }

        delete(edge)
    }

    delete(gd.edges)
    gd.edges = new_edges
    gd.total_points = new_total_points
}

point2d_distance :: #force_inline proc(l, r: Point2D) -> f32
{
    return math.sqrt((l.x - r.x) * (l.x - r.x) + (l.y - r.y) * (l.y - r.y))
}

distance :: proc { point2d_distance }
