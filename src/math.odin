package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:strings"
import "core:strconv"
import mem "core:mem/virtual"

Vec2 :: distinct [2]f32

GraphDrawing :: struct
{
    name: string,
    min, max: Vec2,
    points: [dynamic]Vec2,
    starts: [dynamic]int,
}

EdgeIt :: struct
{
    cur: int,
    using gd: GraphDrawing,
}

cross :: proc { vec2_cross }
normalize :: proc { graphdrawing_normalize, vec2_normalize }
length_sq :: proc { vec2_length_sq }
length :: proc { vec2_length }

graphdrawing_from_trl :: proc(filename: string) -> (GraphDrawing, bool)
{
    data, ok := os.read_entire_file(filename, allocator=context.temp_allocator)
    if !ok
    {
        fmt.println("Failed to read file `", filename, "`: allocation error", sep="")
        fmt.println()
        return {}, false
    }

    gd := GraphDrawing{
        name = filename,
        min = Vec2{ max(f32), max(f32) },
        max = Vec2{ min(f32), min(f32) },
        points = make([dynamic]Vec2),
        starts = make([dynamic]int),
    }

    str := string(data)
    for line in strings.split_lines_iterator(&str)
    {
        line := string(line)

        edge_start := len(gd.points)

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
                fmt.println()
                return {}, false
            }

            x: f32
            x, ok = strconv.parse_f32(first)
            if !ok
            {
                fmt.println("Failed to read file: failed to parse float:", first)
                fmt.println()
                return {}, false
            }

            y: f32
            y, ok = strconv.parse_f32(secnd)
            if !ok
            {
                fmt.println("Failed to read file: failed to parse float:", secnd)
                fmt.println()
                return {}, false
            }

            point := Vec2{ x, y }

            gd.min.x = min(gd.min.x, point.x)
            gd.min.y = min(gd.min.y, point.y)

            gd.max.x = max(gd.max.x, point.x)
            gd.max.y = max(gd.max.y, point.y)

            append(&gd.points, point)
        }

        if edge_start != len(gd.points) do append(&gd.starts, edge_start)
    }

    fmt.println("Successfully read file:", filename)
    fmt.println(">> total edges:", len(gd.starts) - 1)
    fmt.println(">> total control points:", len(gd.points))
    fmt.println()

    return gd, true
}

graphdrawing_normalize :: proc(gd: ^GraphDrawing, border: f32 = 0.05)
{
    range := Vec2{ gd.max.x - gd.min.x, gd.max.y - gd.min.y }
    scale := (1 - border) / max(range.x, range.y)
    translation := Vec2{ (1 - scale * range.x) / 2, (1 - scale * range.y) / 2 }

    for &point in gd.points
    {
        point = (point - gd.min) * scale + translation
        point = (point - 0.5) * 2
    }

    gd.min, gd.max = Vec2{ -1, -1 }, Vec2{ 1, 1 }
}

graphdrawing_resample :: proc(gd: ^GraphDrawing, spacing: f32 = 0.02)
{
    new_points := make([dynamic]Vec2)
    new_starts := make([dynamic]int)

    it :=  EdgeIt{ gd = gd^ }
    for edge, i in next_edge(&it)
    {
        new_start := len(new_points)
        append(&new_starts, new_start)

        cur_space := spacing
        prv_pos := edge[0]

        append(&new_points, prv_pos)

        for i := 1; i < len(edge);
        {
            cur_pos := edge[i]
            new_space := length(prv_pos - cur_pos)

            if new_space < cur_space
            {
                cur_space -= new_space
                prv_pos = cur_pos
                i += 1
                continue
            }

            t := cur_space / new_space
            prv_pos = prv_pos * (1 - t) + cur_pos * t
            cur_space = spacing

            append(&new_points, prv_pos)
        }

        // If endpoint was not included during the resample pass
        if cur_space < spacing do append(&new_points, edge[len(edge) - 1])
    }
    append(&new_starts, len(new_points))

    fmt.println("Successfully resampled graph:", gd.name)
    fmt.println(">> edges: from", len(gd.starts) - 1, "to", len(new_starts) - 1)
    fmt.println(">> control points: from", len(gd.points), "to", len(new_points))
    fmt.println()

    delete(gd.points)
    delete(gd.starts)

    gd.points = new_points
    gd.starts = new_starts
}

next_edge :: proc(it: ^EdgeIt) -> ([]Vec2, int, bool)
{
    if it.cur < len(it.starts) - 1
    {
        it.cur += 1
        return it.points[it.starts[it.cur - 1] : it.starts[it.cur]], it.cur - 1, true
    }

    return {}, -1, false
}

vec2_normalize :: #force_inline proc(vec: Vec2) -> (Vec2, bool)
{
    len := length(vec)
    if len == 0 do return {}, false
    return vec / len, true
}

vec2_length_sq :: #force_inline proc(vec: Vec2) -> f32
{
    return vec.x * vec.x + vec.y * vec.y
}

vec2_length :: #force_inline proc(vec: Vec2) -> f32
{
    return math.sqrt(length_sq(vec))
}

vec2_cross :: #force_inline proc(l, r: Vec2) -> f32
{
    return l.x * r.y - l.y * r.x
}
