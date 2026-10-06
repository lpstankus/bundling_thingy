package main

import "core:fmt"
import "core:os"
import "core:strings"
import sdl "vendor:sdl3"

AppOptions :: struct
{
    headless:   bool,
    output_dir: string,
}

main :: proc()
{
    opts := parse_args(os.args[1:])

    init_render(opts)
    defer deinit_render()

    if opts.headless
    {
        gd := graphdrawing_from_trl("data/fr-airlines.trl") or_else fail()
        graphdrawing_normalize(&gd)
        original_mesh := build_mesh(gd, 0.002)

        _ = os.make_directory_all(opts.output_dir)
        fmt.println()

        upload_graph_to_gpu(original_mesh)
        render_graph_to_image(opts.output_dir, "original")

        return
    }

    gds := [?]GraphDrawing{
        graphdrawing_from_trl("./data/fr-airlines.trl") or_else fail(),
        graphdrawing_from_trl("./data/us-migrations.trl") or_else fail(),
        graphdrawing_from_trl("./data/us-airlines.trl") or_else fail(),
        graphdrawing_from_trl("./data/rj-notlocal-elective.trl") or_else fail(),
        graphdrawing_from_trl("./data/rj-2019.trl") or_else fail(),
    }

    fmt.println("Normalizing and resampling...")
    fmt.println()

    for &gd in gds
    {
        graphdrawing_normalize(&gd)
        graphdrawing_resample(&gd)
    }

    fmt.println("Done reading, starting main loop...")
    fmt.println()

    i: int

    dirty := true
    outer: for
    {
        free_all(context.temp_allocator)

        begin_frame()

        evt : sdl.Event
        for sdl.PollEvent(&evt)
        {
            #partial switch evt.type
            {
            case .QUIT:
                break outer
            case .KEY_DOWN:
                switch evt.key.key
                {
                case sdl.K_ESCAPE:
                    break outer
                case sdl.K_RIGHT, sdl.K_SPACE:
                    i = (i + 1) %% len(gds)
                    dirty = true
                case sdl.K_LEFT:
                    i = (i - 1) %% len(gds)
                    dirty = true
                }
            }
        }

        if dirty
        {
            mesh := build_mesh(gds[i], 0.002)
            upload_graph_to_gpu(mesh)
            delete(mesh)

            render_graph_to_surface()
            dirty = false
        }
    }
}

parse_args :: proc(args: []string) -> AppOptions
{
    opts := AppOptions{ output_dir = "out" }

    for i := 0; i < len(args); i += 1
    {
        arg := args[i]

        switch arg
        {
        case "--headless":
            opts.headless = true
        case "--out":
            i += 1
            if i >= len(args) || args[i] == "" || strings.has_prefix(args[i], "--")
            {
                fail("Expected a directory after --out")
            }
            opts.output_dir = args[i]
        case:
            fmt.println("Unknown option:", arg)
            fail("Usage: src.bin [--headless] [--out <dir>]")
        }
    }

    return opts
}
