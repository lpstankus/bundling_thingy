package main

import "core:os"
import "core:strings"
import "core:fmt"
import sdl "vendor:sdl3"

main :: proc()
{
    init_app()
    defer deinit_app()

    gds := [?]GraphDrawing{
        graphdrawing_from_trl("./data/fr-airlines.trl") or_else os.exit(1),
        graphdrawing_from_trl("./data/us-migrations.trl") or_else os.exit(1),
        graphdrawing_from_trl("./data/us-airlines.trl") or_else os.exit(1),
        graphdrawing_from_trl("./data/rj-notlocal-elective.trl") or_else os.exit(1),
        graphdrawing_from_trl("./data/rj-2019.trl") or_else os.exit(1),
    }

    meshes := make([][]GraphVertex, len(gds))

    fmt.println("Normalizing and meshing...")
    fmt.println()

    for &gd, i in gds
    {
        graphdrawing_normalize(&gd)
        graphdrawing_resample(&gd)
        meshes[i] = build_mesh(gd, 0.002)
    }

    fmt.println("Done reading, starting main loop...")
    fmt.println()

    i: int
    upload_graph_to_gpu(meshes[i])

    dirty := true
    outer: for
    {
        free_all(context.temp_allocator)

        begin_frame()
        defer end_frame()

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
                    upload_graph_to_gpu(meshes[i])
                    dirty = true
                case sdl.K_LEFT:
                    i = (i - 1) %% len(gds)
                    upload_graph_to_gpu(meshes[i])
                    dirty = true
                }
            }
        }

        if dirty
        {
            render_current_graph()
            dirty = false
        }
    }
}
