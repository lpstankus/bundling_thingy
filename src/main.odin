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

    for &gd in gds 
    {
        graphdrawing_normalize(&gd)
        graphdrawing_resample(&gd)
    }

    i: int
    upload_to_gpu(gds[i])

    free_all(context.temp_allocator)

    outer: for
    {
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
                case sdl.K_RIGHT:
                    i = (i + 1) %% len(gds)
                    upload_to_gpu(gds[i])
                case sdl.K_LEFT:
                    i = (i - 1) %% len(gds)
                    upload_to_gpu(gds[i])
                }
            }
        }
        render_current_graph()
    }
}
