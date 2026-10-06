package main

import "core:strings"
import sdl "vendor:sdl3"

png_write :: proc(path: string, size: u32, pixels: rawptr)
{
    surface := sdl.CreateSurfaceFrom(i32(size), i32(size), .RGBA32, pixels, i32(size * 4))
    if surface == nil do fail("Failed to create PNG surface:", sdl.GetError())
    defer sdl.DestroySurface(surface)

    filename := strings.clone_to_cstring(path, context.temp_allocator)
    if !sdl.SavePNG(surface, filename)
    {
        fail("Failed to write PNG:", path, sdl.GetError())
    }
}
