package main

import "core:os"
import "core:fmt"
import "core:strings"
import sdl "vendor:sdl3"

RENDER_SIZE :: 1600

RenderContext :: struct
{
    window:   ^sdl.Window,
    renderer: ^sdl.Renderer,
    device:   ^sdl.GPUDevice,

    cmd_buffer: ^sdl.GPUCommandBuffer,
    target:     ^sdl.GPUTexture,
    color:      [4]f32,

    last_frame : sdl.Time,
    last_title_change: sdl.Time,

    graph: GraphRenderCtx,
}

GraphRenderCtx :: struct
{
    vert_shader: ^sdl.GPUShader,
    frag_shader: ^sdl.GPUShader,

    vertex_count: u32,
    vertex_bytes: u32,
    vertex_buffer: ^sdl.GPUBuffer,

    pipeline: ^sdl.GPUGraphicsPipeline,
}

render_ctx : RenderContext

init_render :: proc(opts: AppOptions)
{
    windowed := !opts.headless

    if windowed && !sdl.SetHint("SDL_VIDEO_WAYLAND_SCALE_TO_DISPLAY", "1")
    {
        fmt.println("Failed to set SDL hint:", sdl.GetError())
    }

    video_ok := sdl.Init(sdl.INIT_VIDEO)

    // Retry with the offscreen driver for headless exports without a display server.
    if !video_ok && !windowed
    {
        video_ok = sdl.SetHint("SDL_VIDEO_DRIVER", "offscreen") && sdl.Init(sdl.INIT_VIDEO)
    }

    if !video_ok
    {
        fail("Failed to init SDL:", sdl.GetError())
    }

    if windowed
    {
        ok := sdl.CreateWindowAndRenderer(
            "Bundling thingy",
            RENDER_SIZE, RENDER_SIZE,
            nil,
            &render_ctx.window,
            &render_ctx.renderer,
        )
        if !ok do fail("Failed to create window:", sdl.GetError())
    }

    render_ctx.device = sdl.CreateGPUDevice({ .SPIRV }, true, nil)
    if render_ctx.device == nil
    {
        fail("Failed to create GPU device:", sdl.GetError())
    }

    if render_ctx.window != nil
    {
        ok := sdl.ClaimWindowForGPUDevice(render_ctx.device, render_ctx.window)
        if !ok do fail("Failed to claim window for GPU device:", sdl.GetError())
    }

    color_format: sdl.GPUTextureFormat = .R8G8B8A8_UNORM
    if windowed
    {
        color_format = sdl.GetGPUSwapchainTextureFormat(render_ctx.device, render_ctx.window)
    }

    if !init_graph_render_ctx(color_format)
    {
        fail("Failed to create graph render context:", sdl.GetError())
    }

    render_ctx.color = { 1, 0, 0, 0.05 }

    _ = sdl.GetCurrentTime(&render_ctx.last_frame)
    render_ctx.last_title_change = render_ctx.last_frame
}

deinit_render :: proc()
{
    if render_ctx.graph.vertex_buffer != nil do sdl.ReleaseGPUBuffer(render_ctx.device, render_ctx.graph.vertex_buffer)
    if render_ctx.graph.pipeline != nil do sdl.ReleaseGPUGraphicsPipeline(render_ctx.device, render_ctx.graph.pipeline)

    if render_ctx.graph.frag_shader != nil do sdl.ReleaseGPUShader(render_ctx.device, render_ctx.graph.frag_shader)
    if render_ctx.graph.vert_shader != nil do sdl.ReleaseGPUShader(render_ctx.device, render_ctx.graph.vert_shader)

    if render_ctx.device != nil do sdl.DestroyGPUDevice(render_ctx.device)
    if render_ctx.window != nil do sdl.DestroyWindow(render_ctx.window)

    sdl.Quit()

    render_ctx = {}
}

begin_frame :: proc()
{
    prev_time := render_ctx.last_frame
    curr_time : sdl.Time
    ok := sdl.GetCurrentTime(&curr_time)
    if !ok do return

    render_ctx.last_frame = curr_time

    if curr_time - render_ctx.last_title_change >= 1_000_000_000
    {
        title := strings.clone_to_cstring(
            fmt.tprintf("Bundling Thingy (%.2fms)", f32(curr_time - prev_time) / 1_000_000_000),
            allocator=context.temp_allocator,
        )
        sdl.SetWindowTitle(render_ctx.window, title)
        render_ctx.last_title_change = curr_time
    }
}

upload_graph_to_gpu :: proc(mesh: []GraphVertex)
{
    render_ctx.graph.vertex_count = u32(len(mesh))
    if render_ctx.graph.vertex_count == 0 do return

    bytes := render_ctx.graph.vertex_count * size_of(GraphVertex)

    if render_ctx.graph.vertex_bytes < bytes
    {
        fmt.println("Resizing graph vertex buffer, from", render_ctx.graph.vertex_bytes, "to", bytes)

        render_ctx.graph.vertex_bytes = bytes

        sdl.ReleaseGPUBuffer(render_ctx.device, render_ctx.graph.vertex_buffer)
        render_ctx.graph.vertex_buffer = sdl.CreateGPUBuffer(
            render_ctx.device,
            sdl.GPUBufferCreateInfo{ usage = { .VERTEX }, size = bytes },
        )
    }

    upload_buffer := sdl.CreateGPUTransferBuffer(
        render_ctx.device,
        sdl.GPUTransferBufferCreateInfo{ usage = .UPLOAD, size = bytes },
    )
    defer sdl.ReleaseGPUTransferBuffer(render_ctx.device, upload_buffer)

    {
        raw_transfer := sdl.MapGPUTransferBuffer(render_ctx.device, upload_buffer, false)
        defer sdl.UnmapGPUTransferBuffer(render_ctx.device, upload_buffer)

        transfer := (transmute([^]GraphVertex)raw_transfer)[:render_ctx.graph.vertex_count]
        copy(transfer[:], mesh[:])
    }

    cmd_buffer := sdl.AcquireGPUCommandBuffer(render_ctx.device)
    defer {
        if !sdl.SubmitGPUCommandBuffer(cmd_buffer)
        {
            fail("Failed to submit command buffer:", sdl.GetError())
        }
    }

    copypass := sdl.BeginGPUCopyPass(cmd_buffer)
    defer sdl.EndGPUCopyPass(copypass)

    sdl.UploadToGPUBuffer(
        copypass,
        sdl.GPUTransferBufferLocation{ transfer_buffer = upload_buffer, offset = 0 },
        sdl.GPUBufferRegion{ buffer = render_ctx.graph.vertex_buffer, offset = 0, size = bytes },
        false,
    )
}

draw_graph :: proc()
{
    target_info := sdl.GPUColorTargetInfo{
        clear_color = { 1, 1, 1, 1 },
        texture  = render_ctx.target,
        load_op  = .CLEAR,
        store_op = .STORE
    }

    renderpass := sdl.BeginGPURenderPass(render_ctx.cmd_buffer, &target_info, 1, nil)
    sdl.BindGPUGraphicsPipeline(renderpass, render_ctx.graph.pipeline)
    sdl.BindGPUVertexBuffers(
        renderpass,
        0,
        raw_data(&[?]sdl.GPUBufferBinding{ { buffer = render_ctx.graph.vertex_buffer } }),
        1,
    )

    sdl.PushGPUVertexUniformData(render_ctx.cmd_buffer, 0, &render_ctx.color, size_of(render_ctx.color))
    sdl.DrawGPUPrimitives(renderpass, render_ctx.graph.vertex_count, 1, 0, 0)
    sdl.EndGPURenderPass(renderpass)
}

render_graph_to_surface :: proc()
{
    render_ctx.cmd_buffer = sdl.AcquireGPUCommandBuffer(render_ctx.device)
    defer {
        ok := sdl.SubmitGPUCommandBuffer(render_ctx.cmd_buffer)
        if !ok do fail("Failed to submit command buffer:", sdl.GetError())

        render_ctx.cmd_buffer = nil
        render_ctx.target = nil
    }

    ok := sdl.WaitAndAcquireGPUSwapchainTexture(render_ctx.cmd_buffer, render_ctx.window, &render_ctx.target, nil, nil)
    if !ok do fail("Failed to acquire swapchain texture:", sdl.GetError())

    if render_ctx.target != nil do draw_graph()
}

render_graph_to_image :: proc(output_dir: string, name: string)
{
    if render_ctx.graph.vertex_count == 0
    {
        fmt.println("Skipping empty mesh:", name)
        return
    }

    render_ctx.target = sdl.CreateGPUTexture(
        render_ctx.device,
        {
            type = .D2,
            format = .R8G8B8A8_UNORM,
            usage = { .COLOR_TARGET },
            width = RENDER_SIZE,
            height = RENDER_SIZE,
            layer_count_or_depth = 1,
            num_levels = 1,
        },
    )
    if render_ctx.target == nil do fail("Failed to create render texture:", sdl.GetError())
    defer {
        sdl.ReleaseGPUTexture(render_ctx.device, render_ctx.target)
        render_ctx.target = nil
    }

    download_buffer := sdl.CreateGPUTransferBuffer(
        render_ctx.device,
        { usage = .DOWNLOAD, size = RENDER_SIZE * RENDER_SIZE * 4 },
    )
    if download_buffer == nil do fail("Failed to create download buffer:", sdl.GetError())
    defer sdl.ReleaseGPUTransferBuffer(render_ctx.device, download_buffer)

    render_ctx.cmd_buffer = sdl.AcquireGPUCommandBuffer(render_ctx.device)
    if render_ctx.cmd_buffer == nil do fail("Failed to acquire command buffer:", sdl.GetError())
    defer render_ctx.cmd_buffer = nil

    draw_graph()

    download_pass := sdl.BeginGPUCopyPass(render_ctx.cmd_buffer)
    sdl.DownloadFromGPUTexture(
        download_pass,
        {
            texture = render_ctx.target,
            w = RENDER_SIZE,
            h = RENDER_SIZE,
            d = 1,
        },
        {
            transfer_buffer = download_buffer,
            offset = 0,
            pixels_per_row = RENDER_SIZE,
            rows_per_layer = RENDER_SIZE,
        },
    )
    sdl.EndGPUCopyPass(download_pass)

    if !sdl.SubmitGPUCommandBuffer(render_ctx.cmd_buffer) do fail("Failed to submit command buffer:", sdl.GetError())
    if !sdl.WaitForGPUIdle(render_ctx.device) do fail("Failed to wait for the GPU:", sdl.GetError())

    pixels := sdl.MapGPUTransferBuffer(render_ctx.device, download_buffer, false)
    if pixels == nil do fail("Failed to map download buffer:", sdl.GetError())
    defer sdl.UnmapGPUTransferBuffer(render_ctx.device, download_buffer)

    path := fmt.tprintf("%s/%s.png", output_dir, name)
    png_write(path, RENDER_SIZE, pixels)

    fmt.println("wrote", path, "(vertices:", render_ctx.graph.vertex_count, ")")
}

load_shader :: proc(
    shaderfile: string,
    sampler_count: u32 = 0,
    uniform_buffer_count: u32 = 0,
    storage_buffer_count: u32 = 0,
    storage_texture_count: u32 = 0,
) -> (^sdl.GPUShader, bool)
{
    stage: sdl.GPUShaderStage

    switch shaderfile[len(shaderfile) - 5:]
    {
    case ".vert":
        stage = .VERTEX
    case ".frag":
        stage = .FRAGMENT
    case:
        fmt.println("Invalid shader stage:", shaderfile)
        return nil, false
    }

    if .SPIRV not_in sdl.GetGPUShaderFormats(render_ctx.device)
    {
        fmt.println("Current GPU device does not support SPIR-V shaders")
        return nil, false
    }

    fullpath := fmt.tprint("shaders/bin/", shaderfile, ".spv", sep="")

    code, err := os.read_entire_file(fullpath, allocator=context.temp_allocator)
    if err != nil
    {
        fmt.println("Failed to read contents of shader file:", fullpath)
        return nil, false
    }

    shader_info := sdl.GPUShaderCreateInfo{
        code       = raw_data(code),
        code_size  = len(code),
        entrypoint = "main",
        stage      = stage,
        format     = { .SPIRV },
        num_samplers         = sampler_count,
        num_uniform_buffers  = uniform_buffer_count,
        num_storage_buffers  = storage_buffer_count,
        num_storage_textures = storage_texture_count,
    }

    shader := sdl.CreateGPUShader(render_ctx.device, shader_info)
    if shader == nil
    {
        fmt.println("Failed to create GPU shader:", sdl.GetError())
        return nil, false
    }

    return shader, true
}

init_graph_render_ctx :: proc(color_format: sdl.GPUTextureFormat) -> bool
{
    vert_shader, vert_ok := load_shader("points.vert", uniform_buffer_count = 1)
    frag_shader, frag_ok := load_shader("solid.frag")
    render_ctx.graph.vert_shader = vert_shader
    render_ctx.graph.frag_shader = frag_shader
    if !vert_ok || !frag_ok do return false

    pipeline_info := sdl.GPUGraphicsPipelineCreateInfo{
        target_info = {
            num_color_targets = 1,
            color_target_descriptions = raw_data(&[?]sdl.GPUColorTargetDescription{
                {
                    format = color_format,
                    blend_state = {
                        enable_blend = true,
                        color_blend_op = .ADD,
                        alpha_blend_op = .ADD,
                        src_color_blendfactor = .SRC_ALPHA,
                        dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
                        src_alpha_blendfactor = .ONE,
                        dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
                    },
                },
            }),
        },
        vertex_input_state = {
            num_vertex_buffers = 1,
            vertex_buffer_descriptions = raw_data(&[?]sdl.GPUVertexBufferDescription{
                {
                    slot  = 0,
                    pitch = size_of(GraphVertex),
                    input_rate = .VERTEX,
                    instance_step_rate = 0,
                },
            }),
            num_vertex_attributes = 1,
            vertex_attributes = raw_data(&[?]sdl.GPUVertexAttribute{
                {
                    buffer_slot = 0,
                    format      = .FLOAT2,
                    location    = 0,
                    offset      = 0,
                },
            }),
        },
        primitive_type  = .TRIANGLELIST,
        vertex_shader   = render_ctx.graph.vert_shader,
        fragment_shader = render_ctx.graph.frag_shader,
    }
    pipeline_info.rasterizer_state.fill_mode = .FILL

    render_ctx.graph.pipeline = sdl.CreateGPUGraphicsPipeline(render_ctx.device, pipeline_info)
    return render_ctx.graph.pipeline != nil
}
