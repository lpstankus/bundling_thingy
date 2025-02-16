package main

import "core:os"
import "core:fmt"
import sdl "vendor:sdl3"

RenderContext :: struct
{
    window:   ^sdl.Window,
    renderer: ^sdl.Renderer,
    device:   ^sdl.GPUDevice,

    graph: GraphRenderCtx,
}

GraphRenderCtx :: struct
{
    vert_shader: ^sdl.GPUShader,
    frag_shader: ^sdl.GPUShader,

    vertex_edges: u32,
    vertex_bytes: u32,
    vertex_buffer: ^sdl.GPUBuffer,

    pipeline: ^sdl.GPUGraphicsPipeline,
}

@(private="file")
render_ctx : RenderContext

init_app :: proc()
{
    using render_ctx

    if !sdl.Init(sdl.INIT_VIDEO)
    {
        fmt.println("Failed to init SDL:", sdl.GetError())
        os.exit(1)
    }

    if !sdl.CreateWindowAndRenderer("Bundling thingy", 800, 800, nil, &window, &renderer)
    {
        fmt.println("Failed to create window:", sdl.GetError())
        os.exit(1)
    }

    device = sdl.CreateGPUDevice({ .SPIRV }, true, nil)
    if device == nil
    {
        fmt.println("Failed to create GPU device:", sdl.GetError())
        os.exit(1)
    }

    if !sdl.ClaimWindowForGPUDevice(device, window)
    {
        fmt.println("Failed to claim window for GPU device:", sdl.GetError())
        os.exit(1)
    }

    { // Graph drawing render ctx
        graph.vert_shader = load_shader("points.vert", uniform_buffer_count = 1) or_else os.exit(1)
        graph.frag_shader = load_shader("solid.frag") or_else os.exit(1)

        pipeline_info := sdl.GPUGraphicsPipelineCreateInfo{
            target_info = {
                num_color_targets = 1,
                color_target_descriptions = raw_data(&[?]sdl.GPUColorTargetDescription{
                    {
                        format = sdl.GetGPUSwapchainTextureFormat(device, window),
                        blend_state = {
                            enable_blend = true,
                            color_blend_op = .ADD,
                            alpha_blend_op = .ADD,
                            src_color_blendfactor = .SRC_ALPHA,
                            dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
                            src_alpha_blendfactor = .SRC_ALPHA,
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
                        pitch = size_of(Point2D),
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
                    }
                }),
            },
            primitive_type  = .LINELIST,
            vertex_shader   = graph.vert_shader,
            fragment_shader = graph.frag_shader,
        }
        pipeline_info.rasterizer_state.fill_mode = .FILL

        graph.pipeline = sdl.CreateGPUGraphicsPipeline(device, pipeline_info)
        if graph.pipeline == nil
        {
            fmt.println("Failed to create fill graphics pipeline:", sdl.GetError())
            os.exit(1)
        }

        graph.vertex_edges = 5000
        graph.vertex_bytes = u32(5000 * size_of([2]Point2D))
        graph.vertex_buffer = sdl.CreateGPUBuffer(
            device,
            sdl.GPUBufferCreateInfo{ usage = { .VERTEX }, size = graph.vertex_bytes },
        )
    }
}

deinit_app :: proc()
{
    using render_ctx

    sdl.ReleaseGPUBuffer(device, graph.vertex_buffer)
    sdl.ReleaseGPUGraphicsPipeline(device, graph.pipeline)

    sdl.ReleaseGPUShader(device, graph.frag_shader)
    sdl.ReleaseGPUShader(device, graph.vert_shader)

    sdl.DestroyGPUDevice(device)
    sdl.DestroyWindow(window)
    sdl.Quit()

}

upload_to_gpu :: proc { graphdrawing_upload_to_gpu }

graphdrawing_upload_to_gpu :: proc(gd: GraphDrawing)
{
    using render_ctx

    graph.vertex_edges = u32(len(gd.endpoints)) / 2
    bytes := u32(len(gd.endpoints) * size_of([2]Point2D))
    if graph.vertex_bytes < bytes
    {
        fmt.println("Resizing graph vertex buffer, from", graph.vertex_bytes, "to", bytes)

        graph.vertex_bytes = bytes

        sdl.ReleaseGPUBuffer(device, graph.vertex_buffer)
        graph.vertex_buffer = sdl.CreateGPUBuffer(
            device,
            sdl.GPUBufferCreateInfo{ usage = { .VERTEX }, size = bytes },
        )
    }

    upload_buffer := sdl.CreateGPUTransferBuffer(
        render_ctx.device,
        sdl.GPUTransferBufferCreateInfo{
            usage = .UPLOAD,
            size = u32(len(gd.endpoints) * size_of([2]Point2D)) * 2,
        }
    )
    defer sdl.ReleaseGPUTransferBuffer(render_ctx.device, upload_buffer)

    {
        raw_transfer := sdl.MapGPUTransferBuffer(render_ctx.device, upload_buffer, false)
        defer sdl.UnmapGPUTransferBuffer(render_ctx.device, upload_buffer)

        transfer := (transmute([^][2]Point2D)raw_transfer)[:len(gd.endpoints)]
        copy(transfer[:], gd.endpoints[:])
    }

    cmd_buffer := sdl.AcquireGPUCommandBuffer(render_ctx.device)
    defer {
        if !sdl.SubmitGPUCommandBuffer(cmd_buffer)
        {
            fmt.println("Failed to submit command buffer:", sdl.GetError())
            os.exit(1)
        }
    }

    copypass := sdl.BeginGPUCopyPass(cmd_buffer)
    defer sdl.EndGPUCopyPass(copypass)

    sdl.UploadToGPUBuffer(
        copypass,
        sdl.GPUTransferBufferLocation{
            transfer_buffer = upload_buffer,
            offset = 0,
        },
        sdl.GPUBufferRegion{
            buffer = graph.vertex_buffer,
            offset = 0,
            size = u32(len(gd.endpoints) * size_of([2]Point2D)),
        },
        false,
    )
}

render_current_graph :: proc()
{
    using render_ctx

    cmd_buffer := sdl.AcquireGPUCommandBuffer(device)
    defer {
        if !sdl.SubmitGPUCommandBuffer(cmd_buffer)
        {
            fmt.println("Failed to submit command buffer:", sdl.GetError())
            os.exit(1)
        }
    }

    swapchain_tex: ^sdl.GPUTexture
    if !sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buffer, window, &swapchain_tex, nil, nil)
    {
        fmt.println("Failed to acquire swapchain texture:", sdl.GetError())
        os.exit(1)
    }

    if swapchain_tex != nil
    {
        target_info := sdl.GPUColorTargetInfo{
            clear_color = { 1, 1, 1, 1 },
            texture  = swapchain_tex,
            load_op  = .CLEAR,
            store_op = .STORE
        }

        renderpass := sdl.BeginGPURenderPass(cmd_buffer, &target_info, 1, nil)
        defer sdl.EndGPURenderPass(renderpass)

        sdl.BindGPUGraphicsPipeline(renderpass, graph.pipeline)
        sdl.BindGPUVertexBuffers(
            renderpass,
            0,
            raw_data(&[?]sdl.GPUBufferBinding{ { buffer = graph.vertex_buffer } }),
            1,
        )

        color_uniform := [4]f32{ 1.0, 0.0, 0.0, 0.05 }
        sdl.PushGPUVertexUniformData(cmd_buffer, 0, &color_uniform, size_of(type_of(color_uniform)))
        sdl.DrawGPUPrimitives(renderpass, graph.vertex_edges, 1, 0, 0)
    }
}

load_shader :: proc(
    shaderfile: string,
    sampler_count: u32 = 0,
    uniform_buffer_count: u32 = 0,
    storage_buffer_count: u32 = 0,
    storage_texture_count: u32 = 0,
) -> (^sdl.GPUShader, bool)
{
    using render_ctx

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

    if .SPIRV not_in sdl.GetGPUShaderFormats(device)
    {
        fmt.println("Current GPU device does not support SPIR-V shaders")
        return nil, false
    }

    fullpath := fmt.tprint("shaders/bin/", shaderfile, ".spv", sep="")

    code, ok := os.read_entire_file(fullpath, allocator=context.temp_allocator)
    if !ok
    {
        fmt.println("Failed to read contents of shader file:", fullpath)
        return nil, false
    }

    shader_info := sdl.GPUShaderCreateInfo{
        code       = raw_data(code),
        code_size  = len(code),
        entrypoint = "main",
        stage      = stage,
        format     = {.SPIRV},
        num_samplers         = sampler_count,
        num_uniform_buffers  = uniform_buffer_count,
        num_storage_buffers  = storage_buffer_count,
        num_storage_textures = storage_texture_count,
    }

    shader := sdl.CreateGPUShader(device, shader_info)
    if shader == nil
    {
        fmt.println("Failed to create GPU shader:", sdl.GetError())
        return nil, false
    }

    return shader, true
}
