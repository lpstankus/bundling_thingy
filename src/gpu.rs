use crate::graph::Point;
use color_eyre::eyre::{OptionExt, Result, ensure};
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};
use wgpu::util::DeviceExt;
use winit::window::Window;

pub const SIZE: u32 = 1600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Gpu {
    pub device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline: wgpu::RenderPipeline,
    color: wgpu::BindGroup,

    vertices: wgpu::Buffer,
    vertex_count: u32,
    capacity: u64,

    target: wgpu::Texture,
    msaa: wgpu::TextureView,
    download: wgpu::Buffer,

    pub surface: Option<wgpu::Surface<'static>>,
    pub config: Option<wgpu::SurfaceConfiguration>,
}

impl Gpu {
    pub async fn new(window: Option<Arc<Window>>, validation: bool) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: if validation {
                wgpu::InstanceFlags::debugging()
            } else {
                wgpu::InstanceFlags::empty()
            },
            ..Default::default()
        });

        let surface = window
            .as_ref()
            .map(|w| instance.create_surface(w.clone()))
            .transpose()?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface.as_ref(),
            })
            .await?;

        let info = adapter.get_info();
        ensure!(
            info.device_type != wgpu::DeviceType::Cpu,
            "software GPU is not permitted"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Bundling Thingy"),
                ..Default::default()
            })
            .await?;

        let config = if let (Some(surface), Some(window)) = (&surface, &window) {
            let size = window.inner_size();
            let mut config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .ok_or_eyre("no compatible surface configuration")?;
            let caps = surface.get_capabilities(&adapter);
            config.format = *caps
                .formats
                .iter()
                .find(|f| !f.is_srgb())
                .ok_or_eyre("surface has no linear UNORM format")?;
            surface.configure(&device, &config);
            Some(config)
        } else {
            None
        };

        let format = config.as_ref().map_or(FORMAT, |c| c.format);
        ensure!(
            adapter
                .get_texture_format_features(format)
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4),
            "4x MSAA unsupported"
        );

        let (pipeline, color) = Self::pipeline(&device, format);

        let target = Self::texture(&device, SIZE, SIZE, FORMAT, 1);
        let (width, height) = config
            .as_ref()
            .map_or((SIZE, SIZE), |c| (c.width, c.height));
        let msaa = Self::texture(&device, width, height, format, 4)
            .create_view(&wgpu::TextureViewDescriptor::default());

        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let vertices = buffer(
            "vertices",
            8,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );

        let download = buffer(
            "pixels",
            u64::from(SIZE * SIZE * 4),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            color,
            vertices,
            vertex_count: 0,
            capacity: 8,
            target,
            msaa,
            download,
            surface,
            config,
        })
    }

    fn pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("graph"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color"),
            contents: bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 0.05]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let color = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        (pipeline, color)
    }

    fn texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | if samples == 1 {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        })
    }

    pub fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })?;
        Ok(())
    }

    pub fn upload(&mut self, mesh: &[Point]) -> Result<()> {
        self.vertex_count = u32::try_from(mesh.len())?;
        if mesh.is_empty() {
            return Ok(());
        }

        let bytes = bytemuck::cast_slice(mesh);

        let byte_count = u64::try_from(bytes.len())?;
        ensure!(
            byte_count <= self.device.limits().max_buffer_size,
            "mesh exceeds GPU buffer limit"
        );

        if byte_count > self.capacity {
            self.capacity = byte_count;
            self.vertices = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vertices"),
                size: self.capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        self.queue.write_buffer(&self.vertices, 0, bytes);

        self.queue.submit([]);
        self.wait()
    }

    fn encode_draw(&self, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("graph"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa,
                    depth_slice: None,
                    resolve_target: Some(view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.color, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }

        encoder.finish()
    }

    pub fn render(&self) -> Result<()> {
        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        self.wait()
    }

    fn map_copy(&self, buffer: &wgpu::Buffer) -> Result<Vec<u8>> {
        let (tx, rx) = mpsc::sync_channel(1);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        self.wait()?;
        rx.recv()??;

        let result = buffer.slice(..).get_mapped_range().to_vec();
        buffer.unmap();
        Ok(result)
    }

    pub fn readback(&self) -> Result<Vec<u8>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.download,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * 4),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);
        self.map_copy(&self.download)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
            self.msaa = Self::texture(&self.device, width, height, config.format, 4)
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    pub fn present(&self) -> Result<()> {
        let surface = self.surface.as_ref().ok_or_eyre("no window surface")?;

        let frame = surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        frame.present();
        self.wait()
    }
}
