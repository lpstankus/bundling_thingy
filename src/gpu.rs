use crate::{
    density::{self, Density, DensityEncoder, DensityInput},
    graph::ResampledGraph,
    palette::Palette,
    profiling::GpuTimings,
};
use color_eyre::eyre::{OptionExt, Result, ensure};
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};
use wgpu::{
    SurfaceError::{Lost, Outdated, Timeout},
    TextureFormat::{Bgra8Unorm, Rgb10a2Unorm, Rgba8Unorm},
    util::DeviceExt,
};
use winit::window::Window;

pub const SIZE: u32 = 1600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Gpu {
    pub device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline: wgpu::RenderPipeline,
    density: Density,
    density_binding: wgpu::BindGroup,
    palette: wgpu::Buffer,
    density_timings: GpuTimings,
    render_timings: GpuTimings,
    graph: Option<UploadedGraph>,

    target: wgpu::Texture,
    msaa: wgpu::TextureView,
    download: wgpu::Buffer,

    surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
}

struct UploadedGraph {
    bindings: wgpu::BindGroup,
    segment_count: u32,
}

impl Gpu {
    #[tracing::instrument(name = "gpu.init", skip_all, fields(validation = validation), err)]
    pub async fn new(
        window: Option<Arc<Window>>,
        validation: bool,
        palette: Palette,
    ) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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

        let limits = adapter.limits();
        let timing_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Bundling Thingy"),
                required_features: timing_features,
                required_limits: wgpu::Limits {
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    ..Default::default()
                },
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
                .find(|f| matches!(f, Bgra8Unorm | Rgba8Unorm | Rgb10a2Unorm))
                .ok_or_eyre("surface has no linear UNORM format with blending and 4x MSAA")?;
            surface.configure(&device, &config);
            Some(config)
        } else {
            None
        };

        let format = config.as_ref().map_or(FORMAT, |c| c.format);
        let (width, height) = config
            .as_ref()
            .map_or((SIZE, SIZE), |c| (c.width, c.height));
        let density = Density::new(&device, width, height)?;
        let pipeline = Self::pipeline(&device, format);
        let palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("density palette"),
            contents: bytemuck::cast_slice(&palette.colors()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let density_binding = Self::density_binding(&device, &pipeline, &density.texture, &palette);
        let density_timings = GpuTimings::new(&device, &queue, &density::TIMING_STAGES)?;
        let render_timings = GpuTimings::new(&device, &queue, &["render.draw"])?;

        let target = Self::texture(&device, SIZE, SIZE, FORMAT, 1);
        let msaa = Self::texture(&device, width, height, format, 4)
            .create_view(&wgpu::TextureViewDescriptor::default());

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: u64::from(SIZE * SIZE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            density,
            density_binding,
            palette,
            density_timings,
            render_timings,
            graph: None,
            target,
            msaa,
            download,
            surface: surface.zip(config),
        })
    }

    fn pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("graph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render_paths.wgsl").into()),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("graph"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
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
        })
    }

    fn density_binding(
        device: &wgpu::Device,
        pipeline: &wgpu::RenderPipeline,
        density: &wgpu::Texture,
        palette: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear density"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view = density.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density rendering"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: palette.as_entire_binding(),
                },
            ],
        })
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

    pub fn dimensions(&self) -> [u32; 2] {
        [self.density.texture.width(), self.density.texture.height()]
    }

    #[tracing::instrument(name = "gpu.upload_and_density", skip_all, err)]
    pub fn upload(&mut self, graph: &ResampledGraph) -> Result<()> {
        let upload_timing = tracing::info_span!("gpu.upload").entered();
        ensure!(
            graph.dimensions() == self.dimensions(),
            "resampled graph dimensions differ from render target"
        );
        let segment_count = u32::try_from(graph.segments().len())?;
        let input = DensityInput::new(&self.device, graph)?;
        let segments = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resampled graph segments"),
                contents: bytemuck::cast_slice(graph.segments()),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resampled graph rendering"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.points().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: segments.as_entire_binding(),
                },
            ],
        });
        drop(upload_timing);

        let _density_timing = tracing::info_span!("density.compute").entered();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let density_encoder =
            DensityEncoder::new(&self.device, &mut encoder, &self.density_timings);
        self.density.encode(density_encoder, &input)?;
        self.queue.submit([encoder.finish()]);
        self.graph = Some(UploadedGraph {
            bindings,
            segment_count,
        });
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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.025,
                            b: 0.035,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: self.render_timings.render_pass(),
            });
            if let Some(graph) = &self.graph {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.density_binding, &[]);
                pass.set_bind_group(1, &graph.bindings, &[]);
                pass.draw(0..6, 0..graph.segment_count);
            }
        }

        self.render_timings.resolve(&mut encoder);
        encoder.finish()
    }

    #[tracing::instrument(name = "render.offscreen", skip_all, err)]
    pub fn render(&self) -> Result<()> {
        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        self.wait()
    }

    #[tracing::instrument(name = "pixels.readback", skip_all, err)]
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
        let (tx, rx) = mpsc::sync_channel(1);
        let slice = self.download.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        self.wait()?;
        rx.recv()??;

        let result = slice.get_mapped_range().to_vec();
        self.download.unmap();
        Ok(result)
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        if let Some((surface, config)) = &mut self.surface {
            if config.width == width && config.height == height {
                return Ok(());
            }
            let _timing = tracing::info_span!("gpu.resize", width, height).entered();
            let density = Density::new(&self.device, width, height)?;
            let binding = Self::density_binding(
                &self.device,
                &self.pipeline,
                &density.texture,
                &self.palette,
            );
            self.density = density;
            self.density_binding = binding;
            self.graph = None;

            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
            self.msaa = Self::texture(&self.device, width, height, config.format, 4)
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        Ok(())
    }

    #[tracing::instrument(name = "render.present", skip_all, err)]
    pub fn present(&self, window: &Window) -> Result<()> {
        let (surface, config) = self.surface.as_ref().ok_or_eyre("no window surface")?;

        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(error @ (Lost | Outdated | Timeout)) => {
                if error != Timeout {
                    surface.configure(&self.device, config);
                }
                window.request_redraw();
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        frame.present();
        self.wait()
    }
}
