use crate::{
    Options,
    density::{self, Density},
    graph::Graph,
    neighbor_mean::NeighborMean,
    palette::{ColorBy, Palette},
    profiling::GpuTimings,
    resampling::{self, Resampler},
    shader,
};
use color_eyre::eyre::{OptionExt, Result, ensure};
use std::{sync::Arc, time::Duration};
use wgpu::{
    CommandEncoderDescriptor, PipelineCompilationOptions,
    SurfaceError::{Lost, Outdated, Timeout},
    TextureFormat::{Bgra8Unorm, Rgb10a2Unorm, Rgba8Unorm},
    TextureViewDescriptor,
};
use winit::window::Window;

pub const SIZE: u32 = 1600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    map: wgpu::Buffer,

    pipeline: wgpu::RenderPipeline,
    endpoint_outline_pipeline: wgpu::RenderPipeline,
    endpoint_pipeline: wgpu::RenderPipeline,
    endpoint_dots: bool,
    color_by: ColorBy,
    density: Density,
    density_binding: wgpu::BindGroup,
    palette: wgpu::Buffer,
    density_timings: GpuTimings,
    render_timings: GpuTimings,
    resampling_timings: GpuTimings,
    resampler: Resampler,
    neighbor_mean: Option<NeighborMean>,
    graph: Option<UploadedGraph>,

    target: wgpu::Texture,
    msaa: wgpu::TextureView,
    sample_count: u32,
    download: wgpu::Buffer,

    surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
}

struct UploadedGraph {
    bindings: wgpu::BindGroup,
    geometry: resampling::Geometry,
    samples: u32,
}

impl Gpu {
    #[tracing::instrument(name = "gpu.init", skip_all, fields(validation = opts.validate_gpu), err)]
    pub async fn new(window: Option<Arc<Window>>, opts: &Options) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: if opts.validate_gpu {
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
        let (device, queue) = Self::request_device(&adapter, opts.msaa).await?;

        let config = if let (Some(surface), Some(window)) = (&surface, &window) {
            Some(Self::configure_surface(&device, &adapter, surface, window)?)
        } else {
            None
        };

        let format = config.as_ref().map_or(FORMAT, |c| c.format);
        let (width, height) = config
            .as_ref()
            .map_or((SIZE, SIZE), |c| (c.width, c.height));
        let map = shader::uniform(
            &device,
            "shared map dimensions",
            bytemuck::cast_slice(&[width, height, 0, 0]),
        );
        let density = Density::new(&device, [width, height], &map)?;
        let [pipeline, endpoint_outline_pipeline, endpoint_pipeline] =
            Self::pipelines(&device, &adapter, format, opts.msaa, opts.color_by)?;
        let palette = Self::palette_buffer(&device, opts.palette);
        let density_binding =
            Self::density_binding(&device, &pipeline, &density.texture, &palette, &map);
        let density_timings = GpuTimings::new(&device, &queue, &density::TIMING_STAGES)?;
        let render_timings = GpuTimings::new(&device, &queue, &["render.draw"])?;
        let resampler = Resampler::new(&device, opts, [width, height])?;
        let resampling_timings = GpuTimings::new(&device, &queue, resampler.timing_stages())?;
        let neighbor_mean = opts
            .neighbor_radius
            .map(|radius| {
                NeighborMean::new(
                    &device,
                    &queue,
                    radius,
                    [width, height],
                    opts.color_by,
                    &map,
                )
            })
            .transpose()?;
        tracing::info!(adapter = %info.name, max_buffer_bytes = limits.max_buffer_size, max_storage_binding_bytes = limits.max_storage_buffer_binding_size, "GPU graph processing");

        let target = Self::texture(&device, SIZE, SIZE, FORMAT, 1);
        let msaa = Self::texture(&device, width, height, format, opts.msaa)
            .create_view(&TextureViewDescriptor::default());
        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: u64::from(SIZE * SIZE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            map,
            pipeline,
            endpoint_outline_pipeline,
            endpoint_pipeline,
            endpoint_dots: opts.endpoint_dots,
            color_by: opts.color_by,
            density,
            density_binding,
            palette,
            density_timings,
            render_timings,
            resampling_timings,
            resampler,
            neighbor_mean,
            graph: None,
            target,
            msaa,
            sample_count: opts.msaa,
            download,
            surface: surface.zip(config),
        })
    }

    async fn request_device(
        adapter: &wgpu::Adapter,
        samples: u32,
    ) -> Result<(wgpu::Device, wgpu::Queue)> {
        let limits = adapter.limits();
        let mut features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        if samples == 2 {
            features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        Ok(adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Bundling Thingy"),
                required_features: features,
                required_limits: wgpu::Limits {
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?)
    }

    fn configure_surface(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface<'_>,
        window: &Window,
    ) -> Result<wgpu::SurfaceConfiguration> {
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or_eyre("no compatible surface configuration")?;
        let caps = surface.get_capabilities(adapter);
        config.format = *caps
            .formats
            .iter()
            .find(|f| matches!(f, Bgra8Unorm | Rgba8Unorm | Rgb10a2Unorm))
            .ok_or_eyre("surface has no linear UNORM format with blending")?;
        surface.configure(device, &config);
        Ok(config)
    }

    fn palette_buffer(device: &wgpu::Device, palette: Palette) -> wgpu::Buffer {
        let mut colors = Vec::from(palette.colors());
        colors.push(palette.endpoint_color());
        shader::uniform(
            device,
            "path and endpoint colors",
            bytemuck::cast_slice(&colors),
        )
    }

    fn pipelines(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        format: wgpu::TextureFormat,
        samples: u32,
        color_by: ColorBy,
    ) -> Result<[wgpu::RenderPipeline; 3]> {
        let flags = adapter.get_texture_format_features(format).flags;
        ensure!(
            flags.sample_count_supported(samples),
            "{samples}x MSAA is not supported for {format:?}"
        );
        let shader = shader::module(device, "graph", include_str!("shaders/render_paths.wgsl"));
        let density_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density and palette"),
            entries: &[0, 1, 2, 9].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: match binding {
                    0 => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    1 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    _ => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                count: None,
            }),
        });
        let graph_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resampled points, segments, endpoint dots, and edges"),
            entries: &[0, 1, 2, 3].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("paths and endpoints"),
            bind_group_layouts: &[&density_layout, &graph_layout],
            push_constant_ranges: &[],
        });

        let constants = [(
            "color_by_edge",
            f64::from(u8::from(color_by == ColorBy::EdgeId)),
        )];
        let pipeline = |vertex, fragment| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex),
                    compilation_options: PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment),
                    compilation_options: PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        Ok([
            pipeline("vs_main", "fs_main"),
            pipeline("vs_endpoint", "fs_endpoint_outline"),
            pipeline("vs_endpoint", "fs_endpoint"),
        ])
    }

    fn density_binding(
        device: &wgpu::Device,
        pipeline: &wgpu::RenderPipeline,
        density: &wgpu::Texture,
        palette: &wgpu::Buffer,
        map: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear density"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view = density.create_view(&TextureViewDescriptor::default());
        shader::bindings(
            device,
            &pipeline.get_bind_group_layout(0),
            [
                (0, wgpu::BindingResource::TextureView(&view)),
                (1, wgpu::BindingResource::Sampler(&sampler)),
                (2, palette.as_entire_binding()),
                (9, map.as_entire_binding()),
            ],
        )
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

    fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })?;
        Ok(())
    }

    #[tracing::instrument(name = "gpu.process_graph", skip_all, err)]
    pub fn upload(&mut self, graph: &Graph) -> Result<u32> {
        self.graph = None;
        if let Some(neighbor_mean) = &mut self.neighbor_mean {
            neighbor_mean.reset();
        }
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        if self.color_by == ColorBy::Density {
            self.density.clear(&mut encoder);
        }
        let pending = self.resampler.encode(
            &self.device,
            &mut encoder,
            graph,
            self.density.counts(),
            &self.map,
            &self.resampling_timings,
        )?;
        if self.color_by == ColorBy::Density {
            self.density.encode(&mut encoder, &self.density_timings);
        }
        self.queue.submit([encoder.finish()]);
        self.wait()?;
        let (geometry, samples) = pending.finish()?;
        let bindings = Self::graph_bindings(&self.device, &self.pipeline, &geometry);
        self.graph = Some(UploadedGraph {
            bindings,
            geometry,
            samples,
        });
        Ok(samples)
    }

    #[tracing::instrument(name = "graph.iteration", skip_all, err)]
    pub fn iterate(&mut self) -> Result<()> {
        let neighbor_mean = self
            .neighbor_mean
            .as_mut()
            .ok_or_eyre("neighbor radius is not configured")?;
        let graph = self.graph.as_mut().ok_or_eyre("no uploaded graph")?;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        if self.color_by == ColorBy::Density {
            self.density.clear(&mut encoder);
        }
        neighbor_mean.encode(
            &self.device,
            &mut encoder,
            &mut graph.geometry,
            graph.samples,
            self.density.counts(),
        )?;
        if self.color_by == ColorBy::Density {
            self.density.encode(&mut encoder, &self.density_timings);
        }
        graph.bindings = Self::graph_bindings(&self.device, &self.pipeline, &graph.geometry);
        self.queue.submit([encoder.finish()]);
        self.wait()
    }

    fn graph_bindings(
        device: &wgpu::Device,
        pipeline: &wgpu::RenderPipeline,
        geometry: &resampling::Geometry,
    ) -> wgpu::BindGroup {
        shader::bindings(
            device,
            &pipeline.get_bind_group_layout(1),
            [
                (0, geometry.points.as_entire_binding()),
                (1, geometry.segments.as_entire_binding()),
                (2, geometry.endpoints.as_entire_binding()),
                (3, geometry.edges.as_entire_binding()),
            ],
        )
    }

    fn encode_draw(&self, view: &wgpu::TextureView) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
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
                timestamp_writes: self.render_timings.render_pass(),
                ..Default::default()
            });
            if let Some(graph) = &self.graph {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.density_binding, &[]);
                pass.set_bind_group(1, &graph.bindings, &[]);
                pass.draw_indirect(&graph.geometry.commands, resampling::PATH_DRAW_OFFSET);
                if self.endpoint_dots {
                    // Draw all outlines first so shared endpoints keep their colored centers.
                    pass.set_pipeline(&self.endpoint_outline_pipeline);
                    pass.draw_indirect(&graph.geometry.commands, resampling::ENDPOINT_DRAW_OFFSET);
                    pass.set_pipeline(&self.endpoint_pipeline);
                    pass.draw_indirect(&graph.geometry.commands, resampling::ENDPOINT_DRAW_OFFSET);
                }
            }
        }

        self.render_timings.resolve(&mut encoder);
        encoder.finish()
    }

    #[tracing::instrument(name = "render.offscreen", skip_all, err)]
    pub fn render(&self) -> Result<()> {
        let view = self.target.create_view(&TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        self.wait()
    }

    #[tracing::instrument(name = "pixels.readback", skip_all, err)]
    pub fn readback(&self) -> Result<Vec<u8>> {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
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

        let pixels = shader::readback(&encoder, &self.download, |result| {
            result.map(<[u8]>::to_vec)
        });
        self.queue.submit([encoder.finish()]);
        self.wait()?;
        Ok(pixels.recv()??)
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
            if let Some(neighbor_mean) = &mut self.neighbor_mean {
                self.resampler
                    .resize_grid([width, height], neighbor_mean.radius())?;
                neighbor_mean.resize(&self.device, [width, height])?;
            }
            let density = self
                .density
                .resize(&self.device, [width, height], &self.map)?;
            let binding = Self::density_binding(
                &self.device,
                &self.pipeline,
                &density.texture,
                &self.palette,
                &self.map,
            );
            self.queue
                .write_buffer(&self.map, 0, bytemuck::cast_slice(&[width, height, 0, 0]));
            self.density = density;
            self.density_binding = binding;
            self.graph = None;

            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
            self.msaa = Self::texture(
                &self.device,
                width,
                height,
                config.format,
                self.sample_count,
            )
            .create_view(&TextureViewDescriptor::default());
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
        let view = frame.texture.create_view(&TextureViewDescriptor::default());

        self.queue.submit([self.encode_draw(&view)]);
        frame.present();
        self.wait()
    }
}
