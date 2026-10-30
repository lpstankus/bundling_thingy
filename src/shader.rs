use std::sync::mpsc;
use wgpu::util::DeviceExt;

pub const MAP_UNIFORM_BYTES: u64 = 16;
pub type Readback<T> = mpsc::Receiver<Result<T, wgpu::BufferAsyncError>>;

pub fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn bindings<const N: usize>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    inputs: [(u32, wgpu::BindingResource<'_>); N],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shader inputs"),
        layout,
        entries: &inputs.map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
    })
}

pub fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    layout: Option<&wgpu::PipelineLayout>,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

pub fn readback<T: Send + 'static>(
    encoder: &wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    read: impl FnOnce(Result<&[u8], wgpu::BufferAsyncError>) -> Result<T, wgpu::BufferAsyncError>
    + Send
    + 'static,
) -> Readback<T> {
    let (tx, rx) = mpsc::sync_channel(1);
    let mapped = buffer.clone();
    encoder.map_buffer_on_submit(buffer, wgpu::MapMode::Read, .., move |result| {
        let result = match result {
            Ok(()) => {
                let value = read(Ok(&mapped.get_mapped_range(..)));
                mapped.unmap();
                value
            }
            Err(error) => read(Err(error)),
        };
        let _ = tx.send(result);
    });
    rx
}

pub fn uniform(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{source}", include_str!("shaders/map.wgsl")).into(),
        ),
    })
}
