#![allow(clippy::arithmetic_side_effects)]

use super::{Density, DensityInput};
use crate::graph::{Graph, Point};
use color_eyre::eyre::{Result, ensure};
use std::{sync::mpsc, time::Duration};

const WIDTH: u32 = 29;
const HEIGHT: u32 = 23;

#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_density_matches_direct_gaussian_convolution() -> Result<()> {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::debugging(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        ensure!(
            adapter.get_info().device_type != wgpu::DeviceType::Cpu,
            "this test requires a hardware GPU"
        );
        Ok::<_, color_eyre::Report>(
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await?,
        )
    })?;
    let density = Density::new(&device, WIDTH, HEIGHT)?;
    let width = usize::try_from(WIDTH)?;
    let mut points = Vec::new();
    let mut expected_counts = vec![0_u32; width * usize::try_from(HEIGHT)?];

    // Exercise collisions across workgroups, nearest-cell rounding, orientation,
    // map boundaries, and partial 16x16 workgroups on a rectangular map.
    for (x, y, copies, cell_x, cell_y) in [
        (0.1_f32, 0.1_f32, 513_u16, 0_usize, 0_usize),
        (27.9, 21.9, 7, 28, 22),
        (4.49, 17.51, 3, 4, 18),
        (12.51, 6.49, 9, 13, 6),
        (9.0, 11.0, 2, 9, 11),
        (-3.0, 25.0, 1, 0, 22),
    ] {
        let point = Point::new((x / 28.0).mul_add(2.0, -1.0), (y / 22.0).mul_add(-2.0, 1.0));
        points.extend(std::iter::repeat_n(point, usize::from(copies)));
        expected_counts[cell_y * width + cell_x] += u32::from(copies);
    }
    let input = DensityInput::new(&device, &singletons(&points)?)?;
    verify(&device, &queue, &density, Some(&input), &expected_counts)?;

    // Reusing the map must clear both counts and the previous, larger maximum.
    expected_counts.fill(0);
    expected_counts[11 * width + 14] = 1;
    let input = DensityInput::new(&device, &singletons(&[Point::ZERO])?)?;
    verify(&device, &queue, &density, Some(&input), &expected_counts)?;
    expected_counts.fill(0);
    verify(&device, &queue, &density, None, &expected_counts)?;
    verify_spacing(&device, &queue)
}

fn singletons(points: &[Point]) -> Result<Graph> {
    let mut graph = Graph::parse("0: 0 0")?;
    graph.points = points.to_vec();
    graph.starts = (0..=points.len()).collect();
    Ok(graph)
}

fn verify_spacing(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
    for text in [
        "0: -1 0 1 0",
        "0: -1 0 -0.25 0 0 0 0.5 0 1 0",
        "0: -1 0 -1 0 1 0",
    ] {
        let graph = Graph::parse(text)?;
        let input = DensityInput::new(device, &graph)?;
        // Reuse the uploaded geometry at different render dimensions.
        for (width, height) in [(29, 23), (57, 45), (1, 23), (29, 1), (1, 1)] {
            let density = Density::new(device, width, height)?;
            let width = usize::try_from(width)?;
            let height = usize::try_from(height)?;
            let mut expected = vec![0_u32; width * height];
            expected[height / 2 * width..(height / 2 + 1) * width].fill(1);
            verify(device, queue, &density, Some(&input), &expected)?;
        }
    }

    // A 3:4:5 diagonal has an exact integer length, with no rounding ties.
    let graph = Graph::parse("0: -1 -1 1 1")?;
    let input = DensityInput::new(device, &graph)?;
    for (width, height, length) in [(25, 33, 40), (49, 65, 80)] {
        let density = Density::new(device, width, height)?;
        let width = usize::try_from(width)?;
        let height = usize::try_from(height)?;
        let mut expected = vec![0_u32; width * height];
        for step in 0..=length {
            let x = (step * 3 + 2) / 5;
            let y = height - 1 - (step * 4 + 2) / 5;
            expected[y * width + x] += 1;
        }
        verify(device, queue, &density, Some(&input), &expected)?;
    }

    let density = Density::new(device, 33, 17)?;
    for (text, ranges) in [
        ("0: -1 0 -0.5 0\n1: 0.5 0 1 0", vec![(0, 8), (24, 32)]),
        ("0: -0.734375 0 0.734375 0", vec![(4, 28)]),
        ("0: 0 0 0 0 0 0", vec![(16, 16)]),
    ] {
        let graph = Graph::parse(text)?;
        let input = DensityInput::new(device, &graph)?;
        let mut expected = vec![0_u32; 33 * 17];
        for (start, end) in ranges {
            for x in start..=end {
                expected[8 * 33 + x] = 1;
            }
        }
        verify(device, queue, &density, Some(&input), &expected)?;
    }
    ensure!(
        Density::new(device, 0, 17).is_err(),
        "accepted a zero-width map"
    );
    Ok(())
}

fn verify(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    density: &Density,
    input: Option<&DensityInput>,
    expected_counts: &[u32],
) -> Result<()> {
    let size = density.texture.size();
    let width = usize::try_from(size.width)?;
    let height = usize::try_from(size.height)?;
    let count_bytes = u64::from(size.width * size.height * 4);
    let texture_offset = count_bytes.next_multiple_of(256);
    let row_bytes = (size.width * 8).next_multiple_of(256);
    let download = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density test readback"),
        size: texture_offset + u64::from(row_bytes * size.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    density.encode(device, &mut encoder, input);
    encoder.copy_buffer_to_buffer(&density.counts, 0, &download, 0, count_bytes);
    encoder.copy_texture_to_buffer(
        density.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &download,
            layout: wgpu::TexelCopyBufferLayout {
                offset: texture_offset,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.height),
            },
        },
        density.texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::sync_channel(1);
    download.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).expect("return density readback result");
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(10)),
    })?;
    rx.recv()??;
    let data = download.get_mapped_range(..);
    let actual_counts: Vec<_> = data[..usize::try_from(count_bytes)?]
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect();
    ensure!(
        actual_counts == expected_counts,
        "GPU sample counts differ on a {}x{} map; first mismatch {:?}",
        size.width,
        size.height,
        actual_counts
            .iter()
            .zip(expected_counts)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
    );

    let reference = direct_convolution(expected_counts, width, height);
    let peak = reference.iter().copied().fold(0.0, f64::max);
    for y in 0..height {
        for x in 0..width {
            let offset = usize::try_from(texture_offset)? + y * usize::try_from(row_bytes)? + x * 8;
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
            let actual = unit_half(bits);
            let expected = if peak > 0.0 {
                reference[y * width + x] / peak
            } else {
                0.0
            };
            ensure!(
                (actual - expected).abs() < 0.001,
                "density at ({x}, {y}): {actual}, expected {expected}"
            );
            ensure!(
                data[offset + 2..offset + 6] == [0; 4],
                "nonzero unused channels"
            );
            ensure!(
                u16::from_le_bytes([data[offset + 6], data[offset + 7]]) == 0x3c00,
                "invalid density alpha"
            );
        }
    }
    drop(data);
    download.unmap();
    Ok(())
}

fn direct_convolution(counts: &[u32], width: usize, height: usize) -> Vec<f64> {
    let weight_sum: f64 = (-18_i32..=18)
        .map(|offset| (-f64::from(offset * offset) / 72.0).exp())
        .sum();
    let mut result = vec![0.0; width * height];
    for (source, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        for y in 0..height {
            for x in 0..width {
                let dx = u32::try_from(x.abs_diff(source % width)).expect("small test grid");
                let dy = u32::try_from(y.abs_diff(source / width)).expect("small test grid");
                if dx <= 18 && dy <= 18 {
                    let weight = (-f64::from(dx * dx + dy * dy) / 72.0).exp();
                    result[y * width + x] += f64::from(count) * weight / weight_sum.powi(2);
                }
            }
        }
    }
    result
}

fn unit_half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 0x1f;
    let fraction = f64::from(bits & 0x03ff);
    assert_eq!(bits & 0x8000, 0, "density must be non-negative");
    assert!(exponent < 31, "density must be finite");
    let value = if exponent == 0 {
        fraction * 2.0_f64.powi(-24)
    } else {
        (1.0 + fraction / 1024.0) * 2.0_f64.powi(i32::from(exponent) - 15)
    };
    assert!(value <= 1.0, "density must be normalized");
    value
}
