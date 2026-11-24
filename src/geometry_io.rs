use color_eyre::eyre::{OptionExt, Result, WrapErr, ensure};
use std::{fs, path::Path};

pub struct GeometryData {
    pub points: Vec<u8>,
    pub paths: Vec<u8>,
    pub edges: Vec<u8>,
    pub segments: Vec<u32>,
    pub endpoints: Vec<u32>,
    pub samples: u32,
}

impl GeometryData {
    pub fn load(
        folder: &Path,
        budget: u64,
        binding_limit: u64,
        extra_sample_bytes: u64,
    ) -> Result<Self> {
        let mut lengths = [0u64; 3];
        for ((name, stride), length) in [("points.bin", 12), ("paths.bin", 12), ("edges.bin", 16)]
            .into_iter()
            .zip(&mut lengths)
        {
            let path = folder.join(name);
            let bytes = fs::metadata(&path)
                .wrap_err_with(|| format!("read {}", path.display()))?
                .len();
            ensure!(
                bytes > 0 && bytes.is_multiple_of(stride),
                "invalid {name} record length"
            );
            ensure!(bytes <= binding_limit, "{name} exceeds GPU buffer limits");
            *length = bytes;
        }
        let [point_bytes, path_bytes, edge_bytes] = lengths;
        let samples = u32::try_from(point_bytes / 12)?;
        let paths_count = path_bytes / 12;
        ensure!(
            paths_count == edge_bytes / 16,
            "path and edge counts differ"
        );
        let estimated_bytes = point_bytes
            .saturating_add(path_bytes)
            .saturating_add(edge_bytes)
            .saturating_add(u64::from(samples).saturating_mul(extra_sample_bytes.saturating_add(4)))
            .saturating_add(paths_count.saturating_mul(8))
            .saturating_add(48);
        ensure!(
            estimated_bytes <= budget,
            "imported geometry exceeds --max-graph-memory-mib"
        );
        let mut data = Self {
            points: fs::read(folder.join("points.bin"))?,
            paths: fs::read(folder.join("paths.bin"))?,
            edges: fs::read(folder.join("edges.bin"))?,
            segments: Vec::new(),
            endpoints: Vec::new(),
            samples,
        };
        let actual_lengths = [
            u64::try_from(data.points.len())?,
            u64::try_from(data.paths.len())?,
            u64::try_from(data.edges.len())?,
        ];
        ensure!(
            actual_lengths == lengths,
            "geometry files changed while reading"
        );
        data.validate_paths()?;
        Ok(data)
    }

    fn validate_paths(&mut self) -> Result<()> {
        let mut active_paths = 0u32;
        for path in self.paths.as_chunks::<12>().0 {
            let [_, _, count] = bytemuck::pod_read_unaligned::<[u32; 3]>(path);
            active_paths = active_paths.saturating_add(u32::from(count > 0));
        }
        let segment_count = self
            .samples
            .checked_sub(active_paths)
            .ok_or_eyre("more paths than samples")?;
        self.segments = vec![0; usize::try_from(segment_count)?];
        let mut ranges = Vec::new();
        let mut segment_ranges = Vec::new();
        for ((path, edge), edge_id) in self
            .paths
            .as_chunks::<12>()
            .0
            .iter()
            .zip(self.edges.as_chunks::<16>().0)
            .zip(0u32..)
        {
            let [offset, segment_offset, count] = bytemuck::pod_read_unaligned::<[u32; 3]>(path);
            if count == 0 {
                continue;
            }
            ensure!(count >= 2, "path {edge_id} has fewer than two points");
            let end = offset
                .checked_add(count)
                .ok_or_eyre("path range overflow")?;
            ensure!(
                end <= self.samples,
                "path {edge_id} exceeds the point buffer"
            );
            let first_byte = usize::try_from(u64::from(offset).saturating_mul(12))?;
            let last_byte = usize::try_from(u64::from(end).saturating_mul(12))?;
            let points = self
                .points
                .get(first_byte..last_byte)
                .ok_or_eyre("invalid point range")?;
            for point in points.as_chunks::<12>().0 {
                let [x, y, id] = bytemuck::pod_read_unaligned::<[u32; 3]>(point);
                let valid_position = (-1.0..=1.0).contains(&f32::from_bits(x))
                    && (-1.0..=1.0).contains(&f32::from_bits(y));
                ensure!(
                    valid_position,
                    "path {edge_id} has an invalid clip-space point"
                );
                ensure!(
                    id == edge_id,
                    "path {edge_id} contains another edge's points"
                );
            }
            let [first_x, first_y, _] = bytemuck::pod_read_unaligned::<[u32; 3]>(
                points
                    .first_chunk::<12>()
                    .ok_or_eyre("missing first point")?,
            );
            let [last_x, last_y, _] = bytemuck::pod_read_unaligned::<[u32; 3]>(
                points.last_chunk::<12>().ok_or_eyre("missing last point")?,
            );
            let actual = [first_x, first_y, last_x, last_y];
            let expected = bytemuck::pod_read_unaligned::<[u32; 4]>(edge);
            ensure!(
                actual == expected,
                "path {edge_id} endpoints differ from edge metadata"
            );
            let segment_end = segment_offset
                .checked_add(count.saturating_sub(1))
                .ok_or_eyre("segment range overflow")?;
            let segments = self
                .segments
                .get_mut(usize::try_from(segment_offset)?..usize::try_from(segment_end)?)
                .ok_or_eyre("invalid segment range")?;
            for (segment, index) in segments.iter_mut().zip(offset..end.saturating_sub(1)) {
                *segment = index;
            }
            self.endpoints.extend([offset, end.saturating_sub(1)]);
            ranges.push((offset, end));
            segment_ranges.push((segment_offset, segment_end));
        }
        validate_partition(ranges, self.samples, "point")?;
        validate_partition(segment_ranges, segment_count, "segment")?;
        Ok(())
    }
}

fn validate_partition(mut ranges: Vec<(u32, u32)>, expected: u32, name: &str) -> Result<()> {
    ranges.sort_unstable();
    let mut cursor = 0;
    for (start, end) in ranges {
        ensure!(start == cursor, "{name} ranges overlap or leave gaps");
        cursor = end;
    }
    ensure!(cursor == expected, "{name} buffer has unassigned entries");
    Ok(())
}
