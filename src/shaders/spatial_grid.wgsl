struct GridParameters {
    size: vec2<u32>,
    radius: u32,
    dispatch_limit: u32,
}

struct Cell {
    start: u32,
    end: atomic<u32>,
}

struct CellSummary {
    relative_sum: vec2<f32>,
    minimum: vec2<f32>,
    maximum: vec2<f32>,
}

struct Grid {
    cursor: atomic<u32>,
    task_count: atomic<u32>,
    cells: array<Cell>,
}

struct IndexedPoint {
    x: f32,
    y: f32,
    index: u32,
}

struct Task {
    cell: u32,
    start: u32,
}

struct Dispatch {
    x: u32,
    y: u32,
    z: u32,
}

struct Dispatches {
    summaries: Dispatch,
    tiled: Dispatch,
    sparse: Dispatch,
}

@group(0) @binding(2) var<uniform> grid: GridParameters;
@group(0) @binding(5) var<storage, read_write> spatial: Grid;
@group(0) @binding(6) var<storage, read_write> positions: array<IndexedPoint>;
@group(0) @binding(7) var<storage, read_write> tasks: array<Task>;
@group(0) @binding(8) var<storage, read_write> summaries: array<CellSummary>;
@group(1) @binding(0) var<storage, read_write> dispatch: Dispatches;

fn indexed_position(index: u32) -> vec2<f32> {
    let point = positions[index];
    return vec2<f32>(point.x, point.y);
}

fn texel_position(position: vec2<f32>) -> vec2<f32> {
    return (position + vec2<f32>(1.0)) * 0.5 * vec2<f32>(map.size - vec2<u32>(1u));
}

fn grid_cell(texels: vec2<f32>) -> vec2<u32> {
    return min(vec2<u32>(max(texels / f32(grid.radius), vec2<f32>(0.0))), grid.size - vec2<u32>(1u));
}

fn cell_index(position: vec2<f32>) -> u32 {
    let cell = grid_cell(texel_position(position));
    return cell.y * grid.size.x + cell.x;
}

@compute @workgroup_size(block_size)
fn count_cells(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index < arrayLength(&input_points) {
        atomicAdd(&spatial.cells[cell_index(point_position(input_points[index]))].end, 1u);
    }
}

var<workgroup> prefix: array<u32, block_size>;
var<workgroup> block_start: u32;

@compute @workgroup_size(block_size)
fn reserve_cells(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    var count = 0u;
    if index < arrayLength(&spatial.cells) {
        count = atomicLoad(&spatial.cells[index].end);
    }
    prefix[lane] = count;
    workgroupBarrier();
    for (var stride = 1u; stride < block_size; stride *= 2u) {
        var previous = 0u;
        if lane >= stride {
            previous = prefix[lane - stride];
        }
        workgroupBarrier();
        prefix[lane] += previous;
        workgroupBarrier();
    }
    // Blocks reserve disjoint ranges, so no global prefix scan or cell capacity limit is needed.
    if lane == 0u {
        block_start = atomicAdd(&spatial.cursor, prefix[block_size - 1u]);
    }
    workgroupBarrier();
    if index < arrayLength(&spatial.cells) {
        let start = block_start + prefix[lane] - count;
        spatial.cells[index].start = start;
        atomicStore(&spatial.cells[index].end, start);
        let task_count = (count + block_size - 1u) / block_size;
        if task_count > 0u {
            let task_start = atomicAdd(&spatial.task_count, task_count);
            for (var task = 0u; task < task_count; task += 1u) {
                tasks[task_start + task] = Task(index, start + task * block_size);
            }
        }
    }
}

@compute @workgroup_size(block_size)
fn scatter_points(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = id.y * groups.x * block_size + id.x;
    if index < arrayLength(&input_points) {
        let position = point_position(input_points[index]);
        let slot = atomicAdd(&spatial.cells[cell_index(position)].end, 1u);
        positions[slot] = IndexedPoint(position.x, position.y, index);
    }
}

fn dispatch_size(count: u32) -> Dispatch {
    let y = (count + grid.dispatch_limit - 1u) / grid.dispatch_limit;
    return Dispatch((count + y - 1u) / y, y, 1u);
}

@compute @workgroup_size(1)
fn prepare_dispatch() {
    let count = atomicLoad(&spatial.task_count);
    let samples = atomicLoad(&spatial.cursor);
    var summaries = dispatch_size(count);
    var tiled = summaries;
    var sparse = Dispatch(0u, 1u, 1u);
    // Sparse cells leave most lanes idle and do not amortize workgroup barriers.
    if samples / count < 8u {
        summaries = Dispatch(0u, 1u, 1u);
        tiled = Dispatch(0u, 1u, 1u);
        sparse = dispatch_size((samples + block_size - 1u) / block_size);
    }
    dispatch = Dispatches(summaries, tiled, sparse);
}

var<workgroup> current_task: Task;
var<workgroup> cell_range: vec2<u32>;
var<workgroup> sums: array<vec2<f32>, block_size>;
var<workgroup> minima: array<vec2<f32>, block_size>;
var<workgroup> maxima: array<vec2<f32>, block_size>;

fn load_task(index: u32, lane: u32) -> Task {
    if lane == 0u {
        current_task = Task(arrayLength(&spatial.cells), 0u);
        if index < atomicLoad(&spatial.task_count) {
            current_task = tasks[index];
        }
    }
    return workgroupUniformLoad(&current_task);
}

@compute @workgroup_size(block_size)
fn summarize_cells(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let task = load_task(group.y * groups.x + group.x, lane);
    if task.cell >= arrayLength(&spatial.cells) {
        return;
    }
    if lane == 0u {
        cell_range = vec2<u32>(spatial.cells[task.cell].start, atomicLoad(&spatial.cells[task.cell].end));
    }
    let range = workgroupUniformLoad(&cell_range);
    // Only the first query task for a cell computes its summary.
    if task.start != range.x {
        return;
    }
    let anchor = indexed_position(range.x);
    var sum = vec2<f32>(0.0);
    var minimum = anchor;
    var maximum = anchor;
    for (var index = range.x + lane; index < range.y; index += block_size) {
        let position = indexed_position(index);
        sum += position - anchor;
        minimum = min(minimum, position);
        maximum = max(maximum, position);
    }
    sums[lane] = sum;
    minima[lane] = minimum;
    maxima[lane] = maximum;
    workgroupBarrier();
    for (var stride = block_size / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            sums[lane] += sums[lane + stride];
            minima[lane] = min(minima[lane], minima[lane + stride]);
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        }
        workgroupBarrier();
    }
    if lane == 0u {
        summaries[task.cell] = CellSummary(sums[0], minima[0], maxima[0]);
    }
}
