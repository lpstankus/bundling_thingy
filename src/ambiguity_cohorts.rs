//! Fixed, threshold-dependent first-fit complete-link cohorts of whole original routes.
//! Different cohort IDs do not imply pair incompatibility.
use crate::{
    ambiguity_similarity::{self, Config},
    weak_frechet::{Budget, Error},
};
use std::{cmp::Ordering, mem::size_of};

/// An eligible original route. IDs must be unique; constants are eligible.
pub struct Route<'a> {
    pub edge_id: u32,
    pub points: &'a [[f32; 2]],
}
/// Frozen membership, ordered by canonical geometry and then stable ID.
#[derive(serde::Serialize)]
pub struct Membership {
    pub edge_id: u32,
    pub cohort: usize,
}
/// Borrowed evidence valid only during the inference callback.
pub struct Cohorts<'a> {
    pub memberships: &'a [Membership],
    pub cohort_count: usize,
    pub unique_route_count: usize,
    pub closed_route_count: usize,
    points: &'a [[f32; 2]],
    routes: &'a [Canonical],
}
impl Cohorts<'_> {
    /// Borrow the exact geometry used for the membership at this index.
    /// The index follows `memberships`, including duplicate routes.
    #[must_use]
    pub fn canonical_points(&self, index: usize) -> Option<&[[f32; 2]]> {
        let route = self.routes.get(index)?;
        self.points.get(route.start..route.end)
    }
}
struct Canonical {
    id: u32,
    start: usize,
    end: usize,
    cohort: usize,
    representative: bool,
    next: Option<usize>,
}
fn compare(a: &[[f32; 2]], b: &[[f32; 2]]) -> Ordering {
    for (p, q) in a.iter().zip(b) {
        let order = p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1]));
        if !order.is_eq() {
            return order;
        }
    }
    a.len().cmp(&b.len())
}
// Grow an exact floating expansion. Each coordinate product is exact in f64.
// TwoSum preserves the exact sum, including cancellation across exponent gaps.
fn collinear(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Result<bool, Error> {
    let [ax, ay] = a.map(f64::from);
    let [bx, by] = b.map(f64::from);
    let [cx, cy] = c.map(f64::from);
    let mut parts = [0.0; 6];
    let mut len = 0;
    for mut value in [ax * by, bx * cy, cx * ay, -ay * bx, -by * cx, -cy * ax] {
        let mut next = [0.0; 6];
        let mut used = 0usize;
        for &part in parts.get(..len).ok_or(Error::InvalidIndex)? {
            let sum = value + part;
            let virtual_part = sum - value;
            let error = (value - (sum - virtual_part)) + (part - virtual_part);
            if error != 0.0 {
                *next.get_mut(used).ok_or(Error::InvalidIndex)? = error;
                used = used.checked_add(1).ok_or(Error::SizeOverflow)?;
            }
            value = sum;
        }
        if value != 0.0 {
            *next.get_mut(used).ok_or(Error::InvalidIndex)? = value;
            used = used.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
        parts = next;
        len = used;
    }
    Ok(len == 0)
}
fn redundant(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Result<bool, Error> {
    let between = a
        .into_iter()
        .zip(b)
        .zip(c)
        .all(|((a, b), c)| b >= a.min(c) && b <= a.max(c));
    Ok(between && collinear(a, b, c)?)
}
fn storage<T>(count: usize) -> Result<usize, Error> {
    count.checked_mul(size_of::<T>()).ok_or(Error::SizeOverflow)
}
fn allocated<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}
/// Canonicalizes signed zero, duplicate vertices and exact forward-collinear
/// subdivisions.
///
/// Retains retracing and stored closed-route starts. Reversal
/// chooses the lexicographically smaller sequence; closed routes are not rotated.
///
/// Identical canonical traces share inference but retain every original ID.
/// Candidates are examined in cohort creation order and every unique member
/// must pass the shared coherent route and endpoint predicate.
///
/// All storage is reserved before allocation and released after the callback,
/// including failures. Work remains charged. Sort comparisons are bounded by
/// stored record counts and are separate from counted visits.
/// Unique cohort members form creation-order linked chains, with the
/// representative first. Endpoint candidates use append, sort and dedup storage
/// bounded by twice the input route count. Both matched endpoints must pass
/// the kernel's f64 squared-distance and square-root predicate under at least
/// one orientation before any whole-route assessment.
/// # Errors
/// Returns kernel errors for invalid configuration, geometry, duplicate IDs,
/// exhausted budgets, allocation failure or size overflow.
pub fn with_cohorts<R>(
    input: &[Route<'_>],
    config: &Config,
    budget: &mut Budget,
    operation: impl FnOnce(Cohorts<'_>, &mut Budget) -> Result<R, Error>,
) -> Result<R, Error> {
    if !config.route_tolerance.is_finite()
        || config.route_tolerance <= 0.0
        || !config.endpoint_tolerance.is_finite()
        || config.endpoint_tolerance <= 0.0
        || !config.sigma.is_finite()
        || config.sigma < 0.0
    {
        return Err(Error::InvalidCurve);
    }
    let mut count = 0usize;
    for route in input {
        budget.charge(1)?;
        count = count
            .checked_add(route.points.len())
            .ok_or(Error::SizeOverflow)?;
    }
    let bytes = storage::<[f32; 2]>(count)?
        .checked_add(storage::<Canonical>(input.len())?)
        .and_then(|n| n.checked_add(storage::<(f64, usize)>(input.len().checked_mul(2)?).ok()?))
        .and_then(|n| n.checked_add(storage::<usize>(input.len().checked_mul(4)?).ok()?))
        .and_then(|n| n.checked_add(storage::<Membership>(input.len()).ok()?))
        .ok_or(Error::SizeOverflow)?;
    budget.reserve(bytes)?;
    let result = (|| {
        let mut points = allocated::<[f32; 2]>(count)?;
        let mut routes = allocated::<Canonical>(input.len())?;
        let mut index =
            allocated::<(f64, usize)>(input.len().checked_mul(2).ok_or(Error::SizeOverflow)?)?;
        let mut candidates =
            allocated::<usize>(input.len().checked_mul(2).ok_or(Error::SizeOverflow)?)?;
        let mut heads = allocated::<usize>(input.len())?;
        let mut tails = allocated::<usize>(input.len())?;
        let mut memberships = allocated::<Membership>(input.len())?;
        let closed = canonicalize(input, &mut points, &mut routes, budget)?;
        routes.sort_unstable_by_key(|r| r.id);
        if routes
            .windows(2)
            .any(|r| r.first().map(|x| x.id) == r.last().map(|x| x.id))
        {
            return Err(Error::InvalidCurve);
        }
        routes.sort_unstable_by(|a, b| {
            compare(
                points.get(a.start..a.end).unwrap_or_default(),
                points.get(b.start..b.end).unwrap_or_default(),
            )
            .then(a.id.cmp(&b.id))
        });
        for (i, r) in routes.iter().enumerate() {
            budget.charge(2)?;
            index.push((
                f64::from(points.get(r.start).ok_or(Error::InvalidIndex)?[0]),
                i,
            ));
            index.push((
                f64::from(
                    points
                        .get(r.end.checked_sub(1).ok_or(Error::InvalidIndex)?)
                        .ok_or(Error::InvalidIndex)?[0],
                ),
                i,
            ));
        }
        index.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (cohorts, unique) = assign(
            &points,
            &mut routes,
            &index,
            &mut candidates,
            (&mut heads, &mut tails),
            config,
            budget,
        )?;
        for route in &routes {
            budget.charge(1)?;
            memberships.push(Membership {
                edge_id: route.id,
                cohort: route.cohort,
            });
        }
        operation(
            Cohorts {
                memberships: &memberships,
                cohort_count: cohorts,
                unique_route_count: unique,
                closed_route_count: closed,
                points: &points,
                routes: &routes,
            },
            budget,
        )
    })();
    budget.release(bytes)?;
    result
}

fn canonicalize(
    input: &[Route<'_>],
    points: &mut Vec<[f32; 2]>,
    routes: &mut Vec<Canonical>,
    budget: &mut Budget,
) -> Result<usize, Error> {
    let mut closed = 0usize;
    for route in input {
        let start = points.len();
        if route.points.is_empty() {
            return Err(Error::InvalidCurve);
        }
        for &point in route.points {
            budget.charge(1)?;
            if point.iter().any(|x| !x.is_finite()) {
                return Err(Error::InvalidCurve);
            }
            let point = point.map(|x| if x == 0.0 { 0.0 } else { x });
            if points.len() > start && points.last() == Some(&point) {
                continue;
            }
            while points.len() >= start.checked_add(2).ok_or(Error::SizeOverflow)? {
                budget.charge(1)?;
                let n = points.len();
                if !redundant(
                    *points
                        .get(n.checked_sub(2).ok_or(Error::InvalidIndex)?)
                        .ok_or(Error::InvalidIndex)?,
                    *points
                        .get(n.checked_sub(1).ok_or(Error::InvalidIndex)?)
                        .ok_or(Error::InvalidIndex)?,
                    point,
                )? {
                    break;
                }
                points.pop();
            }
            points.push(point);
        }
        budget.charge(
            u64::try_from(points.len().checked_sub(start).ok_or(Error::InvalidIndex)?)
                .map_err(|_| Error::SizeOverflow)?,
        )?;
        let curve = points.get_mut(start..).ok_or(Error::InvalidIndex)?;
        if curve.len() > 1 && curve.first() == curve.last() {
            closed = closed.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
        let reverse = curve
            .iter()
            .rev()
            .zip(curve.iter())
            .find_map(|(a, b)| {
                let cmp = compare(&[*a], &[*b]);
                (!cmp.is_eq()).then_some(cmp.is_lt())
            })
            .unwrap_or(false);
        if reverse {
            curve.reverse();
        }
        routes.push(Canonical {
            id: route.edge_id,
            start,
            end: points.len(),
            cohort: usize::MAX,
            representative: false,
            next: None,
        });
    }

    Ok(closed)
}

fn assign(
    points: &[[f32; 2]],
    routes: &mut [Canonical],
    index: &[(f64, usize)],
    candidates: &mut Vec<usize>,
    chains: (&mut Vec<usize>, &mut Vec<usize>),
    config: &Config,
    budget: &mut Budget,
) -> Result<(usize, usize), Error> {
    let (heads, tails) = chains;
    let mut cohorts = 0usize;
    let mut unique = 0usize;
    for i in 0..routes.len() {
        budget.charge(1)?;
        let curve = points
            .get(
                routes.get(i).ok_or(Error::InvalidIndex)?.start
                    ..routes.get(i).ok_or(Error::InvalidIndex)?.end,
            )
            .ok_or(Error::InvalidIndex)?;
        budget.charge(u64::try_from(curve.len()).map_err(|_| Error::SizeOverflow)?)?;
        if i > 0
            && compare(
                curve,
                points
                    .get(
                        routes
                            .get(i.checked_sub(1).ok_or(Error::InvalidIndex)?)
                            .ok_or(Error::InvalidIndex)?
                            .start
                            ..routes
                                .get(i.checked_sub(1).ok_or(Error::InvalidIndex)?)
                                .ok_or(Error::InvalidIndex)?
                                .end,
                    )
                    .ok_or(Error::InvalidIndex)?,
            )
            .is_eq()
        {
            routes.get_mut(i).ok_or(Error::InvalidIndex)?.cohort = routes
                .get(i.checked_sub(1).ok_or(Error::InvalidIndex)?)
                .ok_or(Error::InvalidIndex)?
                .cohort;
            continue;
        }
        unique = unique.checked_add(1).ok_or(Error::SizeOverflow)?;
        discover(curve, points, routes, index, candidates, config, budget)?;
        let mut chosen = None;
        for &candidate in candidates.iter() {
            budget.charge(1)?;
            let mut compatible = true;
            let mut cursor = Some(*heads.get(candidate).ok_or(Error::InvalidIndex)?);
            while let Some(member_index) = cursor {
                budget.charge(1)?;
                let member = routes.get(member_index).ok_or(Error::InvalidIndex)?;
                cursor = member.next;
                let other = points
                    .get(member.start..member.end)
                    .ok_or(Error::InvalidIndex)?;
                if !endpoints_compatible(curve, other, config.endpoint_tolerance)? {
                    compatible = false;
                    break;
                }
                budget.charge(1)?;
                let assessment = ambiguity_similarity::assess(curve, other, config, budget)?;
                if assessment.general_incompatible || assessment.endpoint_incompatible {
                    compatible = false;
                    break;
                }
            }
            if compatible {
                chosen = Some(candidate);
                break;
            }
        }
        routes.get_mut(i).ok_or(Error::InvalidIndex)?.cohort = chosen.unwrap_or(cohorts);
        if let Some(cohort) = chosen {
            let tail = *tails.get(cohort).ok_or(Error::InvalidIndex)?;
            routes.get_mut(tail).ok_or(Error::InvalidIndex)?.next = Some(i);
            *tails.get_mut(cohort).ok_or(Error::InvalidIndex)? = i;
        } else {
            heads.push(i);
            tails.push(i);
            routes.get_mut(i).ok_or(Error::InvalidIndex)?.representative = true;
            cohorts = cohorts.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
    }

    Ok((cohorts, unique))
}

// Identical squared-distance and sqrt arithmetic to weak_frechet endpoints.
// Either coherent endpoint orientation must pass before whole-route work.
fn endpoints_compatible(a: &[[f32; 2]], b: &[[f32; 2]], tolerance: f64) -> Result<bool, Error> {
    let first_a = *a.first().ok_or(Error::InvalidCurve)?;
    let last_a = *a.last().ok_or(Error::InvalidCurve)?;
    let first_b = *b.first().ok_or(Error::InvalidCurve)?;
    let last_b = *b.last().ok_or(Error::InvalidCurve)?;
    let squared = |[ax, ay]: [f32; 2], [bx, by]: [f32; 2]| {
        let dx = f64::from(ax) - f64::from(bx);
        let dy = f64::from(ay) - f64::from(by);
        dx.mul_add(dx, dy * dy)
    };
    Ok(squared(first_a, first_b)
        .max(squared(last_a, last_b))
        .sqrt()
        <= tolerance
        || squared(first_a, last_b)
            .max(squared(last_a, first_b))
            .sqrt()
            <= tolerance)
}

fn discover(
    curve: &[[f32; 2]],
    points: &[[f32; 2]],
    routes: &[Canonical],
    index: &[(f64, usize)],
    candidates: &mut Vec<usize>,
    config: &Config,
    budget: &mut Budget,
) -> Result<(), Error> {
    candidates.clear();
    let x = f64::from(curve.first().ok_or(Error::InvalidCurve)?[0]);
    // Doubling the threshold is deliberately conservative, including
    // subtraction rounding. The actual predicate remains unchanged.
    let radius = config.endpoint_tolerance * 2.0;
    budget.charge(u64::from(usize::BITS.saturating_sub(index.len().leading_zeros())) + 1)?;
    let lo = index.partition_point(|entry| entry.0 < x - radius);
    for &(coordinate, j) in index.get(lo..).ok_or(Error::InvalidIndex)? {
        budget.charge(1)?;
        if coordinate > x + radius {
            break;
        }
        if routes.get(j).ok_or(Error::InvalidIndex)?.representative
            && endpoints_compatible(
                curve,
                points
                    .get(
                        routes.get(j).ok_or(Error::InvalidIndex)?.start
                            ..routes.get(j).ok_or(Error::InvalidIndex)?.end,
                    )
                    .ok_or(Error::InvalidIndex)?,
                config.endpoint_tolerance,
            )?
        {
            candidates.push(routes.get(j).ok_or(Error::InvalidIndex)?.cohort);
        }
    }
    candidates.sort_unstable();
    candidates.dedup();

    Ok(())
}
