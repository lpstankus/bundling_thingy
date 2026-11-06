//! Canonical original routes and scoped, lazily retained distance preparations.
use crate::{
    ambiguity_similarity::{self, Config},
    weak_frechet::{Budget, Error},
};
use std::{cmp::Ordering, mem::size_of};

pub struct Route<'a> {
    pub edge_id: u32,
    pub points: &'a [[f32; 2]],
}
#[derive(Clone, Copy)]
pub(crate) struct Canonical {
    pub(crate) id: u32,
    pub(crate) start: usize,
    pub(crate) end: usize,
}
#[derive(Clone, Copy)]
pub struct Originals<'a> {
    pub(crate) points: &'a [[f32; 2]],
    pub(crate) routes: &'a [Canonical],
    pub closed_route_count: usize,
    pub unique_route_count: usize,
}
impl<'a> Originals<'a> {
    #[must_use]
    pub fn canonical_points(&self, index: usize) -> Option<&'a [[f32; 2]]> {
        let route = self.routes.get(index)?;
        self.points.get(route.start..route.end)
    }
    #[must_use]
    pub fn edge_id(&self, index: usize) -> Option<u32> {
        self.routes.get(index).map(|route| route.id)
    }
}
/// Lend geometry-ordered canonical originals and preparations under one ledger.
/// # Errors
/// Invalid geometry, duplicate IDs, configuration or exhausted resources abort.
pub fn with_originals<R>(
    input: &[Route<'_>],
    config: &Config,
    budget: &mut Budget,
    operation: impl for<'a> FnOnce(
        Originals<'a>,
        &mut [Option<ambiguity_similarity::Prepared<'a>>],
        &mut Budget,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    validate_config(config)?;
    let mut count = 0usize;
    for route in input {
        budget.charge(1)?;
        count = count
            .checked_add(route.points.len())
            .ok_or(Error::SizeOverflow)?;
    }
    let bytes = storage::<[f32; 2]>(count)?
        .checked_add(storage::<Canonical>(input.len())?)
        .and_then(|n| {
            n.checked_add(storage::<Option<ambiguity_similarity::Prepared<'_>>>(input.len()).ok()?)
        })
        .ok_or(Error::SizeOverflow)?;
    budget.reserve(bytes)?;
    let result = (|| {
        let mut points = allocated(count)?;
        let mut routes = allocated(input.len())?;
        let closed_route_count = canonicalize(input, &mut points, &mut routes, budget)?;
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
        let mut unique_route_count = 0usize;
        for (i, route) in routes.iter().enumerate() {
            budget.charge(1)?;
            let curve = points
                .get(route.start..route.end)
                .ok_or(Error::InvalidIndex)?;
            budget.charge(u64::try_from(curve.len()).map_err(|_| Error::SizeOverflow)?)?;
            let duplicate = i
                .checked_sub(1)
                .and_then(|j| routes.get(j))
                .is_some_and(|previous| points.get(previous.start..previous.end) == Some(curve));
            if !duplicate {
                unique_route_count = unique_route_count
                    .checked_add(1)
                    .ok_or(Error::SizeOverflow)?;
            }
        }
        let mut preparations = allocated(input.len())?;
        budget.charge(u64::try_from(input.len()).map_err(|_| Error::SizeOverflow)?)?;
        preparations.resize_with(input.len(), || None);
        let result = operation(
            Originals {
                points: &points,
                routes: &routes,
                closed_route_count,
                unique_route_count,
            },
            &mut preparations,
            budget,
        );
        for prepared in preparations.into_iter().flatten() {
            prepared.release(budget)?;
        }
        result
    })();
    budget.release(bytes)?;
    result
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
        });
    }

    Ok(closed)
}
fn validate_config(config: &Config) -> Result<(), Error> {
    if !config.route_tolerance.is_finite()
        || config.route_tolerance <= 0.0
        || !config.endpoint_tolerance.is_finite()
        || config.endpoint_tolerance <= 0.0
        || !config.sigma.is_finite()
        || config.sigma < 0.0
    {
        return Err(Error::InvalidCurve);
    }
    Ok(())
}
