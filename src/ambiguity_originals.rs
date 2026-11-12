//! Canonical original routes and their lazily retained distance preparations.
mod similarity;

use crate::{
    geometry::Snapshot,
    weak_frechet::{Budget, Error},
};
use similarity::Prepared;
pub use similarity::{Assessment, Config};
use std::{cmp::Ordering, mem::size_of};

#[derive(Clone, Copy)]
struct Canonical {
    id: u32,
    start: usize,
    end: usize,
}
/// Geometry-ordered canonical originals owning their certified preparations.
///
/// All storage is persistent on the evaluator's ledger. Preparations are made on
/// demand with one sigma and remain until this owner drops.
pub struct Originals {
    points: Vec<[f32; 2]>,
    routes: Vec<Canonical>,
    preparations: Vec<Option<Prepared>>,
    closed_route_count: usize,
    unique_route_count: usize,
    constant_route_count: usize,
}
impl Originals {
    /// Canonicalize every snapshot route. Snapshot IDs are unique and its
    /// samples are finite, with at least two per route.
    /// # Errors
    /// Exhausted resources or size overflow.
    pub fn new(snapshot: &Snapshot<'_>, budget: &mut Budget) -> Result<Self, Error> {
        let mut count = 0usize;
        for (_, points) in snapshot.edges() {
            budget.charge(1)?;
            count = count.checked_add(points.len()).ok_or(Error::SizeOverflow)?;
        }
        let routes = snapshot.layout().edge_count();
        let bytes = storage::<[f32; 2]>(count)?
            .checked_add(storage::<Canonical>(routes)?)
            .and_then(|n| n.checked_add(storage::<Option<Prepared>>(routes).ok()?))
            .ok_or(Error::SizeOverflow)?;
        budget.reserve(bytes)?;
        let mut points = allocated(count)?;
        let mut routes = allocated(routes)?;
        let closed_route_count = canonicalize(snapshot, &mut points, &mut routes, budget)?;
        routes.sort_unstable_by(|a, b| {
            compare(
                points.get(a.start..a.end).unwrap_or_default(),
                points.get(b.start..b.end).unwrap_or_default(),
            )
            .then(a.id.cmp(&b.id))
        });
        let mut unique_route_count = 0usize;
        let mut constant_route_count = 0usize;
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
            if curve.len() == 1 {
                constant_route_count = constant_route_count
                    .checked_add(1)
                    .ok_or(Error::SizeOverflow)?;
            }
        }
        let mut preparations = allocated(routes.len())?;
        budget.charge(u64::try_from(routes.len()).map_err(|_| Error::SizeOverflow)?)?;
        preparations.resize_with(routes.len(), || None);
        Ok(Self {
            points,
            routes,
            preparations,
            closed_route_count,
            unique_route_count,
            constant_route_count,
        })
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.routes.len()
    }
    #[must_use]
    pub fn edge_id(&self, index: usize) -> Option<u32> {
        self.routes.get(index).map(|route| route.id)
    }
    #[must_use]
    pub const fn closed_route_count(&self) -> usize {
        self.closed_route_count
    }
    #[must_use]
    pub const fn unique_route_count(&self) -> usize {
        self.unique_route_count
    }
    #[must_use]
    pub const fn constant_route_count(&self) -> usize {
        self.constant_route_count
    }
    fn canonical_points(&self, index: usize) -> Result<&[[f32; 2]], Error> {
        let route = self.routes.get(index).ok_or(Error::InvalidIndex)?;
        self.points
            .get(route.start..route.end)
            .ok_or(Error::InvalidIndex)
    }
    /// Assess canonical routes `a` and `b`, passing the later canonical route first.
    ///
    /// Missing preparations of `a`, then `b`, are retained on the ledger even if
    /// a later step fails. `config` must be the configuration of every assessment.
    /// # Errors
    /// Invalid indices, size overflow and exhausted shared resources.
    pub fn assess(
        &mut self,
        a: usize,
        b: usize,
        config: &Config,
        budget: &mut Budget,
    ) -> Result<Assessment, Error> {
        for index in [a, b] {
            budget.charge(1)?;
            if self
                .preparations
                .get(index)
                .ok_or(Error::InvalidIndex)?
                .is_none()
            {
                let prepared =
                    similarity::prepare(self.canonical_points(index)?, config.sigma, budget)?;
                *self
                    .preparations
                    .get_mut(index)
                    .ok_or(Error::InvalidIndex)? = Some(prepared);
            }
        }
        let route = |index: usize| -> Result<(&[[f32; 2]], &Prepared), Error> {
            Ok((
                self.canonical_points(index)?,
                self.preparations
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or(Error::InvalidIndex)?,
            ))
        };
        similarity::assess(route(a.max(b))?, route(a.min(b))?, config, budget)
    }
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
    snapshot: &Snapshot<'_>,
    points: &mut Vec<[f32; 2]>,
    routes: &mut Vec<Canonical>,
    budget: &mut Budget,
) -> Result<usize, Error> {
    let mut closed = 0usize;
    for (id, route) in snapshot.edges() {
        let start = points.len();
        for &point in route {
            budget.charge(1)?;
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
            id,
            start,
            end: points.len(),
        });
    }

    Ok(closed)
}
