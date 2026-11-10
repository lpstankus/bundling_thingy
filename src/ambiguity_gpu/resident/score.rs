//! Scoring of one frame's contact pairs in the gpu-f32 backend: host scheduling
//! of the assessment chunks, cohort-pair sets and reductions. The host reads
//! the final counts and scalars only.
use super::super::{Error, Protocol, Result};
use super::{
    CohortScores, Config, Core, Fault, Interval, Originals, Stage, View,
    distance::{Assessor, PAIRS},
    runner::{Bound, P, add, bits, histogram_words, mul, scan_words, word},
    status_word,
};

const ROW_WORDS: u64 = 12;
const REDUCE: u64 = 256;
const RESULT_WORDS: u64 = 16;
const S_A: usize = 3;
const S_G: u32 = 9;
const S_H: u32 = 10;
const STAGE: Stage = Stage::Score;

/// Frozen baseline evidence on the device: the baseline frame's incompatible
/// contact records and witnessed cohort pairs, both in increasing order.
#[derive(Default)]
pub(super) struct Evidence {
    records: Option<wgpu::Buffer>,
    record_count: u64,
    witnesses: Option<wgpu::Buffer>,
    witness_count: u64,
}

pub(super) struct Scored {
    pub general: View,
    pub endpoints: View,
    pub cohorts: CohortScores,
    pub assessed_pairs: u64,
}

/// Witnessed and contacting cross-cohort pairs of the frame's records.
struct Witnesses {
    list: Option<wgpu::Buffer>,
    witnessed: u64,
    contacting: u64,
}

/// One frame's score records with their marks and scan scratch.
struct Records<'a> {
    log: &'a wgpu::Buffer,
    scores: wgpu::Buffer,
    marks: wgpu::Buffer,
    scratch: wgpu::Buffer,
    contacts: u64,
    /// Contact count and both tolerances.
    count: P,
}

impl Records<'_> {
    /// Mark records by `kind` (0 incompatible, 1 cross-cohort), scan, and
    /// return how many are marked.
    fn marked(&self, core: &mut Core, kind: u32) -> Result<u64> {
        core.run(
            STAGE,
            "record_marks",
            &[(4, &self.scores), (5, &self.marks)],
            P {
                a: kind,
                ..self.count
            },
            add(self.contacts, 1)?,
        )?;
        core.total(&self.marks, self.contacts, &self.scratch)
    }

    /// Reuse frozen baseline evidence, then assess every other cross-cohort
    /// pair in bounded chunks; returns the assessed count.
    fn assess(
        &self,
        core: &mut Core,
        originals: &Originals,
        assessor: &mut Assessor,
        (tolerances, evidence): ((f32, f32), Option<(&wgpu::Buffer, u64)>),
    ) -> Result<u64> {
        let mut binds = vec![
            (1, self.log),
            (3, &originals.routes),
            (4, &self.scores),
            (5, &self.marks),
        ];
        binds.extend(evidence.map(|(buffer, _)| (2, buffer)));
        core.run(
            STAGE,
            "score_init",
            &binds,
            P {
                e: word(evidence.map_or(0, |(_, count)| count))?,
                ..self.count
            },
            add(self.contacts, 1)?,
        )?;
        let needed = core.total(&self.marks, self.contacts, &self.scratch)?;
        let mut first = 0;
        while first < needed {
            let chunk = needed.saturating_sub(first).min(PAIRS);
            let registers = P {
                a: word(first)?,
                b: word(chunk)?,
                ..self.count
            };
            let pairs = assessor.pairs.clone();
            let binds = [
                (3, &originals.routes),
                (4, &self.scores),
                (5, &self.marks),
                (6, &pairs),
            ];
            core.run(STAGE, "score_pairs", &binds, registers, self.contacts)?;
            assessor.assess(core, originals, chunk, tolerances)?;
            core.run(STAGE, "score_collect", &binds, registers, self.contacts)?;
            core.runner.status("pair assessments")?;
            first = add(first, chunk)?;
            tracing::debug!(assessed = first, needed, "gpu-f32 pair assessments");
        }
        Ok(needed)
    }

    /// Build the reduction rows and reduce them; returns the row buffer, the
    /// final row and the status counts.
    fn reduce(
        &self,
        core: &mut Core,
        evidence: Option<(&wgpu::Buffer, u64)>,
        baseline: bool,
    ) -> Result<(wgpu::Buffer, u64, [u32; super::runner::STATUS_WORDS])> {
        let evidence_count = evidence.map_or(0, |(_, count)| count);
        let rows = add(self.contacts, evidence_count)?;
        let row_space = add(add(rows, rows.div_ceil(REDUCE.saturating_sub(1)))?, 4)?;
        let row_buffer = core
            .store
            .words("ambiguity f32 reduction rows", mul(row_space, ROW_WORDS)?)?;
        let mut binds = vec![(1, self.log), (4, &self.scores), (5, &row_buffer)];
        binds.extend(evidence.map(|(buffer, _)| (2, buffer)));
        core.runner.reset();
        core.run(
            STAGE,
            "score_rows",
            &binds,
            P {
                e: word(evidence_count)?,
                f: u32::from(baseline),
                ..self.count
            },
            rows,
        )?;
        let (mut start, mut level) = (0, rows);
        while level > 1 {
            let blocks = level.div_ceil(REDUCE);
            let next = add(start, level)?;
            core.run(
                STAGE,
                "reduce",
                &[(5, &row_buffer)],
                P {
                    n: word(blocks)?,
                    a: word(start)?,
                    b: word(level)?,
                    c: word(next)?,
                    ..P::default()
                },
                blocks,
            )?;
            start = next;
            level = blocks;
        }
        let counts = core.runner.status("score rows")?;
        Ok((row_buffer, start, counts))
    }

    /// Distinct contacting cross-cohort pairs and those with an incompatible
    /// contacting witness.
    fn witnesses(&self, core: &mut Core, originals: &Originals) -> Result<Witnesses> {
        let crossing = self.marked(core, 1)?;
        if crossing == 0 {
            return Ok(Witnesses {
                list: None,
                witnessed: 0,
                contacting: 0,
            });
        }
        let store = &mut core.store;
        let first = store.words("ambiguity f32 cohort pair records", mul(crossing, 3)?)?;
        let second = store.words("ambiguity f32 cohort pair records", mul(crossing, 3)?)?;
        let histogram = store.words(
            "ambiguity f32 cohort pair histogram",
            histogram_words(crossing)?,
        )?;
        let unique = store.words("ambiguity f32 cohort pair marks", add(crossing, 1)?)?;
        core.run(
            STAGE,
            "cross_keys",
            &[(4, &self.scores), (5, &self.marks), (6, &first)],
            self.count,
            self.contacts,
        )?;
        let cohort_bits = bits(add(originals.preparation.cohorts, 1)?);
        let (sorted, _) = core.sort(
            (&first, &second),
            (&histogram, &self.scratch),
            crossing,
            (cohort_bits, cohort_bits),
        )?;
        let crossing_n = P {
            n: word(crossing)?,
            ..P::default()
        };
        let padded = add(crossing, 1)?;
        let keyed = [(4, sorted), (5, &unique)];
        core.run(Stage::Originals, "key_marks", &keyed, crossing_n, padded)?;
        let contacting = core.total(&unique, crossing, &self.scratch)?;
        let flags = core
            .store
            .words("ambiguity f32 witness marks", add(contacting, 1)?)?;
        let marked = [(4, sorted), (5, &unique), (7, &flags)];
        core.run(STAGE, "witness_marks", &marked, crossing_n, crossing)?;
        let witnessed = core.total(&flags, contacting, &self.scratch)?;
        let list = if witnessed > 0 {
            let list = core
                .store
                .words("ambiguity f32 witnessed cohort pairs", mul(witnessed, 2)?)?;
            core.run(
                STAGE,
                "witness_list",
                &[(4, sorted), (5, &unique), (6, &list), (7, &flags)],
                crossing_n,
                crossing,
            )?;
            Some(list)
        } else {
            None
        };
        Ok(Witnesses {
            list,
            witnessed,
            contacting,
        })
    }
}

impl Evidence {
    /// Keep the baseline frame's incompatible records and witnessed cohort
    /// pairs for the session.
    fn freeze(
        &mut self,
        core: &mut Core,
        records: &Records<'_>,
        witnesses: &Witnesses,
    ) -> Result<()> {
        self.record_count = records.marked(core, 0)?;
        if self.record_count > 0 {
            let buffer = core.store.words(
                "ambiguity f32 baseline evidence",
                mul(self.record_count, 8)?,
            )?;
            core.store.retain(&buffer);
            core.run(
                STAGE,
                "evidence_write",
                &[(4, &records.scores), (5, &records.marks), (6, &buffer)],
                records.count,
                records.contacts,
            )?;
            self.records = Some(buffer);
        }
        self.witness_count = witnesses.witnessed;
        if let Some(list) = &witnesses.list {
            core.store.retain(list);
        }
        self.witnesses.clone_from(&witnesses.list);
        Ok(())
    }

    /// Count the cohort pairs of `current` absent from the baseline into
    /// status word `S_G` and the baseline's absent from `current` into `S_H`.
    fn differences(&self, core: &mut Core, current: &Witnesses) -> Result<()> {
        for (list, count, other, other_count, slot) in [
            (
                &current.list,
                current.witnessed,
                &self.witnesses,
                self.witness_count,
                S_G,
            ),
            (
                &self.witnesses,
                self.witness_count,
                &current.list,
                current.witnessed,
                S_H,
            ),
        ] {
            let Some(list) = list else { continue };
            let mut binds = vec![(4, list)];
            binds.extend(other.as_ref().map(|other| (5, other)));
            core.run(
                STAGE,
                "witness_difference",
                &binds,
                P {
                    n: word(count)?,
                    b: word(other_count)?,
                    c: slot,
                    ..P::default()
                },
                count,
            )?;
        }
        Ok(())
    }
}

/// Decode the finished scalars and counts into the two views.
fn views(scalars: &[u32], counts: &[u32]) -> Result<(View, View, f32)> {
    let scalar = |index: usize| {
        let value = scalars
            .get(index)
            .copied()
            .map(f32::from_bits)
            .ok_or(Error::Protocol(Protocol::Readback))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::Fault(Fault::Range("a nonfinite f32 score")))
        }
    };
    let view = |lower: usize, upper: usize, word: usize| -> Result<View> {
        let interval = |offset: usize| -> Result<Interval> {
            Ok(Interval {
                lower: scalar(offset.saturating_add(lower))?,
                upper: scalar(offset.saturating_add(upper))?,
            })
        };
        Ok(View {
            current: status_word(counts, S_A.saturating_add(word)),
            introduced: status_word(counts, S_A.saturating_add(2).saturating_add(word)),
            resolved: status_word(counts, S_A.saturating_add(4).saturating_add(word)),
            squared_excess: interval(0)?,
            introduced_squared_excess: interval(3)?,
            resolved_squared_excess: interval(6)?,
            rms_excess: interval(9)?,
            ratio: interval(12)?,
        })
    };
    Ok((view(0, 1, 0)?, view(2, 2, 1)?, scalar(15)?))
}

/// Score the `contacts` logged pairs. The baseline frame freezes its evidence.
pub(super) fn score(
    core: &mut Core,
    originals: &Originals,
    assessor: &mut Assessor,
    evidence: &mut Evidence,
    (log, contacts): (&wgpu::Buffer, u64),
    config: &Config,
    baseline: bool,
) -> Result<Scored> {
    core.scoped(|core| {
        let tolerances = (config.route_tolerance, config.endpoint_tolerance);
        let store = &mut core.store;
        let records = Records {
            log,
            scores: store.words("ambiguity f32 score records", mul(contacts.max(1), 8)?)?,
            marks: store.words("ambiguity f32 score marks", add(contacts, 1)?)?,
            scratch: store.words(
                "ambiguity f32 score scan sums",
                scan_words(add(contacts, 1)?.max(histogram_words(contacts)?))?,
            )?,
            contacts,
            count: P {
                n: word(contacts)?,
                x: tolerances.0,
                y: tolerances.1,
                ..P::default()
            },
        };
        let frozen = evidence
            .records
            .as_ref()
            .filter(|_| !baseline)
            .map(|buffer| (buffer, evidence.record_count));
        let assessed_pairs = records.assess(core, originals, assessor, (tolerances, frozen))?;
        let evidence_count = frozen.map_or(0, |(_, count)| count);
        let (rows, last, counts) = records.reduce(core, frozen, baseline)?;
        let current = records.witnesses(core, originals)?;
        core.runner.reset();
        if baseline {
            evidence.freeze(core, &records, &current)?;
        } else {
            evidence.differences(core, &current)?;
        }
        let result = core.store.words("ambiguity f32 scores", RESULT_WORDS)?;
        core.run_once(
            STAGE,
            "score_finish",
            &[(5, &rows), (6, &result)],
            P {
                a: word(last)?,
                b: word(current.witnessed)?,
                c: word(current.contacting)?,
                e: word(evidence_count)?,
                ..records.count
            },
            Bound::Fixed,
        )?;
        let differences = core.runner.status("cohort associations")?;
        let scalars = core.runner.read(&result, 0, RESULT_WORDS)?;
        let (general, endpoints, incompatible_fraction) = views(&scalars, &counts)?;
        let difference = |slot: u32| {
            if baseline {
                0
            } else {
                status_word(&differences, usize::try_from(slot).unwrap_or(0))
            }
        };
        Ok(Scored {
            general,
            endpoints,
            cohorts: CohortScores {
                current: current.witnessed,
                baseline: evidence.witness_count,
                introduced: difference(S_G),
                resolved: difference(S_H),
                contacting_cross_cohort: current.contacting,
                incompatible_fraction,
            },
            assessed_pairs,
        })
    })
}
