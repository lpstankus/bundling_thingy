//! Scoring of one frame's contact pairs: host scheduling of the assessment
//! chunks and reductions. The host reads the final counts and scalars only.
use super::{
    Config, Core, Error, Estimate, Originals, Protocol, RANK_SCRATCH_INDEX, Result, View,
    distance::{Assessor, PAIRS},
    ranked_words,
    runner::{Bound, P, STATUS_WORDS, add, mul, spans, word},
    status_word,
};

/// Words of one score or evidence record: both ordinals, the general flag and
/// the low and high squared excess.
const RECORD_WORDS: u64 = 5;
/// Columns of one reduction row: current, introduced and resolved low/high.
const ROW_WORDS: u64 = 6;
const REDUCE: u64 = 256;
/// Six sums, low/high RMS and low/high ratio.
const RESULT_WORDS: u64 = 10;
const S_A: usize = 3;
const S_B: usize = 4;
const S_C: usize = 5;

/// Frozen baseline evidence on the device: the baseline frame's incompatible
/// contact records in increasing pair order.
#[derive(Default)]
pub(super) struct Evidence {
    records: Option<wgpu::Buffer>,
    record_count: u64,
}

pub(super) struct Scored {
    pub general: View,
    pub assessed_pairs: u64,
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
    /// Reuse frozen baseline evidence, then assess every other pair of distinct
    /// representatives in bounded chunks; returns the assessed count.
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
            "score_init",
            &binds,
            P {
                e: word(evidence.map_or(0, |(_, count)| count))?,
                ..self.count
            },
            add(self.contacts, 1)?,
        )?;
        let needed = core.total(&self.marks, self.contacts, &self.scratch)?;
        for (first, chunk) in spans(0, needed, PAIRS) {
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
            core.run("score_pairs", &binds, registers, self.contacts)?;
            assessor.assess(core, originals, chunk, tolerances)?;
            core.run("score_collect", &binds, registers, self.contacts)?;
            core.runner.status("pair assessments")?;
            let assessed = add(first, chunk)?;
            tracing::debug!(assessed, needed, "ambiguity pair assessments");
        }
        Ok(needed)
    }

    /// Build the N contact rows and E evidence rows and reduce them; returns
    /// the row buffer, the final row and the status counts.
    fn reduce(
        &self,
        core: &mut Core,
        evidence: Option<(&wgpu::Buffer, u64)>,
        baseline: bool,
    ) -> Result<(wgpu::Buffer, u64, [u32; STATUS_WORDS])> {
        let evidence_count = evidence.map_or(0, |(_, count)| count);
        let rows = add(self.contacts, evidence_count)?;
        let row_space = add(add(rows, rows.div_ceil(REDUCE.saturating_sub(1)))?, 4)?;
        let row_buffer = core.words("ambiguity f32 reduction rows", mul(row_space, ROW_WORDS)?)?;
        let mut binds = vec![(1, self.log), (4, &self.scores), (5, &row_buffer)];
        binds.extend(evidence.map(|(buffer, _)| (2, buffer)));
        core.runner.reset();
        core.run(
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
}

impl Evidence {
    /// Keep the baseline frame's incompatible records for the session.
    fn freeze(&mut self, core: &mut Core, records: &Records<'_>) -> Result<()> {
        core.run(
            "record_marks",
            &[(4, &records.scores), (5, &records.marks)],
            records.count,
            add(records.contacts, 1)?,
        )?;
        self.record_count = core.total(&records.marks, records.contacts, &records.scratch)?;
        if self.record_count == 0 {
            return Ok(());
        }
        let buffer = core.words(
            "ambiguity f32 baseline evidence",
            mul(self.record_count, RECORD_WORDS)?,
        )?;
        core.store.retain(&buffer);
        core.run(
            "evidence_write",
            &[(4, &records.scores), (5, &records.marks), (6, &buffer)],
            records.count,
            records.contacts,
        )?;
        self.records = Some(buffer);
        Ok(())
    }
}

/// Decode the finished scalars and counts into the general view.
fn view(scalars: &[u32], counts: &[u32]) -> Result<View> {
    let scalar = |index: usize| {
        let value = scalars
            .get(index)
            .copied()
            .map(f32::from_bits)
            .ok_or(Error::Protocol(Protocol::Readback))?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::Range("a nonfinite f32 score"))
        }
    };
    let estimate = |low: usize| -> Result<Estimate> {
        Ok(Estimate {
            low: scalar(low)?,
            high: scalar(low.saturating_add(1))?,
        })
    };
    Ok(View {
        current: status_word(counts, S_A),
        introduced: status_word(counts, S_B),
        resolved: status_word(counts, S_C),
        squared_excess: estimate(0)?,
        introduced_squared_excess: estimate(2)?,
        resolved_squared_excess: estimate(4)?,
        rms_excess: estimate(6)?,
        ratio: estimate(8)?,
    })
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
        let records = Records {
            log,
            scores: core.words(
                "ambiguity f32 score records",
                mul(contacts.max(1), RECORD_WORDS)?,
            )?,
            marks: core.words("ambiguity f32 score marks", add(contacts, 1)?)?,
            scratch: core.words(
                "ambiguity f32 score scan sums",
                ranked_words(contacts)?[RANK_SCRATCH_INDEX],
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
        if baseline {
            evidence.freeze(core, &records)?;
        }
        let result = core.words("ambiguity f32 scores", RESULT_WORDS)?;
        core.run_once(
            "score_finish",
            &[(5, &rows), (6, &result)],
            P {
                a: word(last)?,
                e: word(evidence_count)?,
                ..records.count
            },
            Bound::Fixed,
        )?;
        let scalars = core.runner.read(&result, 0, RESULT_WORDS)?;
        Ok(Scored {
            general: view(&scalars, &counts)?,
            assessed_pairs,
        })
    })
}
