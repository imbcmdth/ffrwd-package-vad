//! The arithmetic the module is built on: per-chunk probabilities turned into
//! spans of speech, and those spans into one row per chunk, each written once
//! it is sure. None of it touches the model, so all of it is tested on the
//! host.

/// Samples one model chunk covers. Silero VAD v5 is trained on exactly this
/// many at 16 kHz and accepts no other length.
pub const CHUNK: usize = 512;

/// The rate the model works at, and the only one the module accepts.
pub const SAMPLE_RATE: u32 = 16_000;

/// Seconds one chunk covers: 32 ms, which is the resolution a span's edges
/// can ever have.
pub const CHUNK_SECONDS: f64 = CHUNK as f64 / SAMPLE_RATE as f64;

/// One span of speech, in seconds from the start of the stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start_t: f64,
    pub end_t: f64,
}

impl Span {
    pub fn duration(&self) -> f64 {
        self.end_t - self.start_t
    }
}

/// A span being built: where it began and where its last voiced chunk ended.
struct Open {
    start: f64,
    voiced_to: f64,
}

/// Chunk probabilities in, spans out.
///
/// A chunk at or above `threshold` is speech. A span opens on the first such
/// chunk and closes once `min_silence` seconds have passed since its last
/// one, so a quiet moment mid-sentence carries the span across rather than
/// splitting it. A span whose voiced extent is shorter than `min_speech` is
/// dropped, which is what keeps a stray loud chunk out of the rows.
pub struct Spans {
    threshold: f64,
    min_speech: f64,
    min_silence: f64,
    open: Option<Open>,
}

impl Spans {
    pub fn new(threshold: f64, min_speech: f64, min_silence: f64) -> Spans {
        Spans {
            threshold,
            min_speech,
            min_silence,
            open: None,
        }
    }

    /// Moves the thresholds without disturbing the span being built, which
    /// goes on under whatever is in force when it closes.
    pub fn retune(&mut self, threshold: f64, min_speech: f64, min_silence: f64) {
        self.threshold = threshold;
        self.min_speech = min_speech;
        self.min_silence = min_silence;
    }

    /// One chunk's probability and the second it starts at, oldest first.
    /// Answers with a span when this chunk is what closed one.
    pub fn push(&mut self, probability: f64, start_t: f64) -> Option<Span> {
        let end_t = start_t + CHUNK_SECONDS;
        if probability >= self.threshold {
            match &mut self.open {
                Some(open) => open.voiced_to = end_t,
                None => {
                    self.open = Some(Open {
                        start: start_t,
                        voiced_to: end_t,
                    })
                }
            }
            return None;
        }
        let open = self.open.as_ref()?;
        if end_t - open.voiced_to < self.min_silence {
            return None;
        }
        self.close()
    }

    /// The end of the stream closes whatever is still open, at its last
    /// voiced chunk rather than at the stream's end.
    pub fn finish(&mut self) -> Option<Span> {
        self.close()
    }

    /// Whether the open span is long enough to be kept, which it stays.
    fn sure(&self) -> bool {
        self.open
            .as_ref()
            .is_some_and(|open| open.voiced_to - open.start >= self.min_speech)
    }

    /// The open span, if it is long enough to be one.
    fn close(&mut self) -> Option<Span> {
        let open = self.open.take()?;
        let span = Span {
            start_t: open.start,
            end_t: open.voiced_to,
        };
        (span.duration() >= self.min_speech).then_some(span)
    }
}

/// One row of the per-chunk rows: the chunk it is stamped at, and the second
/// the span that chunk belongs to began.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Said {
    pub pts: i64,
    pub start_t: f64,
}

/// Spans written as they happen: a row for every chunk of a span, from its
/// first voiced chunk to its last, each carrying the second the span began.
///
/// A row is written once it is sure. The chunks of a span that may yet prove
/// shorter than `min_speech` wait until it is long enough, and a quiet chunk
/// inside one waits until speech resumes within `min_silence`. A span that is
/// dropped writes nothing, and nor does the quiet that closes one, so the rows
/// of a span run from its start to the end of its last voiced chunk, which is
/// the span [`Spans`] closes.
pub struct Speaking {
    spans: Spans,
    /// The chunks of the open span not yet written, oldest first.
    waiting: Vec<i64>,
}

impl Speaking {
    pub fn new(threshold: f64, min_speech: f64, min_silence: f64) -> Speaking {
        Speaking {
            spans: Spans::new(threshold, min_speech, min_silence),
            waiting: Vec::new(),
        }
    }

    /// Moves the threshold. The two durations stay: they decide how long a
    /// row may wait, which the node declared when it opened.
    pub fn retune(&mut self, threshold: f64) {
        let (min_speech, min_silence) = (self.spans.min_speech, self.spans.min_silence);
        self.spans.retune(threshold, min_speech, min_silence);
    }

    /// One chunk's probability, the pts it is stamped at and the second it
    /// starts at, oldest first. Answers the rows this chunk made sure.
    pub fn push(&mut self, probability: f64, pts: i64, start_t: f64) -> Vec<Said> {
        self.spans.push(probability, start_t);
        let Some(start_t) = self.spans.open.as_ref().map(|open| open.start) else {
            self.waiting.clear();
            return Vec::new();
        };
        self.waiting.push(pts);
        if probability < self.spans.threshold || !self.spans.sure() {
            return Vec::new();
        }
        self.waiting
            .drain(..)
            .map(|pts| Said { pts, start_t })
            .collect()
    }
}

/// The most chunks a row waits behind the chunk it is stamped at.
///
/// The longest wait is the first chunk of a span voiced for one chunk short of
/// `min_speech`, then quiet for one chunk short of closing, then voiced again:
/// only that last chunk says the span is kept. Each duration is counted up to
/// the next whole chunk and one past it when it falls on one, since a chunk
/// boundary compared in floating point can land either side.
pub fn latency_chunks(min_speech: f64, min_silence: f64) -> u32 {
    let chunks = |seconds: f64| (seconds / CHUNK_SECONDS + 1e-6).floor() as u32 + 1;
    chunks(min_speech) + chunks(min_silence) - 2
}

/// [`latency_chunks`] in seconds, which is what the node declares.
pub fn latency(min_speech: f64, min_silence: f64) -> f64 {
    f64::from(latency_chunks(min_speech, min_silence)) * CHUNK as f64 / f64::from(SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunk probabilities pushed in order from second zero, and the spans
    /// that came out including whatever the end of the stream closed.
    fn spans_of(spans: &mut Spans, probabilities: &[f64]) -> Vec<Span> {
        let mut found: Vec<Span> = probabilities
            .iter()
            .enumerate()
            .filter_map(|(index, p)| spans.push(*p, index as f64 * CHUNK_SECONDS))
            .collect();
        found.extend(spans.finish());
        found
    }

    /// `count` chunks at `probability`.
    fn run(count: usize, probability: f64) -> Vec<f64> {
        vec![probability; count]
    }

    #[test]
    fn a_run_of_speech_becomes_one_span_at_its_own_times() {
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        // Ten quiet chunks, twenty loud, ten quiet.
        let found = spans_of(
            &mut spans,
            &[run(10, 0.1), run(20, 0.9), run(10, 0.1)].concat(),
        );
        assert_eq!(found.len(), 1);
        assert!((found[0].start_t - 10.0 * CHUNK_SECONDS).abs() < 1e-12);
        assert!((found[0].end_t - 30.0 * CHUNK_SECONDS).abs() < 1e-12);
    }

    #[test]
    fn a_chunk_at_the_threshold_counts_as_speech() {
        let mut spans = Spans::new(0.5, 0.0, 0.1);
        let found = spans_of(&mut spans, &[run(10, 0.5), run(10, 0.0)].concat());
        assert_eq!(found.len(), 1, "0.5 is not below 0.5");

        let mut spans = Spans::new(0.5, 0.0, 0.1);
        let found = spans_of(&mut spans, &[run(10, 0.499), run(10, 0.0)].concat());
        assert!(found.is_empty(), "and 0.499 is");
    }

    #[test]
    fn a_gap_shorter_than_the_minimum_silence_does_not_split_a_span() {
        // 0.1 s of silence closes a span, and two chunks is 64 ms.
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(
            &mut spans,
            &[run(20, 0.9), run(2, 0.1), run(20, 0.9), run(10, 0.1)].concat(),
        );
        assert_eq!(found.len(), 1, "the sentence stayed whole");
        assert!((found[0].start_t - 0.0).abs() < 1e-12);
        assert!((found[0].end_t - 42.0 * CHUNK_SECONDS).abs() < 1e-12);
    }

    #[test]
    fn a_gap_at_the_minimum_silence_does_split_it() {
        // Four chunks is 128 ms, which is past 0.1 s.
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(
            &mut spans,
            &[run(20, 0.9), run(4, 0.1), run(20, 0.9), run(10, 0.1)].concat(),
        );
        assert_eq!(found.len(), 2);
        assert!(
            (found[0].end_t - 20.0 * CHUNK_SECONDS).abs() < 1e-12,
            "the first span ends at its last voiced chunk, not where the \
             silence was counted out"
        );
        assert!((found[1].start_t - 24.0 * CHUNK_SECONDS).abs() < 1e-12);
    }

    #[test]
    fn a_burst_shorter_than_the_minimum_speech_is_dropped() {
        // 0.25 s is just under eight chunks, so four is a burst.
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(
            &mut spans,
            &[run(4, 0.9), run(10, 0.1), run(20, 0.9), run(10, 0.1)].concat(),
        );
        assert_eq!(found.len(), 1, "only the sentence survived");
        assert!((found[0].start_t - 14.0 * CHUNK_SECONDS).abs() < 1e-12);
    }

    #[test]
    fn a_burst_the_gaps_add_up_to_survives() {
        // Four voiced chunks, a short gap, four more: neither run is long
        // enough alone, and the span they make is.
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(
            &mut spans,
            &[run(4, 0.9), run(2, 0.1), run(4, 0.9), run(10, 0.1)].concat(),
        );
        assert_eq!(found.len(), 1);
        assert!((found[0].duration() - 10.0 * CHUNK_SECONDS).abs() < 1e-12);
    }

    #[test]
    fn speech_running_to_the_end_of_the_stream_still_closes() {
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(&mut spans, &[run(10, 0.1), run(20, 0.9)].concat());
        assert_eq!(found.len(), 1);
        assert!(
            (found[0].end_t - 30.0 * CHUNK_SECONDS).abs() < 1e-12,
            "closed at its last voiced chunk"
        );
    }

    #[test]
    fn a_burst_at_the_end_of_the_stream_is_dropped_like_any_other() {
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        assert!(spans_of(&mut spans, &[run(10, 0.1), run(3, 0.9)].concat()).is_empty());
    }

    #[test]
    fn silence_alone_produces_nothing() {
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        assert!(spans_of(&mut spans, &run(100, 0.02)).is_empty());
    }

    #[test]
    fn the_spans_of_a_stream_never_overlap_and_stay_in_order() {
        let mut spans = Spans::new(0.5, 0.25, 0.1);
        let found = spans_of(
            &mut spans,
            &[
                run(20, 0.9),
                run(10, 0.1),
                run(20, 0.8),
                run(10, 0.0),
                run(20, 0.99),
            ]
            .concat(),
        );
        assert_eq!(found.len(), 3);
        for pair in found.windows(2) {
            assert!(pair[0].end_t <= pair[1].start_t, "{pair:?} overlap");
        }
    }

    #[test]
    fn the_thresholds_are_the_callers_to_move() {
        // The same probabilities, read strictly: nothing is speech.
        let probabilities = [run(20, 0.6), run(10, 0.1)].concat();
        let mut lenient = Spans::new(0.5, 0.25, 0.1);
        assert_eq!(spans_of(&mut lenient, &probabilities).len(), 1);
        let mut strict = Spans::new(0.9, 0.25, 0.1);
        assert!(spans_of(&mut strict, &probabilities).is_empty());
    }

    /// A sequence of probabilities that wanders in and out of speech, the
    /// same every run: a linear congruential generator, so no crate is
    /// needed for it.
    fn wandering(seed: u64, count: usize) -> Vec<f64> {
        let mut state = seed;
        let mut voiced = false;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let draw = (state >> 33) as f64 / f64::from(1u32 << 31);
                if draw < 0.2 {
                    voiced = !voiced;
                }
                if voiced {
                    0.5 + draw / 2.0
                } else {
                    draw / 2.0
                }
            })
            .collect()
    }

    /// The rows written for `probabilities`, chunk `n` stamped at pts `n`, as
    /// (row, the chunk whose push wrote it).
    fn rows_of(speaking: &mut Speaking, probabilities: &[f64]) -> Vec<(Said, usize)> {
        let mut written = Vec::new();
        for (index, p) in probabilities.iter().enumerate() {
            for said in speaking.push(*p, index as i64, index as f64 * CHUNK_SECONDS) {
                written.push((said, index));
            }
        }
        written
    }

    /// The rows merged the way the host's span reducer merges them: rows
    /// sharing a start are one span, ending at the end of its last chunk.
    fn merged(written: &[(Said, usize)]) -> Vec<Span> {
        let mut spans: Vec<Span> = Vec::new();
        for (said, _) in written {
            let end_t = (said.pts + 1) as f64 * CHUNK_SECONDS;
            match spans.last_mut() {
                Some(span) if span.start_t == said.start_t => span.end_t = end_t,
                _ => spans.push(Span {
                    start_t: said.start_t,
                    end_t,
                }),
            }
        }
        spans
    }

    const PARAMS: [(f64, f64, f64); 5] = [
        (0.5, 0.25, 0.1),
        (0.5, 0.0, 0.0),
        (0.6, 0.5, 0.3),
        (0.4, 0.1, 0.25),
        (0.5, 0.256, 0.064),
    ];

    #[test]
    fn the_rows_merge_back_into_the_spans_a_span_list_would_close() {
        for (threshold, min_speech, min_silence) in PARAMS {
            for seed in 0..40 {
                let probabilities = wandering(seed, 400);
                let mut spans = Spans::new(threshold, min_speech, min_silence);
                let closed = spans_of(&mut spans, &probabilities);
                let mut speaking = Speaking::new(threshold, min_speech, min_silence);
                let rows = merged(&rows_of(&mut speaking, &probabilities));
                assert_eq!(rows.len(), closed.len(), "seed {seed}");
                for (row, span) in rows.iter().zip(&closed) {
                    assert!(
                        (row.start_t - span.start_t).abs() < 1e-9
                            && (row.end_t - span.end_t).abs() < 1e-9,
                        "seed {seed}: {row:?} against {span:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_row_waits_longer_than_the_latency_declared_for_it() {
        for (threshold, min_speech, min_silence) in PARAMS {
            let most = latency_chunks(min_speech, min_silence) as usize;
            for seed in 0..40 {
                let mut speaking = Speaking::new(threshold, min_speech, min_silence);
                for (said, at) in rows_of(&mut speaking, &wandering(seed, 400)) {
                    let waited = at - said.pts as usize;
                    assert!(waited <= most, "seed {seed}: waited {waited} of {most}");
                }
            }
        }
    }

    #[test]
    fn the_longest_wait_is_a_short_span_that_resumes_just_before_it_closes() {
        // Seven voiced chunks are 224 ms, under the 250 ms kept; three quiet
        // ones are 96 ms, under the 100 ms that closes; the eleventh chunk is
        // voiced and is what says the first one is speech.
        let mut speaking = Speaking::new(0.5, 0.25, 0.1);
        let probabilities = [run(7, 0.9), run(3, 0.1), run(1, 0.9)].concat();
        let written = rows_of(&mut speaking, &probabilities);
        assert_eq!(written.len(), 11, "every chunk of the span, quiet ones too");
        assert_eq!(
            written[0],
            (
                Said {
                    pts: 0,
                    start_t: 0.0
                },
                10
            )
        );
        assert_eq!(latency_chunks(0.25, 0.1), 10);
        assert!((latency(0.25, 0.1) - 0.32).abs() < 1e-12);
    }

    #[test]
    fn a_dropped_span_and_the_quiet_that_closes_one_write_nothing() {
        let mut speaking = Speaking::new(0.5, 0.25, 0.1);
        // A burst of four, then quiet long enough to close it.
        assert!(rows_of(&mut speaking, &[run(4, 0.9), run(10, 0.1)].concat()).is_empty());
        let mut speaking = Speaking::new(0.5, 0.25, 0.1);
        let written = rows_of(&mut speaking, &[run(10, 0.9), run(10, 0.1)].concat());
        assert_eq!(written.len(), 10, "the voiced chunks and none of the quiet");
        assert!(written.iter().all(|(said, _)| said.start_t == 0.0));
    }

    #[test]
    fn a_row_carries_the_second_its_span_began() {
        let mut speaking = Speaking::new(0.5, 0.25, 0.1);
        let written = rows_of(
            &mut speaking,
            &[run(5, 0.1), run(10, 0.9), run(2, 0.1), run(5, 0.9)].concat(),
        );
        let begun = 5.0 * CHUNK_SECONDS;
        assert_eq!(written.len(), 17);
        assert!(written.iter().all(|(said, _)| said.start_t == begun));
        let pts: Vec<i64> = written.iter().map(|(said, _)| said.pts).collect();
        assert_eq!(
            pts,
            (5..22).collect::<Vec<_>>(),
            "in order, the gap included"
        );
    }

    #[test]
    fn a_threshold_moved_live_keeps_the_durations() {
        let mut speaking = Speaking::new(0.5, 0.25, 0.1);
        speaking.retune(0.9);
        assert!(rows_of(&mut speaking, &[run(20, 0.6), run(10, 0.1)].concat()).is_empty());
        assert_eq!(speaking.spans.min_speech, 0.25);
        assert_eq!(speaking.spans.min_silence, 0.1);
    }

    #[test]
    fn the_latency_is_a_whole_number_of_chunks_in_samples() {
        // The host holds a port's progress back by the latency in its time
        // base, rounded up: a whole number of chunks keeps the progress, and
        // so the span reducer's end of a span, on a chunk boundary.
        for min_speech in [0.0, 0.1, 0.25, 0.256, 0.5, 1.0, 2.0] {
            for min_silence in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0] {
                let chunks = latency_chunks(min_speech, min_silence);
                let samples = (latency(min_speech, min_silence) * f64::from(SAMPLE_RATE)).ceil();
                assert_eq!(samples as u64, u64::from(chunks) * CHUNK as u64);
            }
        }
    }
}
