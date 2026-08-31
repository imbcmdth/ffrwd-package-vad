//! The arithmetic the module is built on: samples cut into the chunks the
//! model takes, and per-chunk probabilities turned into spans of speech.
//! Neither half touches the model, so both are tested on the host.

/// Samples one model chunk covers. Silero VAD v5 is trained on exactly this
/// many at 16 kHz and accepts no other length.
pub const CHUNK: usize = 512;

/// The rate the model works at, and the only one the module accepts.
pub const SAMPLE_RATE: u32 = 16_000;

/// Seconds one chunk covers: 32 ms, which is the resolution a span's edges
/// can ever have.
pub const CHUNK_SECONDS: f64 = CHUNK as f64 / SAMPLE_RATE as f64;

/// A timestamp in the stream's own unit, as seconds. `den` is always
/// positive, so a negative timestamp stays negative.
pub fn seconds(ticks: i64, num: i32, den: i32) -> f64 {
    ticks as f64 * f64::from(num) / f64::from(den)
}

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

/// Samples arriving in whatever pieces the host cuts, handed on in the fixed
/// chunks the model takes.
///
/// The clock is counted in samples rather than added up per chunk, so a long
/// stream's last span is timed as exactly as its first.
pub struct Chunker {
    /// Samples not yet a whole chunk.
    pending: Vec<f32>,
    /// The second `consumed` is counted from.
    base: f64,
    /// Samples handed on since `base`.
    consumed: u64,
}

impl Default for Chunker {
    fn default() -> Chunker {
        Chunker::new()
    }
}

impl Chunker {
    pub fn new() -> Chunker {
        Chunker {
            pending: Vec::new(),
            base: 0.0,
            consumed: 0,
        }
    }

    /// Whether nothing is held back, which is when a payload's own timestamp
    /// may be believed over the count kept here.
    pub fn aligned(&self) -> bool {
        self.pending.is_empty()
    }

    /// Moves the clock onto `seconds`. Only meaningful while aligned.
    pub fn seek(&mut self, seconds: f64) {
        self.base = seconds;
        self.consumed = 0;
    }

    /// The second sample `offset` sits at, counted from `base`.
    fn at(&self, offset: u64) -> f64 {
        self.base + offset as f64 / f64::from(SAMPLE_RATE)
    }

    /// The second the samples fed so far run out at, whole chunk or not.
    pub fn end(&self) -> f64 {
        self.at(self.consumed + self.pending.len() as u64)
    }

    /// Mono f32 samples as the host lays them out. Every whole chunk they
    /// complete goes to `run` with the second that chunk starts at; a
    /// remainder waits for the samples that finish it.
    pub fn feed(&mut self, bytes: &[u8], mut run: impl FnMut(&[f32], f64)) {
        let (words, _) = bytes.as_chunks::<4>();
        self.pending
            .extend(words.iter().copied().map(f32::from_le_bytes));

        let Chunker {
            pending,
            base,
            consumed,
        } = self;
        let mut taken = 0;
        while pending.len() - taken >= CHUNK {
            run(
                &pending[taken..taken + CHUNK],
                *base + *consumed as f64 / f64::from(SAMPLE_RATE),
            );
            *consumed += CHUNK as u64;
            taken += CHUNK;
        }
        pending.drain(..taken);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes a host hands over for these samples.
    fn bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    /// A tone at `hz`, which is what a burst of speech stands in for here:
    /// the chunker never looks at the values, only counts them.
    fn tone(count: usize, hz: f64) -> Vec<f32> {
        (0..count)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / f64::from(SAMPLE_RATE)).sin() as f32)
            .collect()
    }

    /// Every chunk the feeds produce, as (first sample, start second).
    fn chunks_of(chunker: &mut Chunker, feeds: &[Vec<f32>]) -> Vec<(f32, f64)> {
        let mut seen = Vec::new();
        for feed in feeds {
            chunker.feed(&bytes(feed), |chunk, at| seen.push((chunk[0], at)));
        }
        seen
    }

    #[test]
    fn a_timestamp_becomes_seconds_in_the_streams_own_unit() {
        assert_eq!(seconds(16_000, 1, 16_000), 1.0);
        assert_eq!(seconds(48_000, 1, 48_000), 1.0);
        assert_eq!(seconds(0, 1, 16_000), 0.0);
        // A frame rate's unit, which an audio instance may still be counted
        // in when the container says so.
        assert!((seconds(30, 1, 25) - 1.2).abs() < 1e-12);
    }

    #[test]
    fn samples_are_handed_on_a_whole_chunk_at_a_time() {
        let mut chunker = Chunker::new();
        let seen = chunks_of(&mut chunker, &[tone(CHUNK * 3, 440.0)]);
        assert_eq!(seen.len(), 3);
        for (index, (_, at)) in seen.iter().enumerate() {
            assert!(
                (at - index as f64 * CHUNK_SECONDS).abs() < 1e-12,
                "chunk {index} starts at {at}"
            );
        }
    }

    #[test]
    fn a_remainder_waits_for_the_samples_that_finish_it() {
        let mut chunker = Chunker::new();
        assert_eq!(
            chunks_of(&mut chunker, &[tone(CHUNK + 100, 440.0)]).len(),
            1
        );
        assert!(!chunker.aligned(), "100 samples are held back");

        // The next feed brings the 412 that finish the chunk they started.
        let seen = chunks_of(&mut chunker, &[tone(CHUNK - 100, 440.0)]);
        assert_eq!(seen.len(), 1);
        assert!(
            (seen[0].1 - CHUNK_SECONDS).abs() < 1e-12,
            "and it starts where the first one ended"
        );
        assert!(chunker.aligned());
    }

    #[test]
    fn a_chunk_carries_the_samples_that_went_in() {
        // Each chunk's first sample marks which one it is.
        let mut samples = tone(CHUNK * 2, 440.0);
        samples[0] = -1.0;
        samples[CHUNK] = -2.0;
        let mut chunker = Chunker::new();
        let seen = chunks_of(&mut chunker, &[samples]);
        assert_eq!(seen[0].0, -1.0);
        assert_eq!(seen[1].0, -2.0);
    }

    #[test]
    fn the_clock_starts_where_the_stream_is_seeked_to() {
        let mut chunker = Chunker::new();
        chunker.seek(12.5);
        let seen = chunks_of(&mut chunker, &[tone(CHUNK * 2, 440.0)]);
        assert!((seen[0].1 - 12.5).abs() < 1e-12);
        assert!((seen[1].1 - (12.5 + CHUNK_SECONDS)).abs() < 1e-12);
        assert!((chunker.end() - (12.5 + 2.0 * CHUNK_SECONDS)).abs() < 1e-12);
    }

    #[test]
    fn the_end_counts_the_samples_still_held_back() {
        let mut chunker = Chunker::new();
        chunks_of(&mut chunker, &[tone(CHUNK + 160, 440.0)]);
        // A chunk handed on plus 160 samples, which is 10 ms at 16 kHz.
        assert!((chunker.end() - (CHUNK_SECONDS + 0.01)).abs() < 1e-12);
    }

    #[test]
    fn an_hours_worth_of_chunks_is_still_timed_exactly() {
        // Adding a chunk's length up per chunk would drift; counting samples
        // does not. 112500 chunks is an hour.
        let mut chunker = Chunker::new();
        let mut last = 0.0;
        for _ in 0..112_500 {
            chunker.feed(&bytes(&vec![0.0; CHUNK]), |_, at| last = at);
        }
        let expected = (112_499 * CHUNK) as f64 / f64::from(SAMPLE_RATE);
        assert!(
            (last - expected).abs() < 1e-9,
            "the last chunk is at {last}"
        );
    }

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
}
