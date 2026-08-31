//! Where somebody is speaking: the audio passes through untouched and each
//! span of speech leaves as a cue beside it.
//!
//! The graph is Silero VAD, run through `wasi:nn`. The module never opens a
//! file - the host binds the graph to a name with `-nn speech=<path>` and
//! this module asks for that name and nothing else.
//!
//! # The window
//!
//! The model reads 512 samples at a time at 16 kHz, which is 32 ms and the
//! finest a span's edges can ever be. A window is 32 of those, 16384 samples
//! or 1.024 s: enough that the per-call cost is spread over a second of
//! audio, and a whole number of chunks so no chunk straddles a call. Stride
//! is window, so the windows are disjoint and tile the stream - which is also
//! what lets the samples pass through as the very ones that arrived.
//!
//! # The state
//!
//! Silero is recurrent: each chunk's probability depends on the ones before
//! it, carried in a `state` tensor the model returns alongside the answer.
//! `compute` is stateless, so this module holds that tensor itself and hands
//! it back on the next chunk. It starts as zeros, which is what the model
//! expects at the head of a stream.
//!
//! # The rows
//!
//! One row per span - `text`, `start_t`, `end_t` - in seconds from the start
//! of the stream. `text` is the word "speech": this model hears that somebody
//! is talking, not what they said, and a span minted as a subtitle should say
//! the honest thing rather than a number. A row rides the window the span
//! closed on, which is a little after the span itself; the times are in the
//! row, so nothing downstream reads the window's own.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:vad/speech",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};
use vad_core::{seconds, Chunker, Span, Spans, CHUNK, SAMPLE_RATE};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn speech=<path>`.
const MODEL: &str = "speech";

/// The graph's own names for the tensors it takes and returns.
const INPUT_NAME: &str = "input";
const STATE_NAME: &str = "state";
const RATE_NAME: &str = "sr";
const OUTPUT_NAME: &str = "output";
const NEXT_STATE_NAME: &str = "stateN";

/// The recurrent state's shape, and how many floats that is.
const STATE_DIMS: [u32; 3] = [2, 1, 128];
const STATE_LEN: usize = 2 * 128;

/// Chunks one window covers: a second and a bit of audio.
const CHUNKS_PER_WINDOW: usize = 32;
const WINDOW: u32 = (CHUNK * CHUNKS_PER_WINDOW) as u32;

/// What a span's `text` says.
const SPEECH: &str = "speech";

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"threshold":{"type":"number","minimum":0,"maximum":1,"default":0.5},"min_speech":{"type":"number","minimum":0,"default":0.25},"min_silence":{"type":"number","minimum":0,"default":0.1}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"text":{"type":"string"},"start_t":{"type":"number"},"end_t":{"type":"number"}},"required":["text","start_t","end_t"],"additionalProperties":false}"#;

/// Silero's own recommended cut between speech and everything else.
fn default_threshold() -> f64 {
    0.5
}

/// Shorter than a spoken syllable, so what survives is somebody talking
/// rather than a door or a drum hit the model guessed at.
fn default_min_speech() -> f64 {
    0.25
}

/// A breath mid-sentence is well under this, a turn between speakers is over
/// it, so a sentence stays one span and two speakers do not become one.
fn default_min_silence() -> f64 {
    0.1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default = "default_min_speech")]
    min_speech: f64,
    #[serde(default = "default_min_silence")]
    min_silence: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            threshold: default_threshold(),
            min_speech: default_min_speech(),
            min_silence: default_min_silence(),
        }
    }
}

/// One span of speech, as the row it becomes.
#[derive(Serialize)]
struct Row {
    text: &'static str,
    start_t: f64,
    end_t: f64,
}

impl From<Span> for Row {
    fn from(span: Span) -> Row {
        Row {
            text: SPEECH,
            start_t: span.start_t,
            end_t: span.end_t,
        }
    }
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    /// The unit this stream's timestamps are counted in.
    time_base: (i32, i32),
    chunker: Chunker,
    spans: Spans,
    /// The model's recurrent state, threaded from one chunk to the next.
    state: Vec<f32>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per chunk.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// Parses and validates params, shared by `init` and `set_params`.
fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("speech: bad params: {e}"))?
    };
    if !(0.0..=1.0).contains(&parsed.threshold) {
        return Err(format!(
            "speech: threshold is a probability, and {} is not between 0 and 1",
            parsed.threshold
        ));
    }
    for (name, value) in [
        ("min_speech", parsed.min_speech),
        ("min_silence", parsed.min_silence),
    ] {
        if value < 0.0 {
            return Err(format!(
                "speech: {name} is seconds, and {value} is negative"
            ));
        }
    }
    Ok(parsed)
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("speech: {what}: {code} ({})", error.data())
}

/// Floats as the little-endian bytes a tensor carries.
fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A tensor's bytes back as floats.
fn to_floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().copied().map(f32::from_le_bytes).collect()
}

/// One chunk through the graph: how likely it is to be speech, with the
/// recurrent state advanced onto the next chunk.
fn probability(
    context: &GraphExecutionContext,
    state: &mut Vec<f32>,
    chunk: &[f32],
) -> Result<f64, String> {
    let outputs = context
        .compute(vec![
            (
                INPUT_NAME.to_string(),
                Tensor::new(&[1, CHUNK as u32], TensorType::Fp32, &to_bytes(chunk)),
            ),
            (
                STATE_NAME.to_string(),
                Tensor::new(&STATE_DIMS, TensorType::Fp32, &to_bytes(state)),
            ),
            (
                // The rate is a scalar, so it carries no dimensions at all.
                RATE_NAME.to_string(),
                Tensor::new(&[], TensorType::I64, &i64::from(SAMPLE_RATE).to_le_bytes()),
            ),
        ])
        .map_err(|e| failed("compute", &e))?;

    let named = |want: &str| {
        outputs
            .iter()
            .find(|(name, _)| name == want)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| format!("speech: the graph returned no tensor named {want}"))
    };

    let next = to_floats(&named(NEXT_STATE_NAME)?.data());
    if next.len() != STATE_LEN {
        return Err(format!(
            "speech: the graph returned {} state value(s), expected {STATE_LEN}",
            next.len()
        ));
    }
    *state = next;

    let answer = to_floats(&named(OUTPUT_NAME)?.data());
    let first = answer
        .first()
        .ok_or_else(|| "speech: the graph returned no probability".to_string())?;
    Ok(f64::from(*first))
}

struct Speech;

impl Guest for Speech {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "speech".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // What the model was trained on, and the host conforms to it.
                sample_rates: vec![SAMPLE_RATE],
                channel_counts: vec![1],
                // The rows say that speech is there, not what language it is
                // in, so there is nothing here to tag a track with.
                rows_language: vec![],
            },
            window: WINDOW,
            stride: WINDOW,
            // The model's own state and the span being built both carry from
            // one call to the next.
            pure: false,
            // The samples pass through as they arrived.
            one_to_one: true,
            reads_rows: false,
            // What leaves is this module's own spans and nothing else.
            forwards_rows: false,
            // One stream in: the audio it listens to.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Audio(audio) = format else {
            return Err("speech listens to samples, and this stream is video".to_string());
        };
        if audio.sample_fmt != "f32" {
            return Err(format!(
                "speech does not accept sample format {}",
                audio.sample_fmt
            ));
        }
        if audio.sample_rate != SAMPLE_RATE {
            return Err(format!(
                "speech listens at {SAMPLE_RATE} Hz, and this instance is {} Hz",
                audio.sample_rate
            ));
        }
        if audio.channels != 1 {
            return Err(format!(
                "speech listens in mono, and this instance has {} channels",
                audio.channels
            ));
        }
        let parsed = parse_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first chunk is what a provider picks its kernels on, and every
        // chunk after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                time_base: (stream_info.time_base.num, stream_info.time_base.den),
                chunker: Chunker::new(),
                spans: Spans::new(parsed.threshold, parsed.min_speech, parsed.min_silence),
                // Zeros are what the model expects at the head of a stream.
                state: vec![0.0; STATE_LEN],
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                // A span already open runs on under the new thresholds; the
                // audio behind it is gone, so there is nothing to remeasure.
                opened
                    .spans
                    .retune(parsed.threshold, parsed.min_speech, parsed.min_silence);
            }
        });
        Ok(())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        OPENED.with(|o| {
            let mut borrowed = o.borrow_mut();
            let opened = borrowed
                .as_mut()
                .expect("init loads the graph before any audio arrives");
            let Opened {
                time_base,
                chunker,
                spans,
                state,
                context,
                ..
            } = opened;

            let mut out: Vec<OutFrame> = Vec::with_capacity(window.len() as usize);
            for index in 0..window.len() {
                let pts = window.pts(index);
                // Every window arrives whole, so the clock is taken from the
                // payload's own timestamp rather than accumulated.
                if chunker.aligned() {
                    chunker.seek(seconds(pts, time_base.0, time_base.1));
                }
                let mut rows: Vec<String> = Vec::new();
                chunker.feed(&window.fetch(index), |chunk, at| {
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than reporting silence.
                    let found = probability(context, state, chunk)
                        .unwrap_or_else(|message| panic!("{message}"));
                    if let Some(span) = spans.push(found, at) {
                        rows.push(row(span));
                    }
                });
                out.push(OutFrame {
                    pts,
                    frame: FramePayload::Same,
                    rows,
                });
            }

            // The end of the stream closes whatever is still open. It rides
            // the last payload when there is one, and trails the stream when
            // the final call carried no samples at all.
            let mut trailing = Vec::new();
            if last {
                if let Some(span) = spans.finish() {
                    match out.last_mut() {
                        Some(frame) => frame.rows.push(row(span)),
                        None => trailing.push(row(span)),
                    }
                }
            }
            Processed {
                frames: out,
                trailing,
            }
        })
    }
}

/// One span as the NDJSON line it leaves as.
fn row(span: Span) -> String {
    serde_json::to_string(&Row::from(span)).expect("row serializes")
}

export!(Speech);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}", "  "] {
            let parsed = parse_params(written).expect("the defaults");
            assert_eq!(parsed.threshold, 0.5);
            assert_eq!(parsed.min_speech, 0.25);
            assert_eq!(parsed.min_silence, 0.1);
        }
    }

    #[test]
    fn each_parameter_can_be_set_on_its_own() {
        let parsed = parse_params(r#"{"threshold":0.8}"#).expect("one of them");
        assert_eq!(parsed.threshold, 0.8);
        assert_eq!(parsed.min_speech, 0.25, "the rest keep their defaults");
    }

    #[test]
    fn a_threshold_outside_a_probability_is_refused() {
        assert!(parse_params(r#"{"threshold":1.5}"#).is_err());
        assert!(parse_params(r#"{"threshold":-0.1}"#).is_err());
    }

    #[test]
    fn a_negative_duration_is_refused() {
        assert!(parse_params(r#"{"min_speech":-1}"#).is_err());
        assert!(parse_params(r#"{"min_silence":-0.5}"#).is_err());
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(parse_params(r#"{"treshold":0.5}"#).is_err());
    }

    #[test]
    fn a_span_becomes_a_cue_shaped_row() {
        let written = row(Span {
            start_t: 1.5,
            end_t: 2.25,
        });
        assert_eq!(
            written, r#"{"text":"speech","start_t":1.5,"end_t":2.25}"#,
            "the three columns a cue declares, and nothing else"
        );
    }

    #[test]
    fn a_window_is_a_whole_number_of_chunks() {
        // No chunk straddles a call, which is what lets the state advance a
        // window at a time without anything held over.
        assert_eq!(WINDOW as usize % CHUNK, 0);
    }

    #[test]
    fn floats_survive_the_trip_through_a_tensors_bytes() {
        let values = [0.0f32, -1.0, 0.5, f32::MIN_POSITIVE];
        assert_eq!(to_floats(&to_bytes(&values)), values);
    }
}
