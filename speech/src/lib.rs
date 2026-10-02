//! Where somebody is speaking: a row for every 32 ms of a span of speech,
//! each carrying the second its span began.
//!
//! The graph is Silero VAD, run through `wasi:nn`. The module never opens a
//! file - the host binds the graph to a name with `-nn speech=<path>` and
//! this module asks for that name and nothing else.
//!
//! # The window
//!
//! The model reads 512 samples at a time at 16 kHz, which is 32 ms and the
//! finest a span's edges can ever be. That is the node's window and stride:
//! a tick is one chunk, and the host cuts the stream into them.
//!
//! # The state
//!
//! Silero is recurrent: each chunk's probability depends on the ones before
//! it, carried in a `state` tensor the model returns alongside the answer.
//! `compute` is stateless, so this module holds that tensor itself and hands
//! it back on the next chunk. It starts as zeros, which is what the model
//! expects at the head of a stream. That is why the node is not pure.
//!
//! # The rows
//!
//! `start_t` and `text` on `speech`, a row a chunk from the first voiced
//! chunk of a span to its last, stamped at the chunk. `text` is the word
//! "speech": this model hears that somebody is talking, not what they said.
//! A row leaves once it is sure, at most the latency the shape declares after
//! the chunk it is stamped at, and `ffrwd.merge_spans` turns the rows of a
//! span back into one cue.

// `generate_all`: the world's interfaces are wasi:nn's, a package of its own,
// and without it bindgen expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: "wit-world",
    world: "ffrwd:vad/speech",
    generate_all,
});

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::{Deserialize, Serialize};
use vad_core::{latency, Speaking, CHUNK, SAMPLE_RATE};
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

/// What a row's `text` says.
const SPEECH: &str = "speech";

/// The output the rows leave on.
const OUT: &str = "speech";

/// Silero's own recommended cut between speech and everything else is 0.5. A
/// `min_speech` of 0.25 s is shorter than a spoken syllable, so what survives
/// is somebody talking rather than a door or a drum hit the model guessed at.
/// A breath mid-sentence is well under a `min_silence` of 0.1 s and a turn
/// between speakers is over it, so a sentence stays one span and two speakers
/// do not become one.
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"threshold":{"type":"number","minimum":0,"maximum":1,"default":0.5},"min_speech":{"type":"number","minimum":0,"default":0.25},"min_silence":{"type":"number","minimum":0,"default":0.1}},"additionalProperties":false}"#;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
struct Params {
    threshold: f64,
    min_speech: f64,
    min_silence: f64,
}

/// One row: the second the span this chunk belongs to began.
#[derive(Default, Serialize)]
struct Row {
    start_t: f64,
    text: &'static str,
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

/// Where a chunk's probability comes from: the graph in a module, a script
/// in a test.
trait Listen: Sized + 'static {
    fn load() -> Result<Self, String>;
    fn probability(&mut self, chunk: &[f32]) -> Result<f64, String>;
}

/// Silero through `wasi:nn`, with the recurrent state it threads from one
/// chunk to the next.
struct Model {
    state: Vec<f32>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per chunk.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

impl Listen for Model {
    /// The graph is loaded once per instance, and the session built once:
    /// the first chunk is what a provider picks its kernels on, and every
    /// chunk after it reuses them.
    fn load() -> Result<Model, String> {
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;
        Ok(Model {
            // Zeros are what the model expects at the head of a stream.
            state: vec![0.0; STATE_LEN],
            context,
            _graph: graph,
        })
    }

    /// One chunk through the graph: how likely it is to be speech, with the
    /// recurrent state advanced onto the next chunk.
    fn probability(&mut self, chunk: &[f32]) -> Result<f64, String> {
        let outputs = self
            .context
            .compute(vec![
                (
                    INPUT_NAME.to_string(),
                    Tensor::new(&[1, CHUNK as u32], TensorType::Fp32, &to_bytes(chunk)),
                ),
                (
                    STATE_NAME.to_string(),
                    Tensor::new(&STATE_DIMS, TensorType::Fp32, &to_bytes(&self.state)),
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
        self.state = next;

        let answer = to_floats(&named(OUTPUT_NAME)?.data());
        let first = answer
            .first()
            .ok_or_else(|| "speech: the graph returned no probability".to_string())?;
        Ok(f64::from(*first))
    }
}

struct Speech<L: Listen = Model> {
    a: u32,
    time_base: Rational,
    params: Params,
    speaking: Speaking,
    model: L,
}

impl<L: Listen> Node for Speech<L> {
    const NAME: &'static str = "speech";
    const VERSION: &'static str = "0.2.1";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::audio("a")
                    .clock()
                    .window(CHUNK as u32, CHUNK as u32)
                    .sample_formats(&["f32"])
                    .sample_rates(&[SAMPLE_RATE])
                    .channel_counts(&[1]),
            )
            .output(
                Output::rows(OUT)
                    .schema::<Row>()
                    .latency(latency(params.min_speech, params.min_silence)),
            ))
    }

    fn init(params: Params, init: &Init) -> Result<Speech<L>> {
        let a = init.stream("a")?;
        Ok(Speech {
            a: a.id,
            time_base: a.info.time_base,
            params,
            speaking: Speaking::new(params.threshold, params.min_speech, params.min_silence),
            model: L::load()?,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        if (params.min_speech, params.min_silence)
            != (self.params.min_speech, self.params.min_silence)
        {
            return Err(
                "speech: min_speech and min_silence set how late a row may leave, \
                        which the node declared when it opened; only threshold moves live"
                    .into(),
            );
        }
        self.speaking.retune(params.threshold);
        self.params = params;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.a) {
            let samples = to_floats(&tick.fetch(self.a, frame.index));
            // The stream's last few samples, short of a chunk: the model
            // takes 512 exactly.
            if samples.len() != CHUNK {
                continue;
            }
            let probability = self.model.probability(&samples)?;
            let start_t = self.time_base.seconds(frame.pts);
            for said in self.speaking.push(probability, frame.pts, start_t) {
                let row = Row {
                    start_t: said.start_t,
                    text: SPEECH,
                };
                out.row(OUT, said.pts, &row)?;
            }
        }
        Ok(())
    }
}

ffrwd_node::export!(Speech);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::BoundStream;

    thread_local! {
        static SCRIPT: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Probabilities handed out in order, one a chunk, from the script the
    /// test set before opening the node.
    struct Scripted(std::collections::VecDeque<f64>);

    impl Listen for Scripted {
        fn load() -> Result<Scripted, String> {
            Ok(Scripted(SCRIPT.with(|s| s.borrow().clone()).into()))
        }

        fn probability(&mut self, _: &[f32]) -> Result<f64, String> {
            self.0
                .pop_front()
                .ok_or_else(|| "the script ran out".to_owned())
        }
    }

    fn read(written: &str) -> std::result::Result<Params, String> {
        ffrwd_node::read_params::<Params>(PARAMS_SCHEMA, written).map(|(params, _)| params)
    }

    fn opened(params: &str, script: &[f64]) -> Harness<Speech<Scripted>> {
        SCRIPT.with(|s| *s.borrow_mut() = script.to_vec());
        let a = BoundStream::audio("a", 0, SAMPLE_RATE, 1, "f32");
        Harness::new(params, vec![a]).expect("opens")
    }

    /// One tick of `samples` at `pts`.
    fn chunk(node: &Harness<Speech<Scripted>>, pts: i64, samples: usize) -> ffrwd_node::mock::Tick {
        node.tick(pts)
            .frame_with(0, pts, Some(samples as i64), &[], vec![0; samples * 4])
    }

    /// Every row the chunks at `pts` wrote, as (pts, start_t).
    fn rows(node: &mut Harness<Speech<Scripted>>, pts: &[i64]) -> Vec<(i64, f64)> {
        let mut written = Vec::new();
        for at in pts {
            let emitted = node.process(&chunk(node, *at, CHUNK)).expect("processes");
            for (pts, json) in emitted.messages(OUT) {
                let row: serde_json::Value = serde_json::from_str(&json).expect("json");
                assert_eq!(row["text"], SPEECH);
                written.push((pts, row["start_t"].as_f64().expect("start_t")));
            }
        }
        written
    }

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}", "  "] {
            let parsed = read(written).expect("the defaults");
            assert_eq!(parsed.threshold, 0.5);
            assert_eq!(parsed.min_speech, 0.25);
            assert_eq!(parsed.min_silence, 0.1);
        }
    }

    #[test]
    fn each_parameter_can_be_set_on_its_own() {
        let parsed = read(r#"{"threshold":0.8}"#).expect("one of them");
        assert_eq!(parsed.threshold, 0.8);
        assert_eq!(parsed.min_speech, 0.25, "the rest keep their defaults");
    }

    #[test]
    fn a_threshold_outside_a_probability_is_refused() {
        assert!(read(r#"{"threshold":1.5}"#).is_err());
        assert!(read(r#"{"threshold":-0.1}"#).is_err());
    }

    #[test]
    fn a_negative_duration_is_refused() {
        assert!(read(r#"{"min_speech":-1}"#).is_err());
        assert!(read(r#"{"min_silence":-0.5}"#).is_err());
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(read(r#"{"treshold":0.5}"#).is_err());
    }

    #[test]
    fn floats_survive_the_trip_through_a_tensors_bytes() {
        let values = [0.0f32, -1.0, 0.5, f32::MIN_POSITIVE];
        assert_eq!(to_floats(&to_bytes(&values)), values);
    }

    #[test]
    fn the_shape_is_a_chunk_of_sixteen_kilohertz_mono_in_and_rows_out() {
        let node = opened("", &[]);
        let shape = node.shape();
        let input = &shape.inputs[0];
        assert_eq!((input.window, input.stride), (512, 512));
        assert_eq!(input.accepts.sample_rates, [16_000]);
        assert_eq!(input.accepts.channel_counts, [1]);
        assert_eq!(input.accepts.sample_formats, ["f32"]);
        let output = &shape.outputs[0];
        assert_eq!(output.name, "speech");
        assert!((output.latency - 0.32).abs() < 1e-12, "{}", output.latency);
        let schema: serde_json::Value =
            serde_json::from_str(output.schema.as_deref().expect("a schema")).expect("json");
        assert_eq!(schema["properties"]["start_t"]["type"], "number");
        assert_eq!(schema["properties"]["text"]["type"], "string");
        assert!(
            !shape.pure,
            "the model's state runs from one chunk to the next"
        );
    }

    #[test]
    fn the_latency_follows_the_durations() {
        let node = opened(r#"{"min_speech":0.5,"min_silence":0.3}"#, &[]);
        // 16 chunks kept and 10 to close: 24 chunks of 32 ms.
        assert!((node.shape().outputs[0].latency - 0.768).abs() < 1e-12);
    }

    #[test]
    fn a_span_leaves_as_a_row_a_chunk_stamped_at_the_chunk() {
        // Two quiet chunks, ten voiced, five quiet.
        let script = [vec![0.1; 2], vec![0.9; 10], vec![0.1; 5]].concat();
        let mut node = opened("", &script);
        let pts: Vec<i64> = (0..17).map(|n| n * CHUNK as i64).collect();
        let written = rows(&mut node, &pts);
        assert_eq!(written.len(), 10);
        let begun = 2.0 * CHUNK as f64 / f64::from(SAMPLE_RATE);
        for (n, (pts, start_t)) in written.iter().enumerate() {
            assert_eq!(*pts, (n as i64 + 2) * CHUNK as i64);
            assert_eq!(*start_t, begun);
        }
    }

    #[test]
    fn a_hole_in_the_clock_moves_the_rows_with_it() {
        // The conformed sound of the file the survey found a pts gap on is
        // missing three samples at 37. A tick past a hole starts where the
        // host says it does, and the rows go on in order from there: nothing
        // here passes samples through, so nothing has a gap to keep.
        let mut node = opened("", &[0.9; 10]);
        let pts: Vec<i64> = (0..10)
            .map(|n| n * CHUNK as i64 + if n >= 4 { 3 } else { 0 })
            .collect();
        let written = rows(&mut node, &pts);
        assert_eq!(written.iter().map(|(p, _)| *p).collect::<Vec<_>>(), pts);
        assert!(written.iter().all(|(_, start_t)| *start_t == 0.0));
    }

    #[test]
    fn the_last_few_samples_short_of_a_chunk_are_not_scored() {
        let mut node = opened("", &[]);
        let tick = chunk(&node, 0, CHUNK - 100).last();
        let emitted = node
            .process(&tick)
            .expect("the empty script is never asked");
        assert!(emitted.items.is_empty());
    }

    #[test]
    fn the_threshold_moves_live_and_the_durations_do_not() {
        let mut node = opened("", &[]);
        node.set_params(r#"{"threshold":0.7}"#)
            .expect("the threshold moves");
        let refused = node
            .set_params(r#"{"threshold":0.7,"min_silence":0.2}"#)
            .expect_err("a duration does not");
        assert!(refused.contains("only threshold"), "{refused}");
    }
}
