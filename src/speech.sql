-- The one model export, hosted as the wasm module the package ships. The
-- weights are pinned in the manifest and land beside the module at install.
--
-- `speech` hands the audio back untouched with one cue per span of speech
-- beside it: `text` is the word "speech", and `start_t`/`end_t` are the
-- span's own seconds. The model reads 32 ms at a time and scores each; the
-- three parameters are how those scores become spans. `threshold` is the
-- score a chunk counts as speech at. `min_silence` is how much quiet has to
-- follow a span before it closes, so a breath mid-sentence carries across
-- rather than splitting it. `min_speech` is the shortest span kept, so a
-- door or a drum hit the model guessed at never becomes a cue.
CREATE FUNCTION speech(a audio_stream,
                       threshold number DEFAULT 0.5,
                       min_speech number DEFAULT 0.25,
                       min_silence number DEFAULT 0.1)
RETURNS STRUCT(a audio_stream, segments cue[])
  AS 'target/wasm32-wasip2/release/speech.wasm', 'speech' LANGUAGE wasm;
