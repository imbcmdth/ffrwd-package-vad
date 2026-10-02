-- The one model export, hosted as the wasm module the package ships. The
-- weights are pinned in the manifest and land beside the module at install.
--
-- `speech` returns rows alone, one per 32 ms chunk of each span of speech:
-- `start_t` is the second that chunk's span began and `text` is the word
-- "speech". A reader takes the sound from the source, and
-- `ffrwd.merge_spans(ffrwd.vad.speech(a), max_span => 30)` turns the rows
-- into one cue per span, each running from its first voiced chunk to the
-- end of its last. The model scores 32 ms at a time; the three parameters
-- are how those scores become spans. `threshold` is the score a chunk
-- counts as speech at. `min_silence` is how much quiet has to follow a span
-- before it closes, so a breath mid-sentence carries across rather than
-- splitting it. `min_speech` is the shortest span kept, so a door or a drum
-- hit the model guessed at never becomes a row. A row leaves once it is
-- sure, and the two durations set how long that can take: 0.32 s at the
-- defaults.
CREATE FUNCTION speech(a audio_stream,
                       threshold number DEFAULT 0.5,
                       min_speech number DEFAULT 0.25,
                       min_silence number DEFAULT 0.1)
RETURNS STRUCT(start_t number, text text)[]
  AS 'target/wasm32-wasip2/release/speech.wasm', 'speech' LANGUAGE wasm;
