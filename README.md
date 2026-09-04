# ffrwd/vad

Voice activity detection with Silero VAD, hosted in wasm. `speech` hands
an audio stream back untouched with one cue per span of speech beside
it, so a query can write the spans out, mark them on the clip, or hand
them to a transcriber that then spends nothing on the silence.

```pgsql
COPY (
  SELECT ffrwd.vad.speech(a).segments
  FROM input('interview.mp4') f, unnest(f.audio) a
  WHERE a.index = 1
) TO 'spans.ndjson'
```

Each cue's `text` is the word "speech" and `start_t`/`end_t` are the
span's own seconds. Written beside the clip's video and audio, the cues
land as a subtitle track marking where anybody talks.

The model scores 32 ms at a time; three parameters turn scores into
spans. `threshold` is the score a chunk counts as speech at, 0.5 by
default. `min_silence` is how much quiet closes a span, 0.1 s, so a
breath mid-sentence carries across. `min_speech` is the shortest span
kept, 0.25 s, so a door or a drum hit never becomes a cue.

The weights are pinned in the manifest and land beside the module at
install.

## Exports

- `speech(a audio_stream, threshold DEFAULT 0.5, min_speech DEFAULT
  0.25, min_silence DEFAULT 0.1)` returns `STRUCT(a audio_stream,
  segments cue[])`: the audio as it came, and the spans.

## Recipes

- `spans` - every span of speech in a clip, as ndjson.
- `speech-track` - the clip with a subtitle track marking the speech.

```
ffrwd ffrwd.vad.spans -v source=interview.mp4 -v dest=spans.ndjson
```

## Building

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```
