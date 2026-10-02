# ffrwd/vad

Voice activity detection with Silero VAD. `speech` listens to an audio
stream and writes a row for every 32 ms of speech in it, so a query can
write the spans out, mark them on the clip, or hand them to a
transcriber that then spends nothing on the silence.

Requires ffrwd 0.29.

```pgsql
COPY (
  SELECT ffrwd.merge_spans(ffrwd.vad.speech(a), max_span => 30)
  FROM input('interview.mp4') f, unnest(f.audio) a
  WHERE a.index = 1
) TO 'spans.ndjson'
```

Each row carries `start_t`, the second its span began, and `text`, the
word "speech". `ffrwd.merge_spans` turns the rows of a span into one
cue running from its first voiced chunk to the end of its last; written
beside the clip's video and audio, the cues land as a subtitle track
marking where anybody talks. A span longer than `max_span` is cut there
and goes on as the next cue.

The model scores 32 ms at a time; three parameters turn scores into
spans. `threshold` is the score a chunk counts as speech at, 0.5 by
default. `min_silence` is how much quiet closes a span, 0.1 s, so a
breath mid-sentence carries across. `min_speech` is the shortest span
kept, 0.25 s, so a door or a drum hit never becomes a row.

A row leaves once it is sure: once its span is long enough to keep, and
for a quiet chunk inside one, once the speech resumes. The node declares
how long that can take, 0.32 s at the defaults, so a reader pairing the
rows by time waits that much and no more. `threshold` can change while
a query runs; the two durations cannot, since they set that wait.

The weights are pinned in the manifest and land beside the module at
install.

## Exports

- `speech(a audio_stream, threshold DEFAULT 0.5, min_speech DEFAULT
  0.25, min_silence DEFAULT 0.1)` returns `STRUCT(start_t number, text
  text)[]`: a row a chunk while speech lasts. The sound is not handed
  back; a reader takes it from the source.

## Recipes

- `spans` - every span of speech in a clip, as ndjson.
- `speech-track` - the clip with a subtitle track marking the speech.

```
ffrwd run ffrwd/vad:spans -v source=interview.mp4 -v dest=spans.ndjson
```

## Building

```
cargo build --target wasm32-wasip2 --release
```
