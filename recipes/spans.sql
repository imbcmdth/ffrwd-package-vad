-- Every span of speech in a clip, written as ndjson: one line per span, with the seconds it runs between.
-- variables: source (input media path), track (audio track index, defaults to the first), dest (output path, e.g. spans.ndjson), threshold (score a 32 ms chunk counts as speech at, 0 to 1, defaults to 0.5), min_speech (shortest span kept, in seconds, defaults to 0.25), min_silence (quiet needed to close a span, in seconds, defaults to 0.1)
-- example: ffrwd compile -f packages/ffrwd/vad/recipes/spans.sql -v source=interview.mp4 -v dest=spans.ndjson
COPY (
  SELECT ffrwd.vad.speech(a,
                          COALESCE(:threshold, 0.5),
                          COALESCE(:min_speech, 0.25),
                          COALESCE(:min_silence, 0.1)).segments
  FROM input(:'source') f, unnest(f.audio) a
  WHERE a.index = COALESCE(:track, 1)
) TO :'dest'
