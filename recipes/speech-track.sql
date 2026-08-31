-- The clip with a subtitle track marking where anybody is speaking, for scrubbing dialogue or spotting dead air.
-- variables: source (input media path), track (audio track index, defaults to the first), dest (output path), threshold (score a 32 ms chunk counts as speech at, 0 to 1, defaults to 0.5), min_speech (shortest span kept, in seconds, defaults to 0.25), min_silence (quiet needed to close a span, in seconds, defaults to 0.1)
-- example: ffrwd compile -f packages/ffrwd/vad/recipes/speech-track.sql -v source=interview.mp4 -v dest=marked.mkv
COPY (
  SELECT f.video[1], a,
         ffrwd.vad.speech(a,
                          COALESCE(:threshold, 0.5),
                          COALESCE(:min_speech, 0.25),
                          COALESCE(:min_silence, 0.1)).segments
  FROM input(:'source') f, unnest(f.audio) a
  WHERE a.index = COALESCE(:track, 1)
) TO :'dest'
