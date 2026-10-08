## `ljspeech_sample_*`

8 real utterances (audio + transcript pairs) pulled from `coqui-ai/TTS`'s own
test fixtures (`tests/data/ljspeech` in that repo, commit `dbf1a08`), which
in turn are a small slice of the public LJSpeech corpus: single-speaker
readings of public-domain non-fiction texts, originally recorded for
LibriVox.

`ljspeech_sample_metadata.csv` is `id|raw_text|normalized_text` — the
`normalized_text` column is LJSpeech's own ground-truth spoken form (e.g.
"1455" is annotated "fourteen fifty-five"), written by the dataset's
original authors, not by us. That makes it a genuine independent reference
for correction-stage accuracy, paired with real recorded audio of a human
actually speaking that normalized form.

Used by `scripts/evaluate_on_corpus.py`.
