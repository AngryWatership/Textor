# Session Log

## Turn 1
User asked: is it trivial to build an Android text editor app with TTS, where
TTS = primary TTS model + small LLM for corrections. Wants validation in this
environment (Python) first using sample audio/text, then an architecture doc
with testing/direction guidance.

Actions:
- Inspected repo: empty, branch `claude/android-text-editor-tts-1jqfu6`, no commits.
- Checked installed stack: Python 3.13, numpy only. No torch/transformers/TTS/espeak.
- Installed `espeak-ng` via apt (offline, works, no network needed at runtime).
- Installed `piper-tts` via pip (works), but voice model download failed:
  proxy status shows huggingface.co and github.com are policy-denied
  (403 on CONNECT) in this sandbox's network policy. So no real neural TTS
  voice models or LLM weights can be fetched here.
- Decision: validate the *pipeline shape* (text -> correction stage -> TTS
  stage -> audio) using espeak-ng as a stand-in primary TTS engine and a
  deterministic text-normalization module as a stand-in for the small
  correction LLM, since real model weights are unreachable from this
  container. Document the real production stack (Piper neural voices +
  on-device small LLM via llama.cpp/MediaPipe) separately for Android,
  to be validated on-device or in an environment with model access.

## Turn 2
User asked: is the implementation usable now, and how accurate/good is it —
check using a dataset from the internet if appropriate.

Actions:
- Checked network reachability for text-normalization datasets: kaggle.com,
  openslr.org, zenodo.org, huggingface.co all unreachable (policy-blocked,
  connection refused/000). raw.githubusercontent.com and api.github.com
  behaved inconsistently (raw worked for small files, api.github.com was
  intercepted by the proxy with an unrelated error) and the known dataset
  repo (rwsproat/text-normalization-data) ships data via release assets,
  not raw git files, so it wasn't fetchable either.
- Built a hand-curated 23-case benchmark (`scripts/benchmark_correct.py`)
  covering abbreviations, numbers (small/large), decimals, currency, dates,
  ordinals, acronyms, time, percent — the standard categories any TTS text
  normalizer must handle — and scored `correct_for_speech()` against it
  with strict exact-match.
- Result: 6/23 (26%) exact match. Failures concentrated exactly where
  expected: large numbers, decimals, currency, dates, times, ordinals,
  percents are not handled at all; abbreviation substitution is naive
  (no word boundaries/context, e.g. mishandles "St." meaning Street vs Saint
  in the same sentence).
- Updated `docs/ARCHITECTURE.md` with an explicit "is this usable" section:
  answer is no, current code is a pipeline-shape prototype only. Audio
  quality itself still cannot be assessed in this sandbox (no MOS model, no
  reference audio, no human listener available here).
