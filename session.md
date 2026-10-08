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

## Turn 3
User asked me to actually try code to access datasets, navigate the internet
harder rather than stopping at the first block.

Actions:
- Tested more hosts/mirrors directly: cdn.jsdelivr.net, hf-mirror.com,
  raw.githack.com, gitee.com, archive.org all unreachable (policy or down).
  codeload.github.com and api.github.com returned a distinct error pointing
  at `add_repo` -- realized GitHub access is gated per-repo via that tool,
  not a blanket domain block.
- Used `add_repo` on `rwsproat/text-normalization-data` (the known TN
  Challenge dataset repo) -- confirmed public read access, cloned it via
  git. Found the repo itself only contains a README + a `split.sh` script
  pointing at `https://storage.googleapis.com/text-normalization/en_with_types.tgz`.
- Tried that GCS URL directly: got a real `AccessDenied` from Google itself
  (not our proxy) -- the dataset's bucket is no longer public. Separately
  confirmed kaggle.com (which mirrors the same data) IS blocked by this
  sandbox's org policy (`connect_rejected`) -- two different failure causes,
  now both pinned down precisely instead of lumped together.
- Searched for alternative sources, found `english-text-normalization` on
  PyPI (pypi.org is always reachable) -- a real, independently-written,
  MIT-licensed English TTS text-normalization library with abbreviation,
  number, currency, date, ordinal, percent handling.
- Installed it, ran the same 23-case benchmark through its `general_pipeline`:
  12/23 (52%) exact match, vs. 6/23 (26%) for the hand-rolled regex stand-in.
  Noted several of its "failures" are actually just alternate valid phrasings
  (e.g. "2024" -> "twenty twenty-four"), meaning exact-match string equality
  understates real usability -- flagged this limitation explicitly.
- Updated `scripts/benchmark_correct.py` to score both normalizers side by
  side, regenerated `outputs/correction_benchmark.txt`, and rewrote the
  relevant section of `docs/ARCHITECTURE.md` with the corrected, more
  precise picture (what's genuinely gone vs. what's sandbox-blocked, and
  the second independent accuracy data point).
