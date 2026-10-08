# Textor: Android Text Editor with TTS — Architecture Reference

## 0. Answering the original question

Is it trivial? No, but it's a well-trodden combination, not a research problem.
The three pieces (text editor, neural TTS, small on-device LLM for correction)
each exist as mature off-the-shelf components on Android. The actual
engineering risk isn't "can we get a voice to speak text," it's:

- keeping a 2B-parameter-class LLM and a TTS model both resident on a phone
  without killing battery/RAM,
- making the correction LLM's latency invisible (it must not make TTS feel
  laggy), and
- defining exactly what "correction" is allowed to change, so it never
  alters meaning.

That's what this doc plans for.

## 0.1 Is the current implementation usable? (added after accuracy check)

**No — the code in this repo is a pipeline prototype, not a usable product.**
It proves the *shape* of the architecture works; it is not a preview of the
quality users would get. Concretely:

- The TTS engine is `espeak-ng`, a 1990s-style formant synthesizer used only
  because real neural voices (Piper) are unreachable from this sandbox.
  It is intelligible but robotic — nowhere near what you'd ship.
- The "correction LLM" is 60 lines of regex (`scripts/correct.py`), not a
  model. I benchmarked it against 23 hand-built test sentences covering the
  categories any TTS normalizer must handle (abbreviations, numbers, dates,
  decimals, currency, ordinals, acronyms, time, percent) — see
  `scripts/benchmark_correct.py` and `outputs/correction_benchmark.txt`.

  **Result: 6/23 exact matches (26%).** It does the two things it was built
  to do (small numbers 0–9, the 7 abbreviations in its hardcoded dict) and
  fails almost everything else a real normalizer needs:
  - any number over 2 digits is left as digits ("2024 reasons" stays
    "2024 reasons" instead of "two thousand twenty-four")
  - decimals, currency, dates, times, ordinals, and percents are not
    handled at all ("$9.99" → "nine dollars and ninety-nine cents" never
    happens; it becomes "$nine.99")
  - abbreviation expansion is naive string substitution with no word-boundary
    or context check, so it also mangles things like "St." when it means
    "Saint" vs. "Street" in the same sentence, and silently does nothing for
    any abbreviation not in its small dict (Prof., Fig., etc.)

  This number is exactly what you'd expect from a hand-rolled regex script
  standing in for an LLM — it's a fair, reproducible baseline, not a
  surprise. The point of measuring it is to make the gap explicit instead of
  letting "it ran without errors" be mistaken for "it works."

- **Audio quality itself was not and could not be measured here.** There is
  no reference audio, no MOS (mean opinion score) model, and no human
  listener in this sandbox — only text-level checks. Judging whether Piper
  (or any real voice) sounds natural requires either a real device/human
  listening pass, or an automated speech-quality metric (e.g. UTMOS,
  NISQA) run against generated audio, neither of which is available in
  this container.

**What would make it usable:**
1. Swap `correct_for_speech()` for a real small LLM (§2.2) and re-run
   `benchmark_correct.py` (or a larger version of it) as the acceptance
   bar — target should be close to 100% on a benchmark like this, since
   these are the easy, unambiguous cases; real text will be harder.
2. Swap `espeak-ng` for Piper via sherpa-onnx (§2.1) and do a human listening
   pass, since voice naturalness isn't something text metrics can capture.
3. Treat the 26% number as the "before" baseline worth re-running this exact
   benchmark against once a real correction model is in, to know concretely
   how much it improved.

## 1. What was validated in this sandbox vs. what wasn't

This container's network policy blocks `huggingface.co` and `github.com`
(both return 403 at the proxy), which is where real neural TTS voices
(Piper) and small LLM weights (GGUF files) are hosted. So:

**Validated here** (`scripts/`, see `outputs/validation_report.md` for
results and `outputs/samples/*.wav` for audio):
- The pipeline *shape*: raw text → correction stage → TTS stage → audio,
  as two swappable, independently testable functions.
- `correct_for_speech()` — a deterministic normalizer (abbreviations,
  small numbers, whitespace) standing in for the correction LLM, same
  input/output contract a real model would have.
- `espeak-ng` as a stand-in TTS backend (offline, apt-installed) — proves
  the synthesis call, file I/O, and failure modes (crashes, empty output,
  truncation on long input) are all clean.
- 5 sample sentences covering plain text, abbreviations, numbers, heavy
  punctuation, and a long paragraph — all synthesized without error.

**Not validated here** (needs a real device or an environment with model
access):
- Actual voice quality/naturalness (espeak-ng is a 1990s-style formant
  synthesizer, not representative of Piper's neural voices).
- The correction *LLM* itself — only a rule-based stand-in with the same
  contract was tested. A real small LLM must be swapped in and re-validated
  against the same contract (see §5).
- On-device latency, memory, and battery behavior.
- Android-specific integration (AudioTrack playback, foreground service
  behavior, OS TTS framework hooks).

## 2. Recommended production architecture

```
 ┌─────────────┐    text      ┌───────────────┐   corrected   ┌───────────────┐   audio   ┌──────────┐
 │ Text editor │ ───────────▶ │ Correction LLM │ ────────────▶ │ Primary TTS    │ ────────▶ │ AudioTrack│
 │ (EditText)  │              │ (small, local) │               │ model (Piper)  │           │ playback  │
 └─────────────┘              └───────────────┘               └───────────────┘           └──────────┘
```

### 2.1 Primary TTS: Piper (via sherpa-onnx or piper's own Android build)

- Piper is a small, fast, on-device neural TTS (VITS-family), ONNX-based,
  ships per-voice models in the 20-60 MB range (`medium` quality tier is the
  right default for mobile).
- Two integration paths on Android:
  1. **sherpa-onnx** (k2-fsa) — has first-class Android bindings (AAR +
     JNI), supports Piper voices directly, actively maintained, easiest
     path. **Recommended.**
  2. Raw ONNX Runtime Mobile + hand-rolled Piper inference — more control,
     much more integration work. Only worth it if sherpa-onnx's API doesn't
     fit.
- Fallback option: Android's built-in `TextToSpeech` system API (any
  installed engine, e.g. Google TTS). Zero integration cost, no model to
  ship, but quality/voice selection is out of your control and it may not
  be available offline on all devices. Good as a day-1 fallback while
  Piper integration is being built, and as a permanent fallback on very
  low-end devices.

### 2.2 Correction LLM: a small instruct model via llama.cpp or MediaPipe LLM Inference

- Candidates (all runnable quantized on a modern phone): **Gemma-2 2B**,
  **Phi-3-mini (3.8B, 4-bit)**, **SmolLM2-1.7B**, **Qwen2.5-1.5B-Instruct**.
  Start with the smallest (SmolLM2-1.7B or Gemma-2 2B at Q4) — correction is
  a narrow task, it doesn't need a big model, and size directly trades off
  against load time and RAM.
- Runtime: **llama.cpp** via its Android JNI bindings (or **MediaPipe LLM
  Inference API**, which wraps llama.cpp/XNNPACK and is Google's blessed
  path for on-device Gemma). MediaPipe is the lower-effort integration if
  you're happy pinned to its supported model list; llama.cpp gives you
  model choice freedom.
- **Scope the correction task narrowly, and enforce it with the prompt and
  with output validation, not just instructions:**
  - allowed: fix typos, expand abbreviations/numbers for speech, normalize
    whitespace/punctuation, insert light prosody cues.
  - forbidden: paraphrasing, summarizing, adding/removing content, changing
    tone.
  - validate output isn't wildly longer/shorter than input (e.g. reject and
    fall back to raw text if length ratio is outside ~0.7–1.5x) as a cheap
    safety net against the model going off-script.

### 2.3 Why two separate models instead of one

A single larger LLM "that also does TTS" isn't a realistic mobile option
today — audio-generating LLMs are large and slow. Keeping correction and
synthesis as separate, swappable stages (as prototyped here) means:
- each can be upgraded/replaced independently,
- correction can be skipped entirely for short/already-clean text to save
  latency and battery,
- failure in one stage (e.g. correction LLM crashes or times out) degrades
  gracefully to raw-text-to-TTS instead of killing speech entirely.

## 3. Data flow and failure handling

1. User selects text or starts "read aloud" from cursor.
2. Text is chunked (sentence-level, not whole-document) so speech can start
   fast and so the correction LLM's context stays small.
3. Each chunk → correction LLM → validated output (length-ratio check) →
   on acceptance, feed to TTS; on rejection/timeout, feed the raw chunk to
   TTS directly.
4. TTS chunk → PCM → `AudioTrack` streaming playback, next chunk's
   correction runs concurrently with current chunk's playback (pipeline,
   not strict sequential) to hide correction latency.
5. Expose a user setting to disable the correction stage entirely
   (power users, or low-RAM devices) — it's a quality enhancement, not a
   dependency of core TTS.

## 4. Non-trivial engineering risks, ranked

1. **Model residency/RAM**: running an ONNX TTS model and a quantized LLM
   concurrently on a phone. Budget and test on a low-end target device
   (e.g. 4GB RAM), not just a flagship.
2. **Latency hiding**: correction must not add perceptible delay before
   speech starts. The chunk-pipeline in §3 is the mitigation; measure
   time-to-first-audio as a tracked metric from day one.
3. **Correction scope creep**: an LLM asked to "fix text for speech" will
   drift into rewriting/summarizing if the prompt and validation aren't
   tight. This is a correctness bug, not a polish item — test for it
   explicitly (see §5).
4. **Battery**: LLM inference is the expensive part. Measure
   watt-hours/1000-words-corrected on target hardware before committing to
   "correction always on" as a default.
5. **Model licensing**: check the license of whichever LLM/voice you ship
   (Gemma, Phi-3, Piper voices all have usable-but-different licenses) —
   do this before building the release pipeline around one, not after.

## 5. Test plan

### 5.1 Pipeline-shape tests (done in this sandbox, reusable as-is)
- `scripts/validate.py` — swap `correct_for_speech` and `synthesize_espeak`
  for real implementations; keep the same sample set (plain, abbreviations,
  numbers, punctuation-heavy, long-form) as a regression suite.
- Add: empty input, input that's only whitespace/punctuation, very long
  input (multi-paragraph), non-English characters/emoji (should not crash,
  behavior on mixed-script text should be an explicit decision, not an
  accident).

### 5.2 Correction LLM contract tests (must build once a real model is wired in)
- Length-ratio check (§2.2) as an automated test, not just a runtime guard.
- A fixed set of "trap" inputs designed to tempt rewriting/summarizing
  (e.g. a long opinionated paragraph) — assert the model doesn't change
  meaning. This likely needs a second LLM-as-judge pass or human review
  initially; don't rely on eyeballing a few examples.
- Timeout behavior: correction LLM must have a hard timeout with fallback
  to raw text, tested by actually forcing a timeout, not just code review.

### 5.3 TTS quality tests
- A/B voice quality: Piper output vs. Android's built-in TTS, on the same
  sentences, for a few human listeners — quality is subjective, don't skip
  the human step.
- Pronunciation edge cases: numbers, URLs, code snippets (this is a text
  *editor*, so code-like content is a realistic input) — decide and test
  explicit behavior (e.g. spell out identifiers, skip symbols) rather than
  letting the TTS engine guess.

### 5.4 Device/performance tests
- Time-to-first-audio, steady-state latency per chunk, RAM high-water mark,
  battery drain over a 30-minute continuous read — each on at least one
  low-end and one flagship device.
- Cold-start model load time (both models) — this is user-visible on first
  use after app install/update.

### 5.5 Integration tests
- Interruption handling: incoming call, another app grabbing audio focus,
  screen off, app backgrounded mid-read.
- Resume-from-position after interruption (don't restart from the top of
  the document).

## 6. Suggested build order

1. Ship Android `TextToSpeech` system API + the editor, no custom models —
   get the editor and read-aloud UX right first on a known-good TTS.
2. Swap in Piper via sherpa-onnx for primary TTS, A/B against system TTS.
3. Add the correction LLM as an optional, toggle-able enhancement, gated by
   the contract tests in §5.2 before it ships enabled by default.
4. Only after 1–3 are solid: tune for low-end devices and battery, per §5.4.

Building 2 and 3 before 1 is the classic way this kind of project stalls —
get a working, shippable baseline first.
