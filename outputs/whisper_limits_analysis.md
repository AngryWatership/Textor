# Whisper tiny (GGML) dictation engine: where it works and where it breaks

Model: `models/ggml-tiny.bin` (77.11 MB, whisper.cpp tiny, assembled from the
7-volume split-zip upload -- see git history). Tested via `pywhispercpp` on
this sandbox's CPU (no GPU). Full pipeline: `scripts/dictate.py` (audio ->
Whisper -> `phonetic_correct.py` -> plain `.txt` file, open/save-able).
Stress test: `scripts/test_whisper_limits.py`, raw output in
`/tmp/whisper_limits_output.log` (not committed -- regenerate by rerunning
the script).

## 1. Clean real speech: strong

All 8 real LJSpeech recordings (`data/ljspeech_sample_wavs/`), human-recorded,
not synthesized:

| Metric | Result |
|---|---|
| Mean word match vs. ground truth | **96%** |
| Mean real-time factor (CPU) | **0.12** (transcribes ~8x faster than the audio's own duration) |
| Worst case | LJ001-0007, 88% word match |

This is the headline result: on real human speech, this 77MB model is both
accurate and fast enough for live dictation on CPU alone, no GPU needed.

## 2. Technical jargon: the weak point that matters most for this project

Synthesized via espeak-ng (not real human speech, so treat as a lower bound
on real difficulty -- a real voice would likely do somewhat better, but the
direction of the problem is real):

| Said | Heard | Word match |
|---|---|---|
| "I am using OAuth for authentication." | "I am UV No.1 for authentication." | 67% |
| "This function needs to be idempotent." | "This function leads to the iDou.com." | 50% |
| "We store the config as JSON." | "We store the config as J-1." | 83% |
| "The cluster runs on Kubernetes." | "The cluster run mark uber meets." | 40% |
| "The GPT and LLM APIs are rate limited." | "the PPT and LLM API my rate limited." | 62% |

This is exactly the error class the user hit for real mid-session
("idempotent" -> "I don't potent etc" while dictating this very
conversation) and exactly what `phonetic_correct.py` is meant to fix. Two
observations:

- The errors are phonetically plausible substitutions (not random noise) --
  "idempotent" and "iDou.com" share real phonetic overlap, same for
  "Kubernetes"/"uber meets" -- so the phonetic-bridge approach from the
  earlier analysis is targeting the right kind of error.
- Whisper tiny (the smallest variant) is clearly undertrained on this
  vocabulary. A larger Whisper variant (base/small) would likely do better
  on jargon specifically, at the cost of size/speed -- worth testing if/when
  one becomes available, but not re-attempted here since it needs the same
  blocked network path as every other larger model this session tried.

## 3. Degraded audio: robust, on a mild test

Band-pass filtered (200Hz-3kHz) and quieted (30% volume) version of a clean
recording transcribed identically to the clean original. This is a fairly
mild degradation, though (phone-call-like filtering, not heavy background
noise) -- not a claim that the model is noise-robust in general, just that
it isn't fragile to this specific, realistic kind of signal loss.

## 4. Edge cases

- **Silence (3s):** correctly outputs `[BLANK_AUDIO]` rather than
  hallucinating text. Good -- this is the behavior a text editor needs
  (no dictation = no garbage inserted).
- **Single short word ("hi"):** misheard as "Bye." -- very short utterances
  are unreliable. A real dictation UI should probably require a minimum
  utterance length, or buffer a few words before transcribing, rather than
  transcribing word-by-word in real time.

## 5. Non-English: not usable without explicit configuration

Called without specifying a target language (relying on auto-detect):
French and Spanish input produced nonsense ("Bonjour, comment allez-vous?"
-> "Mu Zhu, come on in.") or a generic fallback tag. This is very likely an
auto-detection/configuration issue rather than a fundamental model
limitation (Whisper is trained multilingual) -- `pywhispercpp`/whisper.cpp
needs the language explicitly set (or language-detection enabled and
trusted) for non-English input; this wasn't tuned in this test. **Treat
this as unverified, not as "multilingual doesn't work"** -- it needs a
follow-up test with the language parameter set explicitly before drawing
any real conclusion.

## Implications for the dictation app

1. **Core dictation of plain English speech is solid today** -- fast enough
   for live use on CPU, accurate enough on real speech (96%).
2. **Technical jargon is the real gap**, and it's the gap the user actually
   cares about. The phonetic-correction stage (`phonetic_correct.py`) is
   the right tool for this, but its target vocabulary needs to include the
   user's actual technical terms (OAuth, idempotent, Kubernetes, JSON,
   GPT/LLM, ...) since CMUdict doesn't cover them (confirmed in the earlier
   analysis) and Whisper tiny mishears them in phonetically-plausible ways
   that the bridge's similarity scoring should be able to catch.
3. **Don't transcribe on every keystroke/short pause** -- buffer enough
   audio (a clause or sentence, not a single word) before sending it to
   Whisper, since very short utterances are unreliable.
4. **Silence handling is already correct** (`[BLANK_AUDIO]`, no
   hallucination) -- no extra work needed there.
5. **Non-English dictation needs a follow-up test** with the language
   explicitly set before concluding anything -- don't build on today's
   negative result as if it were final.
