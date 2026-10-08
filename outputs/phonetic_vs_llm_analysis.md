# Correction-stage architecture: phonetic approach vs. on-device LLM

Target use case (confirmed with the user): a lightweight Android app that is
both a plain-text editor and a TTS reader, where the correction stage's job
is **not** abbreviation/number expansion -- it's fixing words that sound
similar to what was meant but break the sentence (dictation/ASR-style
misrecognition errors, e.g. "I've got two eggs in the basket" coming out as
"I've got to get in the bash kit"), including technical jargon
(OAuth, LLM, GPT, idempotent, JSON, Kubernetes).

## 1. What was tried, in order, and why each step happened

| # | Approach | Result | Why it was abandoned/kept |
|---|---|---|---|
| 1 | Regex stand-in (`correct.py`), abbreviation+number expansion | 26.1% on `benchmark_correct.py`'s 23 cases | Wrong task once the real goal (homophone correction) was clarified -- kept only as reference, not pursued further |
| 2 | `english_text_normalization` pip library, same task | Noticeably higher than #1, 100% on the LJSpeech ground-truth subset | Same wrong-task issue; a real alternative if normalization is ever needed later |
| 3 | SmolLM2-135M-Instruct (bundled GGUF, `pip install llm-smollm2`), direct generation ("rewrite this corrected") | 0% -- it echoed the input verbatim on every prompt, zero-shot and few-shot alike | Model too small to do context-dependent semantic correction at all |
| 4 | Same model, chained 10x (output fed back as next input, temperature 0.8) | Degenerated into repetition loops, never reached the correct word, text ballooned 10x | Self-feedback doesn't help a model that has no mechanism to search for the right answer -- it just hallucinates more around the same error |
| 5 | Same model, asked to translate (FR/ES/DE) | Fluent-looking output, semantically wrong or unrelated every time | Confirms 135M params is well below where real multilingual competence emerges |
| 6 | Fine-tuning (LoRA) on the model we have | Not possible | We only have the Q4_1 GGUF; llama.cpp removed GGUF-native training in 2024, and real LoRA training needs the original HF safetensors checkpoint, which requires huggingface.co (network-policy blocked) |
| 7 | **Phonetic bridge + nearest-neighbor correction** (this document's subject) | 50% exact-match on a 10-case homophone benchmark; clean separation of failure modes (see below) | First approach that actually fixes the target example, with honest, explainable failure modes instead of silent hallucination |

## 2. The phonetic approach

Two data sources, both fully offline, already in the repo's dependency footprint:

- **CMUdict** (`pip install cmudict`): 126,052 English words -> ARPABET
  phonemes, bundled directly in the wheel. Word-level, exact.
- **espeak-ng --ipa**: the same TTS engine `tts_pipeline.py` already uses for
  synthesis, also gives IPA transcription for any text in 100+ languages
  (French, Arabic, German, Spanish verified working), and handles words
  CMUdict doesn't know (including made-up/technical terms) via its built-in
  letter-to-sound rules.

`scripts/phonetic_bridge.py` turns either source into a phoneme sequence and
scores two phrases by normalized phoneme edit distance (1.0 = identical
pronunciation). `scripts/phonetic_correct.py` uses that to scan each 1-2
word span of the input against a target vocabulary and substitute the
closest match, under two different thresholds depending on whether the
candidate is the same word-count (true homophone, require 1.0 similarity)
or fewer words (a word-boundary merge like "bash kit" -> "basket", a looser
0.65 threshold since collapsing words rarely happens by chance).

### Benchmark result: 5/10 (50%) on `scripts/benchmark_phonetic.py`

| Category | Passed | Notes |
|---|---|---|
| merge (e.g. "bash kit" -> "basket") | 2/2 | **The case that motivated this whole investigation now works.** |
| plain passthrough (no error present) | 2/2 | No false positives on clean input, for this test set |
| true homophone (e.g. "whether"/"weather") | 1/6 | See failure modes below |

### Failure modes found, and why they're informative rather than just bugs

1. **True-homophone direction is ambiguous from sound alone.** `to`/`two`/
   `too`, `their`/`there`/`they're`, `write`/`right` are phonetically
   *identical* -- phonetics has no way to know which spelling the sentence
   needs; that information lives in grammar (part of speech, sentence
   position), not pronunciation. Example: `"I need to right a letter with
   my write hand"` should become `"...write a letter with my right hand"`,
   but the corrector (which has no grammar model) instead "corrected" an
   already-correct `"to"` into `"two"` elsewhere in the sentence.
2. **Article-swallowing in the merge rule.** `"the basket"` sometimes
   collapses into just `"basket"` because adding the schwa of "the" barely
   changes the phoneme-distance score enough to clear the merge threshold.
   This is a tuning/precision issue in the matching rule, not a phonetics
   limitation per se.
3. **Long technical terms lose signal when split across words by
   character-level IPA comparison.** `kubernetes` vs a hypothetical ASR
   split like `"cooper netties"` only scores 0.46 similarity (see table
   below) -- char-level IPA comparison is cruder than ARPABET phoneme
   comparison and degrades on longer, multi-syllable words.

### Technical jargon: similarity scores (not run through the auto-corrector; see recommendation)

| Term | Garbled form | Similarity |
|---|---|---|
| OAuth | "o auth" | 1.00 |
| json | "jason" | 0.70 |
| idempotent | "I dont potent" | 0.61 |
| idempotent | "I dont potent etc" | 0.44 |
| kubernetes | "cooper netties" | 0.46 |

This matches the user's own real example mid-session: dictating "Idempotent
code which does only one task" came out as "I don't potent etc... I don't
potent code which does only one task" -- exactly the error class this
whole investigation is about, and exactly the regime (long technical term,
split across several shorter words) where the current scoring is weakest.

## 3. Recommendation for the Android app architecture

**Use the phonetic approach as the primary mechanism, not an LLM**, for three
concrete reasons specific to "lightweight Android app":

- **Size.** CMUdict + espeak-ng's phoneme data are tens of MB, already
  needed for TTS synthesis itself. The smallest LLM that even runs here
  (135M params, ~90MB quantized) still produced 0% correct generations and
  broke on chaining and translation; a model actually capable of this task
  is in the 1B+ range (~0.7-1.5GB quantized) -- a heavy dependency for a
  "lightweight" app, and still unverified since it couldn't be obtained in
  this sandbox.
- **Explainability and control.** The phonetic approach's failures are all
  legible and fixable (tune a threshold, add a grammar rule, improve the
  merge heuristic). The LLM's failures were opaque hallucination with no
  lever to pull except "try a bigger model."
- **Speed.** Phoneme comparison is milliseconds per word on CPU; no model
  inference latency, which matters for a responsive text editor.

**Concrete design implications:**

1. **Don't silently auto-correct true homophones or long technical terms.**
   Both failure modes above come from the app guessing without enough
   information. Instead, **flag** a span whose phonetic similarity to a
   known word crosses a lower threshold (e.g. 0.5-0.7) as "possibly
   misheard," and let the user confirm/pick -- same UX pattern as a
   spell-checker's red squiggle, not a silent rewrite. This sidesteps
   failure mode #1 and #3 entirely by not requiring the system to be
   certain.
2. **Keep silent auto-correction only for the merge case** ("bash kit" ->
   "basket"), which scored cleanly and has low false-positive risk, with a
   fix for the article-swallowing bug (exclude leading function words from
   merge candidates, or require the merge to consume the article
   explicitly as part of the matched span rather than incidentally).
3. **Seed the target vocabulary from the document being edited and a
   technical-terms list the user can extend** (OAuth, LLM, GPT, JSON,
   Kubernetes, idempotent, ...), not the full 126k-word CMUdict unconstrained
   -- nearest-neighbor search over the whole dictionary needs a word-frequency
   prior to avoid "correcting" uncommon-but-correct words into more common
   ones that merely sound similar; that prior isn't built yet.
4. **True-homophone disambiguation (to/two/too, their/there/they're,
   write/right) needs a small grammar/POS signal**, not more phonetics --
   e.g. a part-of-speech check (a verb follows "to", not "two") or a
   curated rule table for the dozen or so homophone sets that actually
   matter in English. This is a separate, smaller piece of work from the
   phonetic bridge itself.
5. **Keep `english_text_normalization` (pip) on the shelf** for number/date/
   currency expansion if that's ever needed for TTS output quality, since
   it scored well (100% on the real LJSpeech ground truth) and the user has
   deprioritized but not ruled it out.

## 4. Files produced this session

- `scripts/phonetic_bridge.py` -- phoneme representation + similarity scoring (CMUdict + espeak-ng IPA, multi-language)
- `scripts/phonetic_correct.py` -- homophone/merge correction using the bridge
- `scripts/benchmark_phonetic.py` -- the 10-case benchmark scored above
- `outputs/phonetic_vs_llm_analysis.md` -- this document
