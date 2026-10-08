"""
Evaluate the correction stage against a real, human-recorded speech corpus.

Data: `data/ljspeech_sample_metadata.csv` + `data/ljspeech_sample_wavs/` -- an
8-sentence subset of LJSpeech (public-domain, LibriVox-recorded readings of
public-domain non-fiction texts), pulled from coqui-ai/TTS's own test
fixtures (tests/data/ljspeech in that repo). This is the real dataset
format: `id|raw_text|normalized_text`, where `normalized_text` is the
original LJSpeech authors' ground-truth spoken form -- e.g. "1455" is
annotated as "fourteen fifty-five". That makes it a genuine, independently
produced reference for what "correct" looks like, not something built for
this task, and it comes with real recorded audio of a human actually
speaking the normalized form, which text-only benchmarks don't have.

This script:
1. Scores both correction normalizers (this repo's regex stand-in, and the
   `english-text-normalization` pip library) against `raw_text ->
   normalized_text`, same methodology as scripts/benchmark_correct.py.
2. Runs each normalizer's output through the TTS stand-in (espeak-ng) and
   measures synthesized duration against the real recording's duration, as
   a cheap (not a real quality metric) sanity check that output isn't
   wildly truncated or bloated relative to real human speech.
"""

import csv
import wave
from pathlib import Path

from correct import correct_for_speech
from tts_pipeline import synthesize_espeak

try:
    from english_text_normalization.normalization_pipeline import general_pipeline
except ImportError:
    general_pipeline = None

DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_PATH = DATA_DIR / "ljspeech_sample_metadata.csv"
WAVS_DIR = DATA_DIR / "ljspeech_sample_wavs"
OUT_DIR = Path(__file__).parent.parent / "outputs" / "corpus_eval"


def load_corpus():
    rows = []
    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 3:
                utt_id, raw_text, normalized_text = row[0], row[1], row[2]
                rows.append((utt_id, raw_text, normalized_text))
    return rows


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def run_normalizer_eval(corpus, normalizer, label: str):
    print(f"=== {label}: text normalization accuracy vs. ground truth ===")
    passed = 0
    for utt_id, raw_text, normalized_text in corpus:
        try:
            actual = normalizer(raw_text)
        except Exception as exc:  # noqa: BLE001
            actual = f"ERROR: {exc}"
        ok = actual.strip() == normalized_text.strip()
        passed += ok
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {utt_id}")
        if not ok:
            print(f"      raw:        {raw_text!r}")
            print(f"      ground truth: {normalized_text!r}")
            print(f"      got:        {actual!r}")
    total = len(corpus)
    print(f"  -> {passed}/{total} = {passed/total:.1%}")
    print()
    return passed, total


def run_audio_duration_check(corpus, normalizer, label: str):
    out_dir = OUT_DIR / label.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {label}: synthesized vs. real recording duration ===")
    ratios = []
    for utt_id, raw_text, _normalized_text in corpus:
        real_wav = WAVS_DIR / f"{utt_id}.wav"
        if not real_wav.exists():
            continue
        real_duration = wav_duration_seconds(real_wav)
        try:
            corrected = normalizer(raw_text)
        except Exception:
            corrected = raw_text
        synth_wav = out_dir / f"{utt_id}.wav"
        synthesize_espeak(corrected, synth_wav)
        synth_duration = wav_duration_seconds(synth_wav)
        ratio = synth_duration / real_duration if real_duration else float("nan")
        ratios.append(ratio)
        print(f"  {utt_id}: real={real_duration:.2f}s synth={synth_duration:.2f}s ratio={ratio:.2f}")
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"  -> mean synth/real duration ratio: {avg:.2f}")
    print()


def main() -> None:
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} real recorded utterances with ground-truth normalization.\n")

    run_normalizer_eval(corpus, correct_for_speech, "rule-based stand-in (scripts/correct.py)")
    if general_pipeline is not None:
        run_normalizer_eval(corpus, general_pipeline, "english-text-normalization (real pip library)")

    run_audio_duration_check(corpus, correct_for_speech, "rule_based_stand_in")
    if general_pipeline is not None:
        run_audio_duration_check(corpus, general_pipeline, "english_text_normalization")


if __name__ == "__main__":
    main()
