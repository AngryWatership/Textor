"""
Cross-check Whisper's text against Allosaurus's independent audio-grounded
phoneme stream, to catch misrecognitions text-only correction can't.

Whisper's output is a guess biased by its training-data vocabulary (it
wrote "enzymes" for a real dictated "gems" -- see outputs/whisper_limits_
analysis.md). Allosaurus has no word-vocabulary bias; it reports phones
from the raw acoustic signal. For each Whisper segment, render Whisper's
own text to phonemes (what its words WOULD sound like) and compare that to
what Allosaurus actually heard in the same time window (what was ACTUALLY
spoken). Large disagreement flags the segment for the user to re-check.
"""

from pathlib import Path

from phoneme_recognize import recognize_phonemes_timestamped
from phonetic_bridge import ipa_phones, phoneme_edit_distance
from pywhispercpp.model import Model

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "ggml-tiny.bin")
_whisper = None


def _get_whisper() -> Model:
    global _whisper
    if _whisper is None:
        _whisper = Model(MODEL_PATH)
    return _whisper


def _phones_in_window(timestamped: list[tuple[float, float, str]], start: float, end: float) -> str:
    return " ".join(p for t, _, p in timestamped if start <= t < end)


def cross_check(wav_16k_path: Path, agreement_threshold: float = 0.5) -> list[dict]:
    """Returns one entry per Whisper segment: text, heard phonemes, agreement."""
    segments = _get_whisper().transcribe(str(wav_16k_path))
    timestamped = recognize_phonemes_timestamped(wav_16k_path)

    results = []
    for seg in segments:
        start_s, end_s = seg.t0 / 100, seg.t1 / 100
        heard = _phones_in_window(timestamped, start_s, end_s)
        expected = ipa_phones(seg.text).replace(" ", "")
        heard_compact = heard.replace(" ", "")
        dist = phoneme_edit_distance(list(expected), list(heard_compact))
        agreement = 1 - dist / max(len(expected), len(heard_compact), 1)
        results.append({
            "start": start_s,
            "end": end_s,
            "whisper_text": seg.text,
            "whisper_text_as_phones": ipa_phones(seg.text),
            "allosaurus_heard": heard,
            "agreement": round(agreement, 2),
            "flagged": agreement < agreement_threshold,
        })
    return results


if __name__ == "__main__":
    import sys

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/voice_memo_16k.wav")
    for r in cross_check(wav):
        mark = "FLAGGED" if r["flagged"] else "ok"
        print(f"[{mark}] {r['start']:.1f}s-{r['end']:.1f}s agreement={r['agreement']:.2f}")
        print(f"  whisper text:        {r['whisper_text']!r}")
        print(f"  whisper text->IPA:   {r['whisper_text_as_phones']!r}")
        print(f"  allosaurus heard:    {r['allosaurus_heard']!r}")
        print()
