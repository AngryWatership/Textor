"""
Audio -> phoneme stream via Allosaurus (real speech-to-phonetics, not
text-to-phonetics like phonetic_bridge.ipa_phones).

This is the missing piece requested when researching whether a
speech-to-phonetics model exists: Allosaurus (Xu, Baevski & Auli, ICASSP
2020) is pretrained, covers 2000+ languages, and its weights come from a
GitHub release asset -- reachable in this sandbox, unlike huggingface.co or
Azure, so no network-policy workaround was needed.

Why this matters for the pipeline: Whisper's text output is already a
*guess* -- when it mishears "gems" as "enzymes", the word "enzymes" it
wrote down is not the actual sound that was spoken, it's whatever word
Whisper's language-model prior decided was most likely. Allosaurus instead
reports what was acoustically present, with no language-model vocabulary
bias pulling it toward real dictionary words. Comparing the two gives a
second, independent, audio-grounded signal: if Whisper's word and the
phonemes Allosaurus actually heard for that time span disagree, that's a
much stronger misrecognition signal than text-only phonetic similarity
between Whisper's own output and a vocabulary list.
"""

from pathlib import Path

from allosaurus.app import read_recognizer

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = read_recognizer()
    return _model


def recognize_phonemes(wav_path: Path) -> str:
    """Space-separated IPA phone sequence for the whole clip, no timestamps."""
    return _get_model().recognize(str(wav_path))


def recognize_phonemes_timestamped(wav_path: Path) -> list[tuple[float, float, str]]:
    """List of (start_sec, duration_sec, phone)."""
    raw = _get_model().recognize(str(wav_path), timestamp=True)
    out = []
    for line in raw.strip().splitlines():
        start, dur, phone = line.split(maxsplit=2)
        out.append((float(start), float(dur), phone))
    return out


if __name__ == "__main__":
    import sys

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/whisper_assemble/LJ001-0001_16k.wav")
    print("phoneme stream:", recognize_phonemes(wav))
