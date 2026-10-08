"""
Dictation pipeline: audio -> Whisper (speech-to-text) -> phonetic correction
-> plain text file, open/save-able like any text editor document.

This is the actual target architecture confirmed with the user: not TTS
reading text aloud, but the reverse -- voice dictation into an editable
text file, the way the user is dictating these very session instructions.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

from pywhispercpp.model import Model

from phonetic_correct import correct_phrase, flag_jargon

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "ggml-tiny.bin")
_model = None


def _get_model() -> Model:
    global _model
    if _model is None:
        _model = Model(MODEL_PATH)
    return _model


def _ensure_16k_mono(wav_path: Path) -> Path:
    """Whisper requires 16kHz mono input; resample into a temp file if needed
    (never next to the source file -- that would pollute the dataset dir)."""
    fd, tmp_name = tempfile.mkstemp(suffix=".16k.wav")
    os.close(fd)
    out = Path(tmp_name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
    )
    return out


def transcribe(wav_path: Path) -> tuple[str, float]:
    """Returns (raw transcript, seconds taken)."""
    audio_16k = _ensure_16k_mono(wav_path)
    start = time.time()
    segments = _get_model().transcribe(str(audio_16k))
    elapsed = time.time() - start
    return " ".join(s.text.strip() for s in segments), elapsed


def _insert_jargon_flags(text: str, flags: list[dict]) -> str:
    """Inline-annotate flagged spans so a plain .txt file can carry the
    suggestion without needing a rich-text editor: "J-1 [possibly: JSON?]".
    The user confirms/dismisses by editing the text directly -- no silent
    auto-correction of a term the system isn't confident about (see
    outputs/whisper_limits_analysis.md on why OAuth/idempotent-type misses
    shouldn't be guessed at)."""
    words = text.split()
    for flag in sorted(flags, key=lambda f: f["word_index"], reverse=True):
        span_word_count = len(flag["span_text"].split())
        idx = flag["word_index"]
        words[idx + span_word_count - 1] += f' [possibly: {flag["suggestion"]}?]'
    return " ".join(words)


def dictate_to_file(wav_path: Path, out_txt: Path, append: bool = False) -> str:
    """Full pipeline: audio -> Whisper -> phonetic/jargon correction -> text file.

    This is the "save" half of the editor: write (or append to) a plain
    .txt file so it behaves like any other document the app can reopen.
    Homophone/merge errors (e.g. "bash kit" -> "basket") are auto-corrected;
    technical-jargon misses are flagged inline for the user to confirm,
    since those misses are often too phonetically divergent to guess at
    reliably (see outputs/whisper_limits_analysis.md).
    """
    raw, _ = transcribe(wav_path)
    corrected = correct_phrase(raw)
    flags = flag_jargon(corrected)
    annotated = _insert_jargon_flags(corrected, flags) if flags else corrected
    mode = "a" if append else "w"
    with open(out_txt, mode, encoding="utf-8") as f:
        if append and out_txt.exists() and out_txt.stat().st_size > 0:
            f.write(" ")
        f.write(annotated)
    return annotated


def open_text_file(path: Path) -> str:
    """The "open" half of the editor: read back a saved document."""
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import sys

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/ljspeech_sample_wavs/LJ001-0001.wav")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../outputs/dictation_test.txt")
    text = dictate_to_file(wav, out)
    print(f"wrote {out}:")
    print(open_text_file(out))
