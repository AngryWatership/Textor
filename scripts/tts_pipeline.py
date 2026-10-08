"""
End-to-end TTS pipeline prototype: raw text -> correction stage -> TTS engine -> wav.

TTS engine here is espeak-ng (offline, installed via apt), standing in for a
real neural voice (Piper, on Android) that this sandbox's network policy
blocks fetching (huggingface.co / github.com denied -- see session.md).
The pipeline shape (correction stage as a pluggable function, TTS engine as
a pluggable backend) is the same shape the real Android implementation uses.
"""

import subprocess
from pathlib import Path

from correct import correct_for_speech


def synthesize_espeak(text: str, out_wav: Path, voice: str = "en-us", speed_wpm: int = 160) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(speed_wpm), "-w", str(out_wav), text],
        check=True,
        capture_output=True,
    )


def run_pipeline(raw_text: str, out_wav: Path) -> str:
    """Returns the corrected text that was actually spoken."""
    corrected = correct_for_speech(raw_text)
    synthesize_espeak(corrected, out_wav)
    return corrected


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test."
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/samples/test.wav")
    spoken = run_pipeline(text, out)
    print(f"wrote {out}")
    print(f"spoke: {spoken}")
