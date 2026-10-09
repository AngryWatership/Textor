"""
Split audio into word/phrase clumps by silence, so Allosaurus's phoneme
stream can be matched against dictionary phoneme-chains per clump instead
of as one long run. Short silences (brief pauses between words) still
split further than ideal -- a "clump" here is typically a few words, not
exactly one -- but it's a much smaller unit than the whole utterance.

Uses ffmpeg's silencedetect filter (already a dependency via tts_pipeline/
dictate.py's resampling step) rather than pulling in a VAD library.
"""

import re
import subprocess
import wave
from pathlib import Path

_SILENCE_START = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)")


def _wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def detect_silences(wav_path: Path, noise_db: int = -30, min_silence_s: float = 0.3) -> list[tuple[float, float]]:
    """[(silence_start, silence_end), ...] in seconds."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(wav_path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m.group(1)) for m in _SILENCE_START.finditer(result.stderr)]
    ends = [float(m.group(1)) for m in _SILENCE_END.finditer(result.stderr)]
    return list(zip(starts, ends))


def non_silent_spans(wav_path: Path, noise_db: int = -30, min_silence_s: float = 0.3,
                      min_span_s: float = 0.15) -> list[tuple[float, float]]:
    """[(speech_start, speech_end), ...] -- the complement of the silences,
    i.e. the word/phrase clumps. Drops spans shorter than min_span_s
    (too short to contain a usable phoneme sequence)."""
    silences = detect_silences(wav_path, noise_db, min_silence_s)
    duration = _wav_duration(wav_path)

    spans = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start > cursor:
            spans.append((cursor, s_start))
        cursor = s_end
    if cursor < duration:
        spans.append((cursor, duration))
    return [(a, b) for a, b in spans if b - a >= min_span_s]


def extract_clip(wav_path: Path, start: float, end: float, out_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
         "-ss", str(start), "-to", str(end), str(out_path)],
        check=True,
    )
    return out_path


if __name__ == "__main__":
    import sys

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/voice_memo_16k.wav")
    spans = non_silent_spans(wav)
    print(f"{len(spans)} speech clumps found:")
    for start, end in spans:
        print(f"  {start:.2f}s - {end:.2f}s  ({end - start:.2f}s)")
