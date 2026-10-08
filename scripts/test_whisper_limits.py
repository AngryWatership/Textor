"""
Stress-test the uploaded Whisper tiny model to find where it actually
breaks, rather than assuming. Covers: accuracy on real recordings, latency/
real-time factor, technical jargon, noisy/degraded audio, and very short
or silent input.
"""

import subprocess
import time
import wave
from pathlib import Path

from dictate import transcribe, MODEL_PATH
from pywhispercpp.model import Model

DATA_DIR = Path(__file__).parent.parent / "data"
WAVS_DIR = DATA_DIR / "ljspeech_sample_wavs"
METADATA = DATA_DIR / "ljspeech_sample_metadata.csv"
TMP = Path("/tmp/whisper_limits")
TMP.mkdir(exist_ok=True)


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def synth(text: str, name: str, voice="en-us") -> Path:
    out = TMP / f"{name}.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-w", str(out), text], check=True, capture_output=True)
    return out


def word_match_rate(expected: str, actual: str) -> float:
    exp_words = expected.lower().replace(",", "").replace(".", "").split()
    act_words = actual.lower().replace(",", "").replace(".", "").split()
    matches = sum(1 for w in exp_words if w in act_words)
    return matches / max(len(exp_words), 1)


print("=== 1. Accuracy + latency on all 8 real LJSpeech recordings ===")
rows = []
with open(METADATA, encoding="utf-8") as f:
    for line in f:
        utt_id, raw_text, normalized = line.strip().split("|")
        wav_path = WAVS_DIR / f"{utt_id}.wav"
        if not wav_path.exists():
            continue
        duration = wav_duration(wav_path)
        actual, elapsed = transcribe(wav_path)
        rtf = elapsed / duration if duration else float("nan")
        match = word_match_rate(raw_text, actual)
        rows.append((utt_id, duration, elapsed, rtf, match))
        print(f"  {utt_id}: audio={duration:.1f}s transcribe={elapsed:.2f}s "
              f"RTF={rtf:.3f} word_match={match:.0%}")
avg_rtf = sum(r[3] for r in rows) / len(rows)
avg_match = sum(r[4] for r in rows) / len(rows)
print(f"  -> mean real-time factor: {avg_rtf:.3f} (lower is faster; <1.0 = faster than real-time)")
print(f"  -> mean word match rate: {avg_match:.0%}")
print()

print("=== 2. Technical jargon (synthesized via espeak-ng, not real speech) ===")
jargon_cases = [
    "I am using OAuth for authentication.",
    "This function needs to be idempotent.",
    "We store the config as JSON.",
    "The cluster runs on Kubernetes.",
    "The GPT and LLM APIs are rate limited.",
]
for text in jargon_cases:
    wav = synth(text, f"jargon_{abs(hash(text))}")
    actual, elapsed = transcribe(wav)
    match = word_match_rate(text, actual)
    print(f"  expected: {text!r}")
    print(f"  actual:   {actual!r}  (match={match:.0%}, {elapsed:.2f}s)")
print()

print("=== 3. Degraded audio: added noise ===")
clean = WAVS_DIR / "LJ001-0002.wav"
noisy = TMP / "LJ001-0002_noisy.wav"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clean),
     "-af", "highpass=f=200,lowpass=f=3000,volume=0.3",
     str(noisy)],
    check=True,
)
clean_text, _ = transcribe(clean)
noisy_text, _ = transcribe(noisy)
print(f"  clean: {clean_text!r}")
print(f"  degraded (filtered+quiet): {noisy_text!r}")
print()

print("=== 4. Edge cases: silence and very short clip ===")
silence = TMP / "silence.wav"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
     "-t", "3", str(silence)],
    check=True,
)
try:
    text, elapsed = transcribe(silence)
    print(f"  3s silence -> {text!r} ({elapsed:.2f}s)")
except Exception as e:
    print(f"  3s silence -> EXCEPTION: {e}")

short_word = synth("hi", "short_word")
try:
    text, elapsed = transcribe(short_word)
    print(f"  single word 'hi' -> {text!r} ({elapsed:.2f}s)")
except Exception as e:
    print(f"  single word 'hi' -> EXCEPTION: {e}")

print()
print("=== 5. Non-English input (model is multilingual by default) ===")
for lang, text in [("fr", "Bonjour, comment allez-vous?"), ("es", "Hola, como estas?")]:
    wav = synth(text, f"nonenglish_{lang}", voice=lang)
    try:
        actual, elapsed = transcribe(wav)
        print(f"  [{lang}] said: {text!r} -> whisper heard: {actual!r} ({elapsed:.2f}s)")
    except Exception as e:
        print(f"  [{lang}] EXCEPTION: {e}")
