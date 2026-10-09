"""
Evaluate Whisper and Allosaurus on the full 32-recording LJSpeech subset
already in this repo (data/ljspeech_sample_wavs/), not just the 8 we had
ground truth for before.

Ground truth for the other 24 came from NVIDIA/tacotron2's filelists
(github.com, reachable; data.keithito.com / openslr.org / commonvoice are
not -- confirmed by direct connection test). It's the raw transcript
column (digits as digits, e.g. "1455"), not the speech-normalized column
our original 8-sample metadata.csv has -- so word-match scoring on the one
entry with a date (LJ001-0007, spoken as "fourteen fifty-five") will show
a spurious mismatch; everything else is plain text with no such gap.
"""

import subprocess
import time
from pathlib import Path

from phoneme_recognize import recognize_phonemes
from phonetic_bridge import ipa_phones, phoneme_edit_distance
from pywhispercpp.model import Model

DATA_DIR = Path(__file__).parent.parent / "data"
WAVS_DIR = DATA_DIR / "ljspeech_sample_wavs"
TRANSCRIPTS = DATA_DIR / "ljspeech_32_transcripts.txt"
TMP = Path("/tmp/eval32")
TMP.mkdir(exist_ok=True)

_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = Model(str(Path(__file__).parent.parent / "models" / "ggml-tiny.bin"))
    return _whisper


def to_16k(wav_path: Path) -> Path:
    out = TMP / f"{wav_path.stem}_16k.wav"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ar", "16000", "-ac", "1", str(out)],
            check=True,
        )
    return out


def word_match_rate(expected: str, actual: str) -> float:
    exp = expected.lower().replace(",", "").replace(".", "").replace('"', "").split()
    act = actual.lower().replace(",", "").replace(".", "").replace('"', "").split()
    return sum(1 for w in exp if w in act) / max(len(exp), 1)


def main():
    rows = []
    with open(TRANSCRIPTS, encoding="utf-8") as f:
        for line in f:
            utt_id, text = line.rstrip("\n").split("|", 1)
            rows.append((utt_id, text))

    whisper_scores, allosaurus_scores = [], []
    for utt_id, text in rows:
        wav = WAVS_DIR / f"{utt_id}.wav"
        wav16 = to_16k(wav)

        t0 = time.time()
        segs = get_whisper().transcribe(str(wav16))
        w_time = time.time() - t0
        w_text = " ".join(s.text.strip() for s in segs)
        w_match = word_match_rate(text, w_text)

        t0 = time.time()
        heard = recognize_phonemes(str(wav16)).replace(" ", "")
        a_time = time.time() - t0
        expected_ipa = ipa_phones(text).replace(" ", "")
        dist = phoneme_edit_distance(list(expected_ipa), list(heard))
        a_sim = 1 - dist / max(len(expected_ipa), len(heard), 1)

        whisper_scores.append(w_match)
        allosaurus_scores.append(a_sim)
        print(f"{utt_id}: whisper_word_match={w_match:.0%} ({w_time:.2f}s)  "
              f"allosaurus_phonetic_sim={a_sim:.2f} ({a_time:.2f}s)")

    print()
    print(f"Whisper mean word match (n={len(rows)}): {sum(whisper_scores)/len(whisper_scores):.1%}")
    print(f"Allosaurus mean phonetic sim (n={len(rows)}): {sum(allosaurus_scores)/len(allosaurus_scores):.2f}")


if __name__ == "__main__":
    main()
