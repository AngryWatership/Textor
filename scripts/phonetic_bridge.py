"""
Phonetic bridge between text and sound.

Instead of asking an LLM to guess corrections (unreliable at small sizes --
see session.md / benchmark results), this represents words as phoneme
sequences and compares those, so "bash kit" and "basket" are recognized as
close even though the text strings are not.

Two phoneme sources, both fully offline:
- English, word-level: CMUdict (ARPABET phonemes), bundled in the `cmudict`
  pip package -- 126k entries, no network call.
- Any word/phrase, 100+ languages: espeak-ng's --ipa output. This is the
  same TTS engine already used in tts_pipeline.py, so the correction stage
  and the synthesis stage share one phoneme inventory.
"""

import subprocess
from functools import lru_cache

import cmudict

_CMUDICT = cmudict.dict()

# ARPABET has no single-character stress marks; strip stress digits (0/1/2)
# so "AE1" and "AE0" compare equal -- stress doesn't change which word it is.
def _strip_stress(phones: list[str]) -> list[str]:
    return [p.rstrip("012") for p in phones]


def arpabet_phones(word: str) -> list[str] | None:
    """CMUdict ARPABET phonemes for an English word, or None if out of vocab."""
    entries = _CMUDICT.get(word.lower())
    return _strip_stress(entries[0]) if entries else None


@lru_cache(maxsize=4096)
def ipa_phones(text: str, lang: str = "en-us") -> str:
    """IPA transcription for any text, in any of espeak-ng's 100+ languages."""
    result = subprocess.run(
        ["espeak-ng", "-v", lang, "--ipa", "-q", text],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def phones_for_phrase(words: list[str]) -> list[str] | None:
    """ARPABET phonemes for a sequence of English words, concatenated.

    Returns None if any word is out of CMUdict's vocabulary (caller should
    fall back to ipa_phones for OOV spans).
    """
    out: list[str] = []
    for w in words:
        p = arpabet_phones(w)
        if p is None:
            return None
        out.extend(p)
    return out


def phoneme_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over phoneme sequences (substitution cost 1)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def phonetic_similarity(phrase_a: str, phrase_b: str, lang: str = "en-us") -> float:
    """1.0 = identical pronunciation, 0.0 = maximally different.

    Tries CMUdict (English word-level, exact ARPABET) first; falls back to
    espeak-ng IPA (any language, including phrases CMUdict doesn't cover)
    compared character-by-character.
    """
    words_a, words_b = phrase_a.split(), phrase_b.split()
    pa, pb = phones_for_phrase(words_a), phones_for_phrase(words_b)
    if pa is not None and pb is not None:
        dist = phoneme_edit_distance(pa, pb)
        return 1 - dist / max(len(pa), len(pb), 1)

    ipa_a, ipa_b = ipa_phones(phrase_a, lang), ipa_phones(phrase_b, lang)
    dist = phoneme_edit_distance(list(ipa_a.replace(" ", "")), list(ipa_b.replace(" ", "")))
    return 1 - dist / max(len(ipa_a), len(ipa_b), 1)


if __name__ == "__main__":
    pairs = [
        ("basket", "bash kit"),
        ("two eggs", "to edge"),
        ("weather", "whether"),
        ("saw a bear", "saw a bare"),
    ]
    for a, b in pairs:
        sim = phonetic_similarity(a, b)
        print(f"{a!r} vs {b!r}: phonetic similarity = {sim:.2f}")

    print()
    print("Multi-language IPA (same function, espeak-ng voice per language):")
    for lang, text in [
        ("fr", "panier"),
        ("ar", "سلة"),
        ("de", "Korb"),
        ("es", "canasta"),
    ]:
        print(f"  [{lang}] {text!r} -> {ipa_phones(text, lang)!r}")
