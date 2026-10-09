"""
Phonemes -> word(s): the missing direction to complete the loop started by
phoneme_recognize.py (audio -> phonemes). Research found no pretrained,
pip-installable phoneme-to-grapheme model, and WikiPron's premined
pronunciation data isn't bundled in its repo (it scrapes Wiktionary live,
which this sandbox's network policy blocks same as huggingface.co). So
this uses the "massive dictionary" option instead: CMUdict's 126k English
words, reverse-indexed by pronunciation, nearest-neighbor matched against
whatever phonemes come in (from Allosaurus's real audio-grounded output,
or anything else).

ARPABET<->IPA mapping is a static table, not an espeak-ng subprocess call
per word -- 126k subprocess calls would take too long. This is standard,
well-documented (CMUdict's own ARPABET phone set maps 1:1 to one IPA
symbol each, no context-dependent allophony needed at this level).
"""

from functools import lru_cache

import cmudict

from phonetic_bridge import phoneme_edit_distance

_CMUDICT = cmudict.dict()

# Standard CMU ARPABET -> IPA. Stress digits already stripped elsewhere.
_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "g", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def _strip_stress(phones: list[str]) -> list[str]:
    return [p.rstrip("012") for p in phones]


@lru_cache(maxsize=1)
def _build_reverse_index() -> dict[str, list[str]]:
    """word's IPA-ish string -> [words with that pronunciation]. Built once,
    cached -- 126k entries, pure in-memory table lookups, no subprocess
    calls, so this is fast despite the dictionary size."""
    index: dict[str, list[str]] = {}
    for word, entries in _CMUDICT.items():
        arpabet = _strip_stress(entries[0])
        ipa = "".join(_ARPABET_TO_IPA.get(p, "?") for p in arpabet)
        index.setdefault(ipa, []).append(word)
    return index


def phonemes_to_words(phoneme_string: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Nearest CMUdict word(s) to a raw phoneme string (IPA, space-separated
    or not -- spaces are stripped). Returns [(word, similarity), ...],
    best first. This is a brute-force nearest-neighbor scan over all 126k
    entries -- fine for interactive use (a few seconds), not for
    real-time per-frame decoding without a smarter index (e.g. a trie)."""
    query = phoneme_string.replace(" ", "")
    index = _build_reverse_index()

    scored: list[tuple[str, float]] = []
    seen_words: set[str] = set()
    for ipa, words in index.items():
        dist = phoneme_edit_distance(list(query), list(ipa))
        sim = 1 - dist / max(len(query), len(ipa), 1)
        for w in words:
            if w not in seen_words:
                scored.append((w, sim))
                seen_words.add(w)
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


if __name__ == "__main__":
    tests = [
        "dʒɛmz",         # exact IPA for "gems" -- sanity check, should be 1.0
        "ndʒɛmz",        # actual Allosaurus substring from the voice memo ("...find the n gems")
        "juːbɚmiːts",    # actual Allosaurus output for "Kubernetes" ("uber meets" test case)
    ]
    for t in tests:
        print(f"{t!r} ->")
        for word, sim in phonemes_to_words(t):
            print(f"    {word!r}  sim={sim:.2f}")
