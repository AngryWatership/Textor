"""
Homophone/misrecognition correction using the phonetic bridge instead of an LLM.

For each 1- or 2-word span of the input, find the phonetically closest word
in a target vocabulary and substitute it if it's close enough and different
from the span itself.

Caveat (see outputs/phonetic_vs_llm_analysis.md): the target vocabulary here
is a constrained list, not the full 126k-word CMUdict, because nearest-
neighbor search over the whole dictionary needs a word-frequency prior to
avoid "correcting" rare real words into more common ones that merely sound
similar. A real implementation needs that frequency weighting; this is a
proof of concept of the phonetic-distance mechanism itself.
"""

from phonetic_bridge import arpabet_phones, phoneme_edit_distance

_DEFAULT_VOCAB = [
    "basket", "two", "eggs", "bear", "weather", "whether", "write", "right",
    "buy", "loaf", "bread", "there", "their", "they're", "flower", "flour",
    "piece", "peace", "break", "brake", "meat", "meet", "sea", "see",
    "night", "knight", "hear", "here", "wait", "weight", "sun", "son",
]


def _span_phones(words: list[str]) -> list[str] | None:
    out: list[str] = []
    for w in words:
        p = arpabet_phones(w)
        if p is None:
            return None
        out.extend(p)
    return out


def correct_phrase(
    text: str,
    vocab: list[str] = _DEFAULT_VOCAB,
    max_span=2,
    same_length_threshold=1.0,
    merge_threshold=0.65,
) -> str:
    """Substitute spans with phonetically-close vocab words.

    Two different rules, because they carry different false-positive risk:
    - same word count (e.g. "whether" -> "weather"): only a true homophone
      (similarity 1.0) qualifies -- anything looser swaps out correct short
      words too often (e.g. "I" ~ "buy" at 0.5 similarity, but they are not
      interchangeable).
    - word-count-reducing merge (e.g. "bash kit" -> "basket"): the actual
      error case this module targets, so a looser threshold is acceptable
      since collapsing two words into one rarely happens by chance.
    """
    words = text.rstrip(".").split()
    trailing = "." if text.rstrip().endswith(".") else ""

    vocab_phones = {v: _span_phones(v.split()) for v in vocab}
    vocab_phones = {k: v for k, v in vocab_phones.items() if v is not None}

    i = 0
    out_words: list[str] = []
    while i < len(words):
        best_match, best_sim, best_span = None, -1.0, 1
        for span in range(1, max_span + 1):
            if i + span > len(words):
                continue
            span_words = words[i : i + span]
            span_text = " ".join(span_words)
            sp = _span_phones(span_words)
            if sp is None:
                continue
            for cand, cp in vocab_phones.items():
                if cand.lower() == span_text.lower():
                    continue
                cand_word_count = len(cand.split())
                if cand_word_count == span:
                    required = same_length_threshold  # true-homophone swap
                elif cand_word_count < span:
                    required = merge_threshold  # word-count-reducing merge
                else:
                    continue  # never expand a span into more words
                dist = phoneme_edit_distance(sp, cp)
                sim = 1 - dist / max(len(sp), len(cp), 1)
                if sim >= required and sim > best_sim:
                    best_match, best_sim, best_span = cand, sim, span
        if best_match is not None:
            out_words.append(best_match)
            i += best_span
        else:
            out_words.append(words[i])
            i += 1
    return " ".join(out_words) + trailing


if __name__ == "__main__":
    tests = [
        "I have to get in the bash kit.",
        "I have two eggs in the basket.",
        "I seen a bare in the whether today.",
    ]
    for t in tests:
        print(f"{t!r} -> {correct_phrase(t)!r}")
