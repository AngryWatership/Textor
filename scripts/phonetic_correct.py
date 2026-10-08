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

from phonetic_bridge import arpabet_phones, ipa_phones, phoneme_edit_distance

_DEFAULT_VOCAB = [
    "basket", "two", "eggs", "bear", "weather", "whether", "write", "right",
    "buy", "loaf", "bread", "there", "their", "they're", "flower", "flour",
    "piece", "peace", "break", "brake", "meat", "meet", "sea", "see",
    "night", "knight", "hear", "here", "wait", "weight", "sun", "son",
]

# Technical jargon CMUdict doesn't cover (confirmed in
# outputs/phonetic_vs_llm_analysis.md), extend with the user's own terms as
# needed -- this list is exactly what makes the correction stage useful for
# a given person's vocabulary rather than generic English.
_DEFAULT_JARGON_VOCAB = [
    "OAuth", "idempotent", "JSON", "Kubernetes", "GPT", "LLM", "API",
    "async", "kernel", "GitHub", "Python", "Docker",
]

# Below this, Whisper's error diverged too far phonetically to recover by
# matching alone (e.g. "OAuth" -> "UV No.1" scores ~0.2-0.4 -- see
# outputs/whisper_limits_analysis.md); flagging for user confirmation rather
# than auto-correcting is the right move below this threshold, same
# reasoning as the true-homophone case in correct_phrase.
JARGON_FLAG_THRESHOLD = 0.6


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


def flag_jargon(
    text: str,
    vocab: list[str] = _DEFAULT_JARGON_VOCAB,
    max_span: int = 2,
    threshold: float = JARGON_FLAG_THRESHOLD,
) -> list[dict]:
    """Find spans that are plausibly a misheard technical term.

    Unlike correct_phrase, this never auto-replaces -- it returns candidate
    flags (span, suggestion, confidence) for the caller (editor UI) to show
    to the user, same "flag, don't silently rewrite" reasoning as true
    homophones: a wrong auto-correction on a real word is worse than asking.

    Uses espeak-ng IPA rather than CMUdict, since jargon like "OAuth" and
    "idempotent" isn't in CMUdict's 126k English words (confirmed in
    outputs/phonetic_vs_llm_analysis.md) but espeak-ng can phonemicize any
    text via its letter-to-sound rules, even invented/technical words.
    """
    words = text.rstrip(".").split()
    vocab_ipa = {v: ipa_phones(v).replace(" ", "") for v in vocab}

    vocab_lower = {v.lower() for v in vocab}
    flags = []
    i = 0
    while i < len(words):
        if words[i].lower() in vocab_lower:
            i += 1  # already an exact match -- don't let a longer fuzzy
            continue  # span starting here outscore this by coincidence
        best_term, best_sim, best_span = None, -1.0, 1
        for span in range(1, max_span + 1):
            if i + span > len(words):
                continue
            # an exact-match word inside the window is a hard boundary --
            # never let a fuzzy span swallow a word that's already correct
            if any(words[j].lower() in vocab_lower for j in range(i + 1, i + span)):
                break
            span_text = " ".join(words[i : i + span])
            span_ipa = ipa_phones(span_text).replace(" ", "")
            for term, term_ipa in vocab_ipa.items():
                if term.lower() == span_text.lower():
                    continue  # already correct, nothing to flag
                dist = phoneme_edit_distance(list(term_ipa), list(span_ipa))
                sim = 1 - dist / max(len(term_ipa), len(span_ipa), 1)
                if sim > best_sim:
                    best_term, best_sim, best_span = term, sim, span
        if best_term is not None and best_sim >= threshold:
            flags.append({
                "span_text": " ".join(words[i : i + best_span]),
                "suggestion": best_term,
                "confidence": round(best_sim, 2),
                "word_index": i,
            })
            i += best_span
        else:
            i += 1
    return flags


if __name__ == "__main__":
    tests = [
        "I have to get in the bash kit.",
        "I have two eggs in the basket.",
        "I seen a bare in the whether today.",
    ]
    for t in tests:
        print(f"{t!r} -> {correct_phrase(t)!r}")

    print()
    print("Jargon flagging (confirm-don't-auto-correct):")
    jargon_tests = [
        "the cluster run mark uber meets",
        "we store the config as J-1",
        "the PPT and LLM API my rate limited",
        "I am UV No.1 for authentication",
    ]
    for t in jargon_tests:
        print(f"  {t!r} -> {flag_jargon(t)}")
