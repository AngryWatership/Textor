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

from phonetic_bridge import (
    arpabet_phones,
    ipa_phones,
    phoneme_edit_distance,
    phonetic_class_similarity,
    phonetic_similarity,
)

# phonetic_class_similarity alone is too permissive once scanning every
# window against every vocab word -- confirmed empirically: it flagged
# "the"->Docker (0.83) and "my"->mirror (0.80) as plausible misses, pure
# false positives. Requiring exact phonetic_similarity to also clear this
# (much lower) bar fixes it: every real miss tested so far scores >=0.33
# here, every false positive found during testing scored <=0.25 -- see
# outputs/phonetic_vs_llm_analysis.md for the full calibration table.
JARGON_EXACT_GATE = 0.3

_DEFAULT_VOCAB = [
    "basket", "two", "eggs", "bear", "weather", "whether", "write", "right",
    "buy", "loaf", "bread", "there", "their", "they're", "flower", "flour",
    "piece", "peace", "break", "brake", "meat", "meet", "sea", "see",
    "night", "knight", "hear", "here", "wait", "weight", "sun", "son",
]

# Technical jargon CMUdict doesn't cover (confirmed in
# outputs/phonetic_vs_llm_analysis.md), extend with the user's own terms --
# this list is exactly what makes the correction stage useful for a given
# person's vocabulary/project rather than generic English. "gems" is a real
# confirmed miss from a real dictated voice memo (Whisper heard "enzymes").
_DEFAULT_JARGON_VOCAB = [
    "OAuth", "idempotent", "JSON", "Kubernetes", "GPT", "LLM", "API",
    "async", "kernel", "GitHub", "Python", "Docker",
    "gems", "cycle", "mirror", "stage",
]

# Calibrated against known cases (see phonetic_bridge.phonetic_class_similarity
# docstring): true matches score above this, unrelated words and the
# hardest confirmed misses (OAuth -> "UV No.1") stay at or below it.
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
) -> tuple[str, list[dict]]:
    """Fix word-boundary merge errors; flag (don't auto-fix) true homophones.

    Two cases, handled differently because sound alone resolves one but not
    the other:
    - word-count-reducing merge (e.g. "bash kit" -> "basket"): auto-corrected.
      Low false-positive risk -- collapsing two words into one rarely
      happens by chance -- and there's nothing sound-based left to be
      ambiguous about once two words plausibly merge into one real word.
    - same word count, true homophone (e.g. "to"/"two", "there"/"their"):
      flagged, never silently substituted. Confirmed necessary on real
      dictated audio (see outputs/whisper_limits_analysis.md addendum):
      whether "to" should become "two" depends on grammar/context the
      sound carries no information about -- auto-correcting it is a
      coin flip that actively breaks already-correct text as often as
      it fixes wrong text.

    Returns (text_with_merges_applied, homophone_flags) -- flags have the
    same shape as flag_jargon()'s, so callers can render both the same way.
    """
    words = text.rstrip(".").split()
    trailing = "." if text.rstrip().endswith(".") else ""

    vocab_phones = {v: _span_phones(v.split()) for v in vocab}
    vocab_phones = {k: v for k, v in vocab_phones.items() if v is not None}

    i = 0
    out_words: list[str] = []
    flags: list[dict] = []
    while i < len(words):
        best_merge, best_merge_sim, best_merge_span = None, -1.0, 1
        best_homophone, best_homophone_sim = None, -1.0
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
                dist = phoneme_edit_distance(sp, cp)
                sim = 1 - dist / max(len(sp), len(cp), 1)
                if cand_word_count == span:
                    if sim >= same_length_threshold and sim > best_homophone_sim:
                        best_homophone, best_homophone_sim = cand, sim
                elif cand_word_count < span:
                    if sim >= merge_threshold and sim > best_merge_sim:
                        best_merge, best_merge_sim, best_merge_span = cand, sim, span
                # cand_word_count > span: never expand a span into more words
        if best_merge is not None:
            out_words.append(best_merge)
            i += best_merge_span
        else:
            if best_homophone is not None:
                flags.append({
                    "span_text": words[i],
                    "suggestion": best_homophone,
                    "confidence": round(best_homophone_sim, 2),
                    "word_index": i,
                })
            out_words.append(words[i])
            i += 1
    return " ".join(out_words) + trailing, flags


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
            for term in vocab:
                if term.lower() == span_text.lower():
                    continue  # already correct, nothing to flag
                # dolgo-prime (broad articulatory class) catches real misses
                # that exact phoneme matching doesn't -- "enzymes" heard for
                # "gems" (a real dictated miss) scores 0.33 on exact IPA
                # edit distance but 0.62 here -- but gated by the exact
                # metric too, since dolgo-prime alone is too permissive on
                # short/common words (see JARGON_EXACT_GATE above).
                if phonetic_similarity(term, span_text) < JARGON_EXACT_GATE:
                    continue
                sim = phonetic_class_similarity(term, span_text)
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
        corrected, flags = correct_phrase(t)
        print(f"{t!r} -> {corrected!r}  homophone_flags={flags}")

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
