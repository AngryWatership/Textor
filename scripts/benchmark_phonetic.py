"""
Benchmark for homophone/misrecognition correction -- the task this session
pivoted to (see outputs/phonetic_vs_llm_analysis.md), not the abbreviation/
number normalization benchmark_correct.py covers.

Each case is a plausible ASR/dictation-style error: a word or words
replaced by something that sounds similar but breaks the sentence's
meaning. Scored by exact match against the intended sentence.
"""

from phonetic_correct import correct_phrase

CASES = [
    # (category, misheard input, intended correction)
    ("merge", "I have to get in the bash kit.", "I have to get in the basket."),
    ("merge", "Put the groceries in the bash kit.", "Put the groceries in the basket."),
    # true_homophone cases: correct_phrase no longer auto-rewrites these (see
    # outputs/phonetic_vs_llm_analysis.md -- sound alone can't tell "to" from
    # "two", and auto-correcting broke already-correct real dictated audio;
    # confirmed on a real voice memo, not just synthetic cases). The pass
    # condition is now "flagged the right word with the right suggestion",
    # not "silently rewrote it correctly".
    ("true_homophone", "I seen a bare in the whether today.", "bare->bear, whether->weather"),
    ("true_homophone", "Can you here me from over hear?", "here->hear, hear->here"),
    ("true_homophone", "Their going to the party, not they're.", "Their->There"),
    ("true_homophone", "I need to right a letter with my write hand.", "right->write, write->right"),
    ("true_homophone", "The flower is on the kitchen counter.", "flower->flour"),
    ("true_homophone", "Please way the package before you weight it.", "way->weigh, weight->wait"),
    ("plain_passthrough", "The sky is blue today.", "The sky is blue today."),
    ("plain_passthrough", "I had breakfast this morning.", "I had breakfast this morning."),
]


def main() -> None:
    results = []
    for category, raw, expected in CASES:
        actual, flags = correct_phrase(raw)
        if category == "true_homophone":
            # pass if every "word->suggestion" pair in `expected` was flagged
            expected_pairs = {tuple(p.split("->")) for p in expected.split(", ")}
            flagged_pairs = {(f["span_text"], f["suggestion"]) for f in flags}
            ok = expected_pairs <= flagged_pairs
        else:
            ok = actual.strip().lower() == expected.strip().lower()
        results.append((category, raw, expected, actual, flags, ok))

    total = len(results)
    passed = sum(r[-1] for r in results)
    print(f"Phonetic correction (scripts/phonetic_correct.py): {passed}/{total} = {passed/total:.1%}")
    print()
    for category, raw, expected, actual, flags, ok in results:
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] ({category}) {raw!r}")
        if not ok:
            print(f"       expected: {expected!r}")
            print(f"       actual:   {actual!r}  flags={flags}")


if __name__ == "__main__":
    main()
