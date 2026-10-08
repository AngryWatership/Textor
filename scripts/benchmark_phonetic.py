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
    ("merge", "I have to get in the bash kit.", "I have two get in the basket."),
    ("merge", "Put the groceries in the bash kit.", "Put the groceries in the basket."),
    ("true_homophone", "I seen a bare in the whether today.", "I seen bear in weather today."),
    ("true_homophone", "Can you here me from over hear?", "Can you hear me from over here."),
    ("true_homophone", "Their going to the party, not they're.", "There going to the party, not they're."),
    ("true_homophone", "I need to right a letter with my write hand.", "I need to write a letter with my right hand."),
    ("true_homophone", "The flower is on the kitchen counter.", "The flour is on the kitchen counter."),
    ("true_homophone", "Please way the package before you weight it.", "Please weigh the package before you wait it."),
    ("plain_passthrough", "The sky is blue today.", "The sky is blue today."),
    ("plain_passthrough", "I had breakfast this morning.", "I had breakfast this morning."),
]


def main() -> None:
    results = []
    for category, raw, expected in CASES:
        actual = correct_phrase(raw)
        ok = actual.strip().lower() == expected.strip().lower()
        results.append((category, raw, expected, actual, ok))

    total = len(results)
    passed = sum(r[-1] for r in results)
    print(f"Phonetic correction (scripts/phonetic_correct.py): {passed}/{total} = {passed/total:.1%}")
    print()
    for category, raw, expected, actual, ok in results:
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] ({category}) {raw!r}")
        if not ok:
            print(f"       expected: {expected!r}")
            print(f"       actual:   {actual!r}")


if __name__ == "__main__":
    main()
