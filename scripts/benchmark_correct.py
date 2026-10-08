"""
Hand-built benchmark for the correction stage (`correct_for_speech`).

There is no real text-normalization dataset reachable from this sandbox
(Google's TN Challenge data lives on Kaggle/GCS, rwsproat/text-normalization-data
on GitHub ships data via release assets, not raw files -- both are behind
the network policy blocking kaggle.com/github.com release downloads here).
So this is a small, hand-curated benchmark covering the categories every
TTS text-normalizer has to handle: abbreviations (in-vocabulary and
out-of-vocabulary), small and large numbers, decimals, dates, currency,
ordinals, acronyms, and ambiguous abbreviations. It exists to give an
honest, reproducible accuracy number for the *current* rule-based stand-in,
and a regression baseline for whatever correction LLM replaces it later.

Scoring: exact string match against a single "acceptable" normalization.
Real text normalization often has multiple valid outputs (e.g. "5" could
stay a digit if a TTS engine handles digits natively) -- exact match is a
conservative, strict metric, chosen so failures are visible rather than
hidden by a lenient scorer.
"""

from correct import correct_for_speech

CASES = [
    # (category, input, expected_output)
    ("abbrev_in_vocab", "Dr. Smith arrived.", "Doctor Smith arrived."),
    ("abbrev_in_vocab", "Mr. and Mrs. Jones", "Mister and Missus Jones"),
    ("abbrev_out_of_vocab", "Prof. Lee gave a talk.", "Professor Lee gave a talk."),
    ("abbrev_out_of_vocab", "See Fig. 3 for details.", "See Figure 3 for details."),
    ("abbrev_ambiguous", "123 St. Mary's St.", "123 Saint Mary's Street."),
    ("number_small", "I have 3 apples.", "I have three apples."),
    ("number_small", "9 lives.", "nine lives."),
    ("number_large", "The year 1999 was wild.", "The year nineteen ninety-nine was wild."),
    ("number_large", "There are 2024 reasons.", "There are two thousand twenty-four reasons."),
    ("number_large", "Population: 45000.", "Population: forty-five thousand."),
    ("decimal", "Pi is about 3.14.", "Pi is about three point one four."),
    ("decimal", "It costs 9.99.", "It costs nine point nine nine."),
    ("currency", "It costs $9.99.", "It costs nine dollars and ninety-nine cents."),
    ("currency", "He earns $45000 a year.", "He earns forty-five thousand dollars a year."),
    ("date", "The meeting is on 5/3.", "The meeting is on May third."),
    ("date", "Born in 1990.", "Born in nineteen ninety."),
    ("ordinal", "He finished 2nd.", "He finished second."),
    ("ordinal", "the 21st century", "the twenty-first century"),
    ("acronym", "I work at NASA.", "I work at NASA."),
    ("acronym", "Send it via FTP.", "Send it via F T P."),
    ("time", "It's 9:30pm.", "It's nine thirty p m."),
    ("percent", "A 50% discount.", "A fifty percent discount."),
    ("plain_passthrough", "The sky is blue.", "The sky is blue."),
]


def main() -> None:
    results = []
    for category, raw, expected in CASES:
        actual = correct_for_speech(raw)
        ok = actual.strip() == expected.strip()
        results.append((category, raw, expected, actual, ok))

    total = len(results)
    passed = sum(1 for r in results if r[-1])

    print(f"Overall exact-match accuracy: {passed}/{total} = {passed/total:.1%}")
    print()
    by_cat: dict[str, list[bool]] = {}
    for category, *_rest, ok in results:
        by_cat.setdefault(category, []).append(ok)
    for category, oks in sorted(by_cat.items()):
        print(f"  {category:22s} {sum(oks)}/{len(oks)}")

    print()
    print("Failures:")
    for category, raw, expected, actual, ok in results:
        if not ok:
            print(f"  [{category}] {raw!r}")
            print(f"    expected: {expected!r}")
            print(f"    actual:   {actual!r}")


if __name__ == "__main__":
    main()
