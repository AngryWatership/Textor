"""
Multi-hypothesis beam search over a phoneme chain, matching against
CMUdict's phoneme->word chains. At each position, every dictionary word
that plausibly matches the phonemes starting there becomes a branch; all
surviving branches (beam) are carried forward in parallel rather than
committing early to one guess. Once the whole chain is consumed, the
surviving paths are ranked and pruned down to the best-scoring one(s) --
"keep all possible chains, proceed, then reduce by compatibility" per the
user's description.

Why this instead of greedy left-to-right matching: greedy matching commits
to the best-looking word at each position immediately, so one locally
plausible but globally wrong match can never be undone. Keeping a beam of
candidates means a path that looked good early but leads nowhere good
later can lose out to a path that looked slightly worse early on.
"""

from dataclasses import dataclass, field

import cmudict
from wordfreq import zipf_frequency

from phonetic_bridge import phoneme_edit_distance

_CMUDICT = cmudict.dict()

_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "g", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

BEAM_WIDTH = 10
MIN_WORD_MATCH_SIM = 0.65
MAX_WORD_LEN = 12  # phonemes; caps how far ahead one branch step looks
MIN_WORD_LEN = 3   # phonemes; below this, nearly anything "matches" something
                   # by chance (confirmed: without this, the search happily
                   # carves "reflection" into "ree"+"flick"+"sh"+"a"...),
                   # so a tiny match isn't trustworthy evidence on its own
SEGMENT_COUNT_PENALTY = 0.15  # per word, applied to the score -- without
                               # this the search always prefers many short
                               # exact matches over fewer long close-enough
                               # matches, which is backwards: a correct
                               # sentence has fewer, longer words, not more
WORD_FREQ_WEIGHT = 0.12  # per word, scaled by wordfreq's zipf score -- the
                          # missing language-model signal: pure phonetic
                          # matching treats "reflection" and "reef lick
                          # shun" as equally valid (confirmed: that's
                          # literally what it returned), since both are
                          # real dictionary words that sound plausible.
                          # Biasing toward common words over rare ones
                          # ("mirror" zipf=4.47 vs "aurum" zipf=2.24) is a
                          # cheap stand-in for a real language model.


def _strip_stress(phones: list[str]) -> list[str]:
    return [p.rstrip("012") for p in phones]


_INDEX: dict[tuple[int, str], list[tuple[str, str]]] | None = None


def _build_index() -> dict[tuple[int, str], list[tuple[str, str]]]:
    """(phoneme-length, first-phoneme) -> [(ipa_string, word), ...], built
    once. Bucketing by first phoneme too (not just length) means a query
    only has to compare against words that could plausibly match instead
    of scanning every word of a given length -- this is what makes
    beam_search_match finish in seconds instead of a minute per clump."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    index: dict[tuple[int, str], list[tuple[str, str]]] = {}
    for word, entries in _CMUDICT.items():
        arpabet = _strip_stress(entries[0])
        ipa = "".join(_ARPABET_TO_IPA.get(p, "?") for p in arpabet)
        if not ipa:
            continue
        index.setdefault((len(ipa), ipa[0]), []).append((ipa, word))
    _INDEX = index
    return index


COVERAGE_WEIGHT = 0.3  # rewards consuming more of the query, but as one
                        # factor among several, not the dominant sort key
                        # it used to be (that bug let a 5-word chain that
                        # merely finished the string always beat a
                        # 2-word chain that was more phonetically correct
                        # but left a short unmatched remainder)


@dataclass
class Hypothesis:
    position: int
    total_length: int = 1
    words: list[str] = field(default_factory=list)
    score: float = 0.0

    def ranking_score(self) -> float:
        """Total phonetic match quality, penalized per word used (anti
        over-segmentation) and boosted by word commonness (a cheap
        language-model stand-in -- see WORD_FREQ_WEIGHT).

        Both the frequency bonus and the base phonetic score are averaged
        per word, not summed -- summing let 5 short common words always
        out-score 1 long correct word, since each extra word added its
        own positive bonus (confirmed: "reflection" scored as a correct
        candidate at sim=0.78, but still lost to a 5-word chain of
        phonetically-plausible fragments because the sum kept growing with
        more words). Averaging removes that structural bias; the penalty
        then only has to counteract a worse *average* match, not a bigger
        raw total.
        """
        n = max(len(self.words), 1)
        avg_sim = self.score / n
        avg_freq = sum(max(zipf_frequency(w, "en"), 0) for w in self.words) / n
        coverage = self.position / max(self.total_length, 1)
        return (
            avg_sim
            - SEGMENT_COUNT_PENALTY * len(self.words)
            + WORD_FREQ_WEIGHT * avg_freq
            + COVERAGE_WEIGHT * coverage
        )


def _candidates_at(query: str, position: int) -> list[tuple[str, float, int]]:
    """[(word, similarity, phonemes_consumed), ...] for plausible matches
    starting at `position` in `query`. Only scans the (length,
    first-phoneme) buckets that could plausibly match -- see _build_index."""
    index = _build_index()
    remaining = len(query) - position
    first = query[position]
    out = []
    for length in range(MIN_WORD_LEN, min(MAX_WORD_LEN, remaining + 2) + 1):
        window = query[position:position + length]
        # an edit distance of up to 1 could still start with a different
        # first phoneme (an insertion/substitution at position 0), so
        # check a couple of plausible first-phoneme buckets, not just the
        # query's own first phoneme
        for bucket_key in {(length, first)} | {(length, c) for c in set(window[:2])}:
            for ipa, word in index.get(bucket_key, []):
                dist = phoneme_edit_distance(list(window), list(ipa))
                sim = 1 - dist / max(len(window), len(ipa), 1)
                if sim >= MIN_WORD_MATCH_SIM:
                    out.append((word, sim, length))
    return out


def beam_search_match(phoneme_string: str, beam_width: int = BEAM_WIDTH) -> list[Hypothesis]:
    """Returns surviving hypotheses, best first, after consuming the whole
    (or as much as possible of the) phoneme chain."""
    query = phoneme_string.replace(" ", "")
    beam = [Hypothesis(position=0, total_length=len(query))]
    finished: list[Hypothesis] = []

    steps = 0
    while beam and steps < 2 * len(query):
        steps += 1
        next_beam: list[Hypothesis] = []
        for hyp in beam:
            if hyp.position >= len(query):
                finished.append(hyp)
                continue
            candidates = _candidates_at(query, hyp.position)
            if not candidates:
                # dead end -- this path can't consume the rest; keep it as
                # a (worse) finished hypothesis rather than dropping it
                # silently, so a partial match still shows up at the end.
                finished.append(hyp)
                continue
            for word, sim, consumed in candidates:
                next_beam.append(Hypothesis(
                    position=hyp.position + consumed,
                    total_length=hyp.total_length,
                    words=hyp.words + [word],
                    score=hyp.score + sim,
                ))
        next_beam.sort(key=lambda h: -h.ranking_score())
        beam = next_beam[:beam_width]

    finished.extend(beam)
    finished.sort(key=lambda h: -h.ranking_score())
    return finished[:beam_width]


if __name__ == "__main__":
    tests = [
        "ndʒɛmz",       # "n gems" from the real voice memo
        "ledʒɛmiz",      # "[each] gem is" from the real voice memo
        "ɹiflɪkʃənɔfəmɛɹ",  # "reflection of a mirror" fragment
    ]
    for t in tests:
        print(f"{t!r}:")
        for h in beam_search_match(t)[:3]:
            print(f"    {h.words}  ranking_score={h.ranking_score():.2f}  consumed={h.position}/{len(t)}")
