"""
Stand-in for the small "correction LLM" stage of the TTS pipeline.

In production this stage is a small on-device LLM (e.g. Gemma-2 2B,
Phi-3-mini, or SmolLM2, run via llama.cpp/MediaPipe LLM Inference on
Android) whose job is narrow: given raw editor text, emit a version
that reads naturally out loud -- expand abbreviations and numbers,
fix obvious typos, and insert punctuation-based prosody cues -- while
never changing the meaning of the text.

This module implements that same narrow contract with deterministic
rules, since no LLM weights are reachable from this sandbox (see
session.md). It exists so the rest of the pipeline (tts_pipeline.py)
can be built and tested against the real interface shape today, and
swapped for a real model call later without touching callers.
"""

import re

_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Dr.": "Doctor",
    "St.": "Street",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
}

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _expand_abbreviations(text: str) -> str:
    for abbr, full in _ABBREVIATIONS.items():
        text = re.sub(re.escape(abbr), full, text)
    return text


def _expand_small_numbers(text: str) -> str:
    def repl(match: re.Match) -> str:
        digits = match.group(0)
        if len(digits) <= 2 and digits in _NUMBER_WORDS:
            return _NUMBER_WORDS[digits]
        return digits

    return re.sub(r"\b\d{1,2}\b", repl, text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def correct_for_speech(text: str) -> str:
    """Normalize raw editor text into a speech-friendly form.

    Contract (must hold for any real LLM implementation too):
    - output must not add or remove factual content
    - output must be safe to pass straight to a TTS engine
    """
    text = _expand_abbreviations(text)
    text = _expand_small_numbers(text)
    text = _collapse_whitespace(text)
    return text


if __name__ == "__main__":
    sample = "Dr.  Smith  met  5 colleagues  on  St. Mary's Ave,  e.g. Bob and 2 others."
    print(correct_for_speech(sample))
