"""Minimal from-scratch tokenizer and vocabulary -- no downloads, no
Hugging Face, no pretrained subword model. A plain regex word/number/
punctuation tokenizer, and a vocabulary built directly from this
project's own corpus (data/papers.jsonl titles+abstracts).
"""
import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, CLS = "[PAD]", "[UNK]", "[CLS]"
SPECIAL_TOKENS = [PAD, UNK, CLS]

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)?|[^\sa-z0-9]")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts, max_size=30000, min_freq=2):
    """Word-level vocabulary from this project's own text, nothing external."""
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counts.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def save_vocab(vocab, path):
    Path(path).write_text(json.dumps(vocab))


def load_vocab(path):
    return json.loads(Path(path).read_text())


def encode(text, vocab, max_length=384):
    unk_id = vocab[UNK]
    ids = [vocab[CLS]] + [vocab.get(tok, unk_id) for tok in tokenize(text)]
    ids = ids[:max_length]
    attention = [1] * len(ids)
    pad_id = vocab[PAD]
    while len(ids) < max_length:
        ids.append(pad_id)
        attention.append(0)
    return ids, attention


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_papers, text_of

    ap = argparse.ArgumentParser(description="Build a vocab.json from papers.jsonl")
    ap.add_argument("--data", default="data/papers.jsonl")
    ap.add_argument("--out", default="outputs/vocab.json")
    ap.add_argument("--max-size", type=int, default=30000)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    papers = load_papers(args.data)
    vocab = build_vocab((text_of(p) for p in papers), args.max_size, args.min_freq)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, args.out)
    print(f"wrote {len(vocab)}-token vocab to {args.out}")
