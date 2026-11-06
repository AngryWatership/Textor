"""Load papers.jsonl, make a stratified train/val/test split, tokenize.

The split is fixed (seed=42) and written to disk on first run so every
later run -- Colab session or not -- evaluates on the exact same held-out
test set. Re-running never reshuffles test once it has been written,
unless --force-resplit is passed explicitly.
"""
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

LABELS = ["not aligned", "adjacent", "aligned"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABELS)}


def load_papers(jsonl_path):
    papers = []
    with open(jsonl_path) as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def make_split(papers, seed=42, val_frac=0.15, test_frac=0.15):
    y = [LABEL_TO_ID[p["verdict"]] for p in papers]
    idx = np.arange(len(papers))

    train_idx, rest_idx, train_y, rest_y = train_test_split(
        idx, y, test_size=val_frac + test_frac, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_frac / (val_frac + test_frac),
        stratify=rest_y, random_state=seed,
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def get_or_make_split(papers, split_path, seed=42, force=False):
    split_path = Path(split_path)
    if split_path.exists() and not force:
        return json.loads(split_path.read_text())
    split = make_split(papers, seed=seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))
    return split


def text_of(paper):
    return f"{paper['title']}. {paper['abstract']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/papers.jsonl")
    ap.add_argument("--split-out", default="outputs/split.json")
    ap.add_argument("--force-resplit", action="store_true")
    args = ap.parse_args()

    papers = load_papers(args.data)
    split = get_or_make_split(papers, args.split_out, force=args.force_resplit)
    from collections import Counter
    for name, idxs in split.items():
        labels = [papers[i]["verdict"] for i in idxs]
        print(name, len(idxs), Counter(labels))
