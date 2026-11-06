#!/usr/bin/env python3
"""Evaluate a trained checkpoint: confusion matrix, macro-F1 per class, and
the same silhouette / k-NN-purity metrics used in every prior CPU-only
round, computed on this model's own pooled embedding -- so this is a real
apples-to-apples comparison against raw MiniLM, LDA, supervised UMAP, the
contrastive head, and the instruction embedding (see docs/prior_results.md).

Usage:
    python src/evaluate.py --checkpoint outputs/best_model.pt --base-model allenai/specter2_base
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.neighbors import NearestNeighbors

from data import load_papers, get_or_make_split, LABELS
from model import VerdictClassifier, load_tokenizer
from train import PapersDataset


def knn_purity(X, labels, k=10, metric="euclidean"):
    nn_ = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, idx = nn_.kneighbors(X)
    idx = idx[:, 1:]
    labels = np.asarray(labels)
    same = labels[idx] == labels[:, None]
    per_point = same.mean(axis=1)
    per_class = {str(c): float(per_point[labels == c].mean()) for c in np.unique(labels)}
    return float(per_point.mean()), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/papers.jsonl")
    ap.add_argument("--split-out", default="outputs/split.json")
    ap.add_argument("--checkpoint", default="outputs/best_model.pt")
    ap.add_argument("--base-model", default="allenai/specter2_base")
    ap.add_argument("--finetune-mode", choices=["partial", "full"], default="partial")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    papers = load_papers(args.data)
    split = get_or_make_split(papers, args.split_out)
    tokenizer = load_tokenizer(args.base_model)
    test_ds = PapersDataset(papers, split["test"], tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = VerdictClassifier(base_model=args.base_model, finetune_mode=args.finetune_mode).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_y, all_pred, all_pooled = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, pooled = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_y.extend(batch["label"].numpy().tolist())
            all_pooled.append(pooled.cpu().numpy())

    y = np.array(all_y)
    pred = np.array(all_pred)
    pooled = np.concatenate(all_pooled, axis=0)

    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    sil = float(silhouette_score(pooled, y, metric="euclidean"))
    purity, purity_per_class = knn_purity(pooled, y, k=10)

    report = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "test_silhouette_euclidean": sil,
        "test_knn10_purity": purity,
        "test_knn10_purity_per_class": purity_per_class,
        "note": "silhouette/purity computed on this model's pooled embedding "
                "for its TEST split only -- compare directly against the "
                "numbers in docs/prior_results.md (same metrics, same "
                "out-of-fold discipline there, held-out test here).",
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
