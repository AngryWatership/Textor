#!/usr/bin/env python3
"""Evaluate a trained checkpoint: confusion matrix, macro-F1 per class, and
the same silhouette / k-NN-purity metrics used in every prior CPU-only
round, computed on this model's own pooled embedding -- so this is a real
apples-to-apples comparison against raw MiniLM, LDA, supervised UMAP, the
contrastive head, and the instruction embedding (see docs/prior_results.md).

Usage:
    python src/evaluate.py --run-id <run_id>
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
from model import VerdictTransformer
from tokenizer import load_vocab, PAD
from train import PapersDataset
from runlog import upsert_run


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
    ap.add_argument("--checkpoint", default=None,
                     help="defaults to <out-dir>/<run-id>/best_model.pt")
    ap.add_argument("--vocab", default=None,
                     help="defaults to <out-dir>/<run-id>/vocab.json "
                          "(written by train.py for this same run)")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--repo-root", default="..")
    ap.add_argument("--run-id", required=True,
                     help="must match the run_id used in train.py, so this "
                          "evaluation's results merge into the same log row")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) / args.run_id
    checkpoint_path = args.checkpoint or str(out_dir / "best_model.pt")
    vocab_path = args.vocab or str(out_dir / "vocab.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    papers = load_papers(args.data)
    split = get_or_make_split(papers, args.split_out)
    vocab = load_vocab(vocab_path)
    test_ds = PapersDataset(papers, split["test"], vocab, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = VerdictTransformer(
        vocab_size=len(vocab), embed_dim=args.embed_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ffn_dim=args.ffn_dim, max_length=args.max_length,
        pad_id=vocab[PAD],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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

    upsert_run(args.repo_root, args.run_id, {
        # test_macro_f1 is NOT set here -- train.py already logs it, and
        # upsert_run does a plain dict merge, so writing None here would
        # clobber that real value instead of leaving it alone.
        "test_silhouette_euclidean": sil,
        "test_knn10_purity": purity,
        "test_knn10_purity_per_class": purity_per_class,
        "confusion_matrix": cm.tolist(),
        "eval_report_path": str(out_dir / "eval_report.json"),
    })
    print(f"run_id={args.run_id} updated in {args.repo_root}/runs_log.jsonl and RUNS.md")


if __name__ == "__main__":
    main()
