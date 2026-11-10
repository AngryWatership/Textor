#!/usr/bin/env python3
"""Train VerdictTransformer (a from-scratch encoder, no pretrained weights,
no Hugging Face) on papers.jsonl. GPU strongly recommended.

Usage:
    python src/train.py --embed-dim 256 --num-layers 4
"""
import argparse
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import load_papers, get_or_make_split, text_of, LABEL_TO_ID
from model import VerdictTransformer
from tokenizer import build_vocab, save_vocab, encode as tok_encode, PAD
from runlog import new_run_id, upsert_run


class PapersDataset(Dataset):
    def __init__(self, papers, indices, vocab, max_length=384):
        self.papers = [papers[i] for i in indices]
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        p = self.papers[idx]
        ids, attn = tok_encode(text_of(p), self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "label": torch.tensor(LABEL_TO_ID[p["verdict"]], dtype=torch.long),
        }


def class_weights_from(papers, indices):
    labels = [LABEL_TO_ID[papers[i]["verdict"]] for i in indices]
    counts = Counter(labels)
    n = len(labels)
    weights = torch.tensor(
        [n / (len(counts) * counts.get(c, 1)) for c in range(3)], dtype=torch.float32
    )
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def macro_f1(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_loader(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_y.extend(batch["label"].numpy().tolist())
    return np.array(all_y), np.array(all_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/papers.jsonl")
    ap.add_argument("--split-out", default="outputs/split.json")
    ap.add_argument("--vocab-out", default=None,
                     help="defaults to <out-dir>/<run-id>/vocab.json; built "
                          "fresh from --data every run (no download, cheap)")
    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--vocab-min-freq", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--model-dropout", type=float, default=0.2)
    ap.add_argument("--loss", choices=["weighted-ce", "focal"], default="weighted-ce")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--repo-root", default="..",
                     help="where runs_log.jsonl / RUNS.md live (repo root, not src/)")
    ap.add_argument("--run-id", default=None,
                     help="defaults to <timestamp>_scratch-transformer_<embed-dim>d"
                          "<num-layers>l; pass explicitly to resume/overwrite a run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. This script is designed to run on a "
              "Colab GPU runtime; CPU will be slow for a transformer encoder "
              "trained from scratch.")

    model_tag = f"scratch-transformer_{args.embed_dim}d{args.num_layers}l"
    run_id = args.run_id or new_run_id(model_tag, "from-scratch")
    print(f"run_id: {run_id}")
    # every run gets its own subdirectory, nothing overwrites a prior run's
    # checkpoint/config/reports by accident
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_out = args.vocab_out or str(out_dir / "vocab.json")

    config = {
        "run_id": run_id, "base_model": model_tag, "finetune_mode": "from-scratch",
        "embed_dim": args.embed_dim, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "ffn_dim": args.ffn_dim,
        "vocab_size": args.vocab_size, "vocab_min_freq": args.vocab_min_freq,
        "loss": args.loss, "max_length": args.max_length, "batch_size": args.batch_size,
        "lr": args.lr, "max_epochs": args.max_epochs, "patience": args.patience,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    papers = load_papers(args.data)
    split = get_or_make_split(papers, args.split_out, seed=args.seed)
    print({k: len(v) for k, v in split.items()})

    # vocabulary built only from this project's own corpus -- no download,
    # no external subword model; trained-split-only would also be
    # defensible, but vocabulary membership carries no label information,
    # so building it from the full corpus (as any tokenizer step normally
    # would) is not a leak of `verdict`
    vocab = build_vocab((text_of(p) for p in papers), args.vocab_size, args.vocab_min_freq)
    save_vocab(vocab, vocab_out)
    print(f"built {len(vocab)}-token vocab -> {vocab_out}")

    train_ds = PapersDataset(papers, split["train"], vocab, args.max_length)
    val_ds = PapersDataset(papers, split["val"], vocab, args.max_length)
    test_ds = PapersDataset(papers, split["test"], vocab, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = VerdictTransformer(
        vocab_size=len(vocab), embed_dim=args.embed_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, ffn_dim=args.ffn_dim, dropout=args.model_dropout,
        max_length=args.max_length, pad_id=vocab[PAD],
    ).to(device)

    weights = class_weights_from(papers, split["train"]).to(device)
    if args.loss == "weighted-ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(gamma=2.0, weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.max_epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        y_val, pred_val = evaluate_loader(model, val_loader, device)
        val_f1 = macro_f1(y_val, pred_val)
        print(f"epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f} "
              f"val_macro_f1={val_f1:.4f} ({time.time()-t0:.0f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(best_state, out_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch+1} (best val_macro_f1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    y_test, pred_test = evaluate_loader(model, test_loader, device)
    test_f1 = macro_f1(y_test, pred_test)
    print(f"TEST macro_f1={test_f1:.4f}")

    train_summary = {
        "base_model": model_tag, "finetune_mode": "from-scratch",
        "best_val_macro_f1": best_val_f1, "test_macro_f1": test_f1,
        "seed": args.seed, "run_id": run_id,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2))
    print(f"wrote {out_dir / 'best_model.pt'} and {out_dir / 'train_summary.json'}")

    # record in the repo-level append-only log immediately -- if this
    # process is interrupted before evaluate.py runs, the run is still
    # visible (with val metrics) rather than silently missing
    upsert_run(args.repo_root, run_id, {
        "base_model": model_tag, "finetune_mode": "from-scratch",
        "best_val_macro_f1": best_val_f1, "test_macro_f1": test_f1,
        "config_path": str(out_dir / "config.json"),
        "local_checkpoint_path": str(out_dir / "best_model.pt"),
    })
    print(f"run_id={run_id} logged to {args.repo_root}/runs_log.jsonl and RUNS.md")


if __name__ == "__main__":
    main()
