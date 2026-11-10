#!/usr/bin/env python3
"""Train VerdictClassifier on papers.jsonl. GPU strongly recommended.

Usage:
    python src/train.py --base-model allenai/specter2_base --finetune-mode partial
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
from model import VerdictClassifier, load_tokenizer
from runlog import new_run_id, upsert_run


class PapersDataset(Dataset):
    def __init__(self, papers, indices, tokenizer, max_length=384):
        self.papers = [papers[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        p = self.papers[idx]
        enc = self.tokenizer(
            text_of(p), truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
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
    ap.add_argument("--base-model", default="allenai/specter2_base")
    ap.add_argument("--finetune-mode", choices=["partial", "full"], default="partial")
    ap.add_argument("--n-unfrozen-layers", type=int, default=2)
    ap.add_argument("--loss", choices=["weighted-ce", "focal"], default="weighted-ce")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--repo-root", default="..",
                     help="where runs_log.jsonl / RUNS.md live (repo root, not src/)")
    ap.add_argument("--run-id", default=None,
                     help="defaults to <timestamp>_<base_model>_<finetune_mode>; "
                          "pass explicitly to resume/overwrite a specific run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. This script is designed to run on a "
              "Colab GPU runtime; CPU will be very slow for full/partial "
              "fine-tuning of a transformer encoder.")

    run_id = args.run_id or new_run_id(args.base_model, args.finetune_mode)
    print(f"run_id: {run_id}")
    # every run gets its own subdirectory, nothing overwrites a prior run's
    # checkpoint/config/reports by accident
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id, "base_model": args.base_model,
        "finetune_mode": args.finetune_mode, "n_unfrozen_layers": args.n_unfrozen_layers,
        "loss": args.loss, "max_length": args.max_length, "batch_size": args.batch_size,
        "lr": args.lr, "max_epochs": args.max_epochs, "patience": args.patience,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    papers = load_papers(args.data)
    split = get_or_make_split(papers, args.split_out, seed=args.seed)
    print({k: len(v) for k, v in split.items()})

    tokenizer = load_tokenizer(args.base_model)
    train_ds = PapersDataset(papers, split["train"], tokenizer, args.max_length)
    val_ds = PapersDataset(papers, split["val"], tokenizer, args.max_length)
    test_ds = PapersDataset(papers, split["test"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = VerdictClassifier(
        base_model=args.base_model, finetune_mode=args.finetune_mode,
        n_unfrozen_layers=args.n_unfrozen_layers,
    ).to(device)

    weights = class_weights_from(papers, split["train"]).to(device)
    if args.loss == "weighted-ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(gamma=2.0, weight=weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

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
        "base_model": args.base_model, "finetune_mode": args.finetune_mode,
        "best_val_macro_f1": best_val_f1, "test_macro_f1": test_f1,
        "seed": args.seed, "run_id": run_id,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2))
    print(f"wrote {out_dir / 'best_model.pt'} and {out_dir / 'train_summary.json'}")

    # record in the repo-level append-only log immediately -- if this
    # process is interrupted before evaluate.py runs, the run is still
    # visible (with val metrics) rather than silently missing
    upsert_run(args.repo_root, run_id, {
        "base_model": args.base_model, "finetune_mode": args.finetune_mode,
        "best_val_macro_f1": best_val_f1, "test_macro_f1": test_f1,
        "config_path": str(out_dir / "config.json"),
        "local_checkpoint_path": str(out_dir / "best_model.pt"),
    })
    print(f"run_id={run_id} logged to {args.repo_root}/runs_log.jsonl and RUNS.md")


if __name__ == "__main__":
    main()
