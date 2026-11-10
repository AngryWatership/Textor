"""Append-only run log, kept at the repo root (runs_log.jsonl) plus an
always-regenerated human-readable RUNS.md table.

This is the thing to read first in a later session to see what's already
been tried, what it scored, and where its checkpoint lives (Drive path),
before deciding what to run next -- never trimmed, never hand-edited.
"""
import json
from pathlib import Path
from datetime import datetime, timezone


def new_run_id(base_model, finetune_mode):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_model = base_model.split("/")[-1]
    return f"{ts}_{short_model}_{finetune_mode}"


def _log_path(repo_root):
    return Path(repo_root) / "runs_log.jsonl"


def upsert_run(repo_root, run_id, fields):
    """Merge `fields` into the record for `run_id` (creating it if absent)
    and rewrite runs_log.jsonl + RUNS.md. Called once after training (with
    config + val metrics) and again after evaluation (with test metrics),
    so the same run_id accumulates both without duplicating a row."""
    log_path = _log_path(repo_root)
    records = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records[rec["run_id"]] = rec

    rec = records.get(run_id, {"run_id": run_id})
    rec.update(fields)
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    records[run_id] = rec

    with open(log_path, "w") as f:
        for run_id_, rec_ in sorted(records.items()):
            f.write(json.dumps(rec_) + "\n")

    _write_runs_md(repo_root, records)


def _write_runs_md(repo_root, records):
    lines = [
        "# Runs\n",
        "Auto-generated from `runs_log.jsonl` -- do not hand-edit, edit that "
        "file or re-run training/eval instead. Append-only: a run's row is "
        "never deleted, only extended (val metrics at train time, test "
        "metrics at eval time).\n",
        "| run_id | base_model | finetune_mode | val_macro_f1 | test_macro_f1 | "
        "test_silhouette | test_knn10_purity | checkpoint | updated_at |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run_id, rec in sorted(records.items(), reverse=True):
        lines.append(
            f"| {run_id} | {rec.get('base_model', '')} | {rec.get('finetune_mode', '')} | "
            f"{rec.get('best_val_macro_f1', '')} | {rec.get('test_macro_f1', '')} | "
            f"{rec.get('test_silhouette_euclidean', '')} | {rec.get('test_knn10_purity', '')} | "
            f"{rec.get('checkpoint_path', 'not yet copied to Drive (local: ' + str(rec.get('local_checkpoint_path', '?')) + ')')} | "
            f"{rec.get('updated_at', '')} |"
        )
    (Path(repo_root) / "RUNS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the run log from the command line")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    log_path = _log_path(args.repo_root)
    if not log_path.exists():
        print("no runs logged yet")
    else:
        for line in log_path.read_text().splitlines():
            print(json.loads(line))
