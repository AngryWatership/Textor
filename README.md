# paper-verdict-finetune

Train a Transformer text encoder, entirely from scratch, to classify
research papers into `aligned` / `adjacent` / `not aligned` — the
pipeline's own verdict label, used directly as ground truth. No
pretrained weights anywhere: no Hugging Face, no download of any model
at runtime. `src/model.py`'s encoder is random-initialized `torch.nn`
layers, and `src/tokenizer.py`'s vocabulary is built directly from this
project's own 2,093 papers.

## Why this exists

Four rounds of CPU-only experiments (see `docs/prior_results.md`) tried to
recover `verdict` from a **frozen** generic sentence embedding
(`all-MiniLM-L6-v2`) via increasingly elaborate projections: LDA, PLS-DA,
supervised UMAP (swept over label weight), a small contrastive projection
head, and an instruction-conditioned re-embedding naming the exact
criterion (sensor auditing / tamper-evidence / anti-falsification
emphasis). All of them plateaued near a weak ceiling (out-of-fold
silhouette ~0, k-NN purity 0.52–0.61 against a 0.45 chance floor). The
best of the lot (a contrastive head, purity 0.613) was also the only one
that used direct supervision rather than hoping label-relevant structure
already existed in the frozen embedding geometry.

The lever none of those rounds could pull on CPU: a representation
actually built around `verdict`, instead of a frozen one re-projected
after the fact. This repo removes that constraint on a GPU (Colab) by
training a small Transformer encoder end to end, from random
initialization, directly against `verdict` -- every parameter is free to
move, not just a head on top of something else's fixed features.

No LLM, no prompting, and no pretrained model of any kind anywhere in
this pipeline -- not even as a starting point. It's a trained,
deterministic classifier -- a fixed function from paper text to label --
not a model you query at inference time with natural-language
instructions, and not a model whose weights were ever fetched from a
hub rather than trained here.

## Structure

```
data/            exported papers.jsonl (id, title, abstract, verdict)
src/
  data.py        load, stratified split, text_of()
  tokenizer.py    from-scratch word-level tokenizer + vocab built from data/papers.jsonl
  model.py       from-scratch nn.Transformer encoder + classification head (no pretrained weights)
  train.py       builds vocab, training loop, class-weighted loss, early stopping, checkpoints
  evaluate.py    confusion matrix, macro-F1, silhouette/purity on learned embeddings
  runlog.py      append-only run log (runs_log.jsonl + RUNS.md), shared by train.py/evaluate.py
notebooks/
  colab_train.ipynb   thin GPU runner: clone this repo, pip install, run train.py + evaluate.py,
                       zip checkpoints for direct browser download, push the lightweight run record back here
requirements.txt
runs_log.jsonl   append-only JSON-lines record of every run (one line per run_id, merged, never deleted)
RUNS.md          human-readable table auto-generated from runs_log.jsonl -- do not hand-edit
docs/
  prior_results.md    summary of the four CPU-only rounds, for context
```

## Quickstart (Colab)

1. Open `notebooks/colab_train.ipynb` in Colab, select a GPU runtime.
2. Run all cells: it clones this branch, installs `requirements.txt`,
   and calls `src/train.py` then `src/evaluate.py`.
3. Results (confusion matrix, macro-F1, silhouette/purity) are written to
   `outputs/<run_id>/` and also displayed inline in the notebook.
4. The notebook zips the full run (including `best_model.pt`) and prompts
   you with a direct browser download -- no third-party account, no Drive.
   It then commits and pushes the lightweight record (`RUNS.md`,
   `runs_log.jsonl`, that run's `config.json`/`train_summary.json`/
   `eval_report.json`) back to this branch -- not the weights, those only
   exist in the zip you downloaded. Checking `RUNS.md` in a later session,
   from a plain `git clone`, tells you every run tried and what it scored
   without needing Colab at all.

## Configuration

See `src/train.py --help`. Key flags (model size/depth, not which
pretrained checkpoint to load -- there isn't one):

- `--embed-dim` : embedding/model width (default 256)
- `--num-layers` : Transformer encoder layers (default 4)
- `--num-heads` : attention heads per layer (default 8)
- `--vocab-size` / `--vocab-min-freq` : vocabulary built fresh from `--data` every run
- `--loss` : `weighted-ce` (default) | `focal`
- `--seed` : fixed by default (42) for reproducibility

## Reviewing past runs / persistence model

Every run gets a `run_id` (`<UTC timestamp>_scratch-transformer_<dims>d<layers>l_from-scratch`)
and its own `outputs/<run_id>/` directory, so nothing overwrites a prior
run's checkpoint, vocab, or config by accident.

- **Full artifact (weights included)**: zipped by the notebook at the end
  of the run (`outputs/<run_id>.zip`) and handed to you as a direct
  browser download via `google.colab.files.download`. No third-party
  storage account, and deliberately kept off Google Drive so it carries
  no risk to anything else already stored there. Save the zip yourself;
  re-upload it to a later Colab session if you need that checkpoint again.
- **Lightweight record (no weights)**: `config.json`, `train_summary.json`,
  `eval_report.json` for that run_id, plus the repo-root `runs_log.jsonl`
  (append-only, one JSON object per run_id, fields merged in by both
  `train.py` and `evaluate.py` via `src/runlog.py:upsert_run`) and the
  auto-regenerated `RUNS.md` table. These get committed and pushed back to
  this branch, so `RUNS.md` is always a current index of every run --
  readable from a fresh `git clone` with no GPU, no Colab, and no account
  of any kind needed.

Before launching a new run, read `RUNS.md` (the notebook does this
automatically) to see what's already been tried and what the current best
is to try to beat, instead of re-running a config already covered.

## Evaluation discipline

- 70/15/15 stratified train/val/test split, fixed seed, test set touched exactly once.
- Early stopping on validation macro-F1, not accuracy (the majority class is 57.7% of the data).
- Final report includes the confusion matrix, not just a single aggregate number —
  a model that only gets better at predicting "not aligned" is not progress.
