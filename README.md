# paper-verdict-finetune

Fine-tune a transformer text encoder, end-to-end, to classify research
papers into `aligned` / `adjacent` / `not aligned` — the pipeline's own
verdict label, used directly as ground truth.

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

The lever none of those rounds could pull on CPU: **full fine-tuning**.
All of the above kept the base encoder frozen and only trained a small
head on top. This repo removes that constraint — on a GPU (Colab), fine-
tune the whole encoder (or its top layers) directly against `verdict`, so
gradient descent can reshape the representation itself around the label,
not just re-project a representation that was never built for it.

No LLM / no prompting anywhere in this pipeline. It's a trained,
deterministic classifier — a fixed function from paper text to label —
not a model you query at inference time with natural-language instructions.

## Structure

```
data/            exported papers.jsonl (id, title, abstract, verdict)
src/
  data.py        load, stratified split, tokenize
  model.py       encoder + classification head; full or partial fine-tune
  train.py       training loop, class-weighted loss, early stopping, checkpoints
  evaluate.py    confusion matrix, macro-F1, silhouette/purity on learned embeddings
notebooks/
  colab_train.ipynb   thin GPU runner: clone this repo, pip install, run train.py + evaluate.py
requirements.txt
docs/
  prior_results.md    summary of the four CPU-only rounds, for context
```

## Quickstart (Colab)

1. Open `notebooks/colab_train.ipynb` in Colab, select a GPU runtime.
2. Run all cells: it clones this branch, installs `requirements.txt`,
   and calls `src/train.py` then `src/evaluate.py`.
3. Results (confusion matrix, macro-F1, silhouette/purity) are written to
   `outputs/` and also displayed inline in the notebook.

## Configuration

See `src/train.py --help`. Key flags:

- `--base-model` : `allenai/specter2_base` (default) | `sentence-transformers/all-MiniLM-L6-v2` | `allenai/scibert_scivocab_uncased`
- `--finetune-mode` : `partial` (default, freeze all but top 2 layers + head) | `full`
- `--loss` : `weighted-ce` (default) | `focal`
- `--seed` : fixed by default (42) for reproducibility

## Evaluation discipline

- 70/15/15 stratified train/val/test split, fixed seed, test set touched exactly once.
- Early stopping on validation macro-F1, not accuracy (the majority class is 57.7% of the data).
- Final report includes the confusion matrix, not just a single aggregate number —
  a model that only gets better at predicting "not aligned" is not progress.
