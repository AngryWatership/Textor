# Prior results (CPU-only, frozen embeddings)

Context for why this repo exists. All numbers below use `verdict`
(aligned/adjacent/not aligned) as ground truth, on the same 2,093-paper
corpus in `data/papers.jsonl`. Class sizes: not aligned 1,208 (57.7%),
adjacent 667 (31.9%), aligned 218 (10.4%) -- a random-label k-NN(10)
purity floor from class sizes alone is ~0.446.

All of the following kept the base embedding **frozen** and only varied
the projection/transform on top of it (round 3) or the instruction given
to a frozen encoder (round 4, approach 1) or a small trained head on top
of frozen features (round 4, approach 2). Metrics are out-of-fold
(5-fold CV) unless noted, never evaluated on data used to fit the method.

| Method | Silhouette | k-NN(10) purity |
|---|---|---|
| Raw MiniLM embedding (384-d), no projection | -0.040 | 0.579 |
| Current 2D UMAP map (unsupervised) | -0.107 | 0.560 |
| PCA, top-2 components (unsupervised eigenvectors) | -0.099 | 0.506 |
| LDA, 2D (generalized eigenvalue problem, supervised) | 0.036 | 0.558 |
| PLS-DA, 2D | 0.010 | 0.552 |
| Supervised UMAP, target_weight=0.5 | 0.019 | 0.578 |
| Supervised UMAP, target_weight=1.0 | FAILED -- 15/2093 points NaN | -- |
| Contrastive projection head (SupCon, 32-d, frozen MiniLM input) | 0.017 | **0.613** |
| Contrastive head, in-sample (optimistic, not held out) | 0.790 | 1.000 |
| Instruction-conditioned re-embedding (e5-large-instruct, 1024-d, raw) | -0.061 | 0.582 |
| Instruction embedding -> CV LDA 2D | 0.005 | 0.517 |

## What this says

1. **The embedding's dominant variance directions have nothing to do
   with `verdict`.** PCA's top-10 components explain only 11.5%, 6.0%,
   3.4%, 3.0%, 3.0%... of total variance (long flat tail), and using the
   top 2 (pca2) actively *hurts* separation versus doing nothing.
2. **No projection of the frozen embedding, linear or nonlinear, beat a
   weak ceiling.** LDA (an eigenvalue method) and supervised UMAP (a
   nonlinear manifold method) land in the same narrow band. If a concept
   isn't a direction (or even a nonlinear region) in a given embedding,
   no downstream projection recovers it -- consistent with the standard
   "linear representation hypothesis" / probing-classifier literature.
3. **Naming the aspect explicitly to an instruction-tuned encoder didn't
   help**, and if anything underperformed the generic embedding slightly.
   The instruction changes how the model reads the text but can't
   manufacture signal that isn't in the title/abstract to begin with.
4. **The only method that beat the ~0.58 ceiling used direct supervision**
   (the contrastive head, trained against `verdict` labels, purity 0.613)
   -- but it was still a small head on top of a *frozen* base embedding.
   It never got to reshape the representation itself.

That's the gap this repo is built to close: full (or partial) fine-tuning
of the base encoder on a GPU, so the representation can be reshaped
directly around `verdict`, rather than re-projected after the fact.
