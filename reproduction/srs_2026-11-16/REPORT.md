# Reproduction check (Colab, 2026-11-16) — SRS

Environment: Google Colab, T4 GPU, Python 3.13, main-branch code, no `--final`.
Scripts 03–06 ran start to finish. CNN training was paused and resumed once;
all 15 checkpoints were trained in this session (~1.5–2 min each).

## Results vs committed
- Majority, logistic regression, SVM: exact match.
- Random forest: AUROC 0.825 vs 0.819 (AUPRC 0.725 vs 0.716).
- ResNet-18: AUROC 0.873 (0.819–0.921) vs 0.858 (0.803–0.909);
  specificity 0.715 vs 0.593 at ≥90% sensitivity; threshold 0.413 vs 0.275.
  Likely run-to-run variability (CIs overlap heavily).
- Per-seed CNN AUROC: 0.862 / 0.846 / 0.832 (mean 0.847); the table value
  (0.873) combines the three seeds.
- Grad-CAM: 42% of heatmap mass inside the lesion outline, which covers
  35% of the input (ratio 1.19).

## Questions for the Modeling Lead
1. Does the logistic regression use all 17 features? The log lists 17, the README says size + depth/width ratio.
2. Is the best epoch picked on the same fold that is scored?
3. Which checkpoints/ensemble will `--final` use?
4. Is the random forest seed fixed? Should GPU determinism be enabled?

## Problems I hit (first-time user)
- `FileNotFoundError: No file matching '*.xlsx'` in 01_prepare_data.py: the xlsx must sit directly in data/raw.
- `Could not open requirements file` : my Drive upload was missing requirements.txt, src and outputs.
- `can't open file '/content/src/05_cnn.py'`: a new Colab session starts in /content; the `%cd` line must be rerun first.
- GPU stuck on "Connecting" for a while; worked on retry.
- cuBLAS "no current CUDA context" UserWarning during the CNN: harmless.
- Suggest GETTING_STARTED.md mention: upload the whole folder, `%cd` after every restart, and the data/raw layout.
