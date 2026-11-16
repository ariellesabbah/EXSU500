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

  ### Cross-validation (out-of-fold, train/validation set)

Values are point estimates with 95% bootstrap confidence intervals. Threshold chosen on cross-validation for sensitivity ≥ 0.90.

| Model             | AUROC               | AUPRC               | Sensitivity         | Specificity         |   Brier |   Threshold |
|:------------------|:--------------------|:--------------------|:--------------------|:--------------------|--------:|------------:|
| baseline_majority | 0.500 (0.500–0.500) | 0.388 (0.323–0.463) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) |   0.237 |       0.388 |
| baseline_logreg   | 0.750 (0.684–0.816) | 0.655 (0.545–0.772) | 0.936 (0.878–0.987) | 0.374 (0.290–0.462) |   0.202 |       0.295 |
| random_forest     | 0.825 (0.763–0.880) | 0.725 (0.616–0.838) | 0.910 (0.843–0.970) | 0.439 (0.355–0.530) |   0.172 |       0.299 |
| svm               | 0.834 (0.773–0.887) | 0.763 (0.664–0.858) | 0.923 (0.859–0.975) | 0.520 (0.435–0.608) |   0.16  |       0.224 |
| cnn_resnet18_crop | 0.873 (0.819–0.921) | 0.790 (0.690–0.879) | 0.910 (0.842–0.970) | 0.715 (0.635–0.793) |   0.147 |       0.413 |

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
