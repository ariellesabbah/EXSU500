# Colab reproduction — Yarden

I reran the project in Google Colab using a T4 GPU, the existing
data/processed/splits.csv, and the default training settings.

## Results

| Model | Original AUROC | Rerun AUROC | Original AUPRC | Rerun AUPRC |
|---|---:|---:|---:|---:|
| Majority baseline | 0.500 | 0.500 | 0.388 | 0.388 |
| Logistic regression | 0.750 | 0.750 | 0.655 | 0.655 |
| Random forest | 0.825 | 0.825 | 0.725 | 0.725 |
| SVM | 0.834 | 0.834 | 0.763 | 0.763 |
| ResNet-18 | 0.858 | 0.867 | 0.779 | 0.811 |

Original values are from the downloaded outputs/results/comparison_cv.md.

The four simpler models matched the original scores to the displayed
precision. ResNet-18 produced slightly different scores. The cause has
not been established; GPU operations and software-version differences
are possible contributors.

All results are cross-validation results on the 201 development cases.
The 51 held-out test cases were not evaluated.

## Saved files

- notebooks/: executed Colab notebook
- outputs/results/: rerun metrics, predictions, and comparison table
- outputs/figures/: rerun graphs and interpretability figures
- requirements-lock.txt: software versions used for this rerun

## AI assistance

ChatGPT assisted with navigating Colab and GitHub and drafting these
reproduction notes.
