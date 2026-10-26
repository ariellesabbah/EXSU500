# EXSU500_group_3
Group: Arielle Sabbah, Yarden Aronson, Sasha Socolow, Altara Wiryadi

## Dataset

This project uses the *BREAST-LESION-USG* dataset obtained from The Cancer Imaging Archive (TCIA)

Source:
Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. (2024). A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis (Breast-Lesions-USG) (Version 1) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9WKK-Q141.

256 B-mode images from 256 women (Poland, 2019–2022, 4 scanner types), each with a radiologist's tumour
outline. After excluding 4 normal cases: **252 lesions — 154 benign, 98 malignant (38.9%)**.

## Problem: 

Our project aims to develop and compare machine-learning models for predicting the malignancy of breast lesions from ultrasound images.

Why it matters: prompt and accurate diagnosis of breast lesions strongly influence follow-up and biopsy requirement. An AI model capable of predicting the malignancy of a lesion could potentially support more consistent and efficient image interpretation, and minimize unnecessary biopsies and associated complications.

## Task type: Binary Image Classification

Input:
Breast ultrasound image

Output:
Malignant or Benign

## Models compared

1. **Baselines:** majority class; logistic regression on lesion size and depth/width ratio
2. **Classical ML:** random forest and SVM on 17 handcrafted shape, intensity and texture features
3. **Deep learning:** ImageNet-pretrained ResNet-18, fine-tuned on the lesion region (3 seeds × 5 folds), with Grad-CAM

## Evaluation protocol

- Splits declared before modelling (`data/processed/splits.csv`): stratified **20% held-out test set** (51 lesions)
  and **5 cross-validation folds** on the remaining 201. One image per patient, so splits are patient-level.
- All tuning and model selection use cross-validation only. The test set is scored once, at the end.
- Metrics: AUROC and AUPRC, plus sensitivity, specificity, PPV and NPV at a threshold chosen on cross-validation
  for ≥ 90% sensitivity. 95% bootstrap confidence intervals throughout; CNN variability across seeds.
- Interpretability/calibration: permutation feature importance, Grad-CAM, reliability curves.

## Results

*Preliminary cross-validation results (train/validation set only, n = 201). Held-out test results to follow.*

| Model | AUROC (95% CI) | AUPRC (95% CI) | Specificity at ~92% sensitivity |
|---|---|---|---|
| Majority class | 0.500 | 0.388 (0.323–0.463) | 0.00 |
| Logistic regression | 0.750 (0.684–0.816) | 0.655 (0.545–0.772) | 0.37 |
| Random forest | 0.819 (0.755–0.876) | 0.716 (0.607–0.830) | 0.44 |
| SVM | 0.834 (0.773–0.887) | 0.763 (0.664–0.858) | 0.52 |
| ResNet-18 | 0.858 (0.803–0.909) | 0.779 (0.682–0.869) | 0.59 |

## Repository structure

```
├── README.md              this file
├── GETTING_STARTED.md     step-by-step guide for running everything
├── MODEL_CARD.md          model card for the best model
├── LICENSE                MIT
├── requirements.txt       Python packages
├── data/
│   ├── raw/               the TCIA download goes here (git-ignored)
│   └── processed/
│       └── splits.csv     the fixed train/validation/test split (committed)
├── notebooks/
│   └── run_on_colab.ipynb runs the whole pipeline on Google Colab
├── src/                   scripts, run in numbered order
│   ├── common.py          shared settings, image loading, metrics
│   ├── evaluate.py        shared cross-validation and test scoring
│   ├── features.py        handcrafted features
│   ├── 01_prepare_data.py
│   ├── 02_make_splits.py
│   ├── 03_baseline.py
│   ├── 04_classical_ml.py
│   ├── 05_cnn.py
│   └── 06_compare_models.py
└── outputs/
    ├── results/           metric tables and predictions (committed)
    └── figures/           ROC, calibration, feature importance, Grad-CAM (committed)
```
## How to reproduce

1. Download the images (.zip) and clinical data (.xlsx) from https://doi.org/10.7937/9WKK-Q141.
   Unzip both into `data/raw/`.
2. Install packages: `pip install -r requirements.txt` (Python 3.10+). On Colab, open `notebooks/run_on_colab.ipynb`.
3. Run the scripts in order:

```bash
python src/01_prepare_data.py      # checks files, prints dataset summary
python src/02_make_splits.py       # skipped if splits.csv already exists
python src/03_baseline.py
python src/04_classical_ml.py
python src/05_cnn.py --gradcam     # GPU recommended
python src/06_compare_models.py
```

Add `--final` to scripts 03–06 to score the held-out test set. See `GETTING_STARTED.md` for details.

## Contributions (CRediT)

*To be completed*

## Use of generative AI

Claude (Anthropic) was used to draft starter code and that all code, results and citations were checked modified and are understood by the team.

## Licence

Code: MIT (see `LICENSE`). Data: CC-BY 4.0, owned by the dataset authors and not redistributed here.
