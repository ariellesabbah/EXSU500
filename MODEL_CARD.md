# Model card: <best model name>

> Course prototype for research and educational purposes. Not a medical device. Not for clinical use.

## Intended use
Research and teaching: estimating the probability that a breast lesion on a single B-mode
ultrasound image is malignant, given the radiologist's outline of the lesion.
Not intended for screening, diagnosis or any patient-care decision.

## Training data
BREAST-LESION-USG (BrEaST), TCIA, CC-BY 4.0: 252 lesions from 252 women (154 benign,
98 malignant), collected in Poland 2019–2022 on 4 scanner types. Labels confirmed by
core-needle biopsy or follow-up. Normal cases (n = 4) excluded.

## Evaluation data
Stratified held-out test set of <n> lesions (<n> malignant) from the same dataset,
scored once after all modelling decisions were fixed.

## Metrics (held-out test, 95% bootstrap CI)
| AUROC | AUPRC | Sensitivity | Specificity | Threshold |
|---|---|---|---|---|
| | | | | |

## Known limitations
- Small, single-country dataset; no external validation (yet).
- Requires a manual lesion outline; performance with automatic or rough outlines is unknown.
- One image per lesion; real exams use several views.
- <add what the error analysis / Grad-CAM shows>

## Populations and settings where it should not be used
- Any clinical decision, including deciding whether to biopsy.
- Men, patients under 18, lesions in implants or the axilla, scanners or settings not represented in training.
- Images with measurement markers, Doppler overlays or text burned in.
