# Getting started: benign vs malignant breast ultrasound

This guide takes the team from an empty repository to a results table, one step at a time.
Run the scripts in order. Each one prints what it did and saves its output for the next.

```
src/
  common.py            shared settings, image loading, metrics with 95% CIs
  evaluate.py          shared cross-validation and test-set scoring
  features.py          handcrafted shape / intensity / texture features
  01_prepare_data.py   read spreadsheet, drop normal cases, check every file
  02_make_splits.py    fix the test set and 5 CV folds - run ONCE
  03_baseline.py       model 1: majority class + logistic regression
  04_classical_ml.py   model 2: random forest and SVM on handcrafted features
  05_cnn.py            model 3: fine-tuned ResNet-18 with Grad-CAM (needs a GPU)
  06_compare_models.py comparison table, ROC/PR curves, calibration plot
```

---

## Step 0 - Put the code in your team repository (data lead)

1. Copy everything in this folder into your GitHub repository and commit it.
   Each member should then commit their own changes from their own account,
   because commit history is graded.
2. Add an MIT licence (GitHub: *Add file > Create new file > LICENSE > Choose a licence template*).
   MIT is compatible with the dataset's CC-BY 4.0 licence.
3. `.gitignore` already keeps the dataset, model weights and checkpoints out of the repository.

## Step 1 - Download the data (data lead, ~10 min)

1. Open the dataset page: <https://doi.org/10.7937/9WKK-Q141>
2. Download the two files: the **images and masks (.zip)** and the **clinical data (.xlsx)**.
3. Create the folder `data/raw/` in the project, unzip the images into it, and put the `.xlsx` next to them.
   Folder names inside `data/raw/` do not matter; the script searches for the files.

What you should have: 256 images named `case001.png` ... plus masks named `case001_tumor.png`.

## Step 2 - Set up the environment

**Option A - Google Colab (recommended, free GPU).**
Upload the project folder (with `data/raw/`) to your Google Drive, then in a new notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/breast-us-classifier   # change to your folder
!pip install -q -r requirements.txt
```

Set *Runtime > Change runtime type > T4 GPU* before Step 6.

**Option B - your own computer.** With Python 3.10+:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3 - Prepare the data and look at it

```bash
python src/01_prepare_data.py
```

This keeps the 252 lesions (154 benign, 98 malignant) and drops the 4 normal cases.
It checks that every image and mask opens, and prints the numbers you need for the
**Data** section of the midterm report: class balance, image sizes, biopsy vs follow-up
verification, the BI-RADS-by-class table, and age.
Copy that printout into your report notes.

Open 10 to 20 images yourself before going further (`data/raw/...`), alongside their masks.

## Step 4 - Fix the splits (do this once, as a team)

```bash
python src/02_make_splits.py
```

- **20% test set** (about 51 lesions), stratified by class. It is not touched until the very end.
- **5 cross-validation folds** on the other 80%, used for every tuning decision.
- Each patient has one image here, so this is already a patient-level split.

**Commit `data/processed/splits.csv` straight away** so everyone uses the same split.
The script refuses to overwrite it.

## Step 5 - Baseline and classical models (no GPU needed, ~2 min)

```bash
python src/03_baseline.py
python src/04_classical_ml.py
```

The first run builds `data/processed/features.csv`: 17 handcrafted features from the image
and the radiologist's tumour outline.

- **Shape:** size, depth/width ratio, circularity, solidity
- **Intensity:** echogenicity, contrast with the surrounding tissue, posterior shadowing
- **Texture:** GLCM statistics

Each script prints cross-validated AUROC and AUPRC with 95% confidence intervals, plus
sensitivity and specificity. The threshold is chosen for 90% sensitivity.
Feature-importance plots are saved to `outputs/figures/`.

## Step 6 - The CNN (GPU, ~20-40 min on a Colab T4)

```bash
python src/05_cnn.py --gradcam
```

This trains a ResNet-18 pretrained on ImageNet, five times (one per fold), for each of 3
random seeds. That gives the seed-to-seed variability the spec requires.
By default it sees the lesion crop with a 20% margin.
To compare against the whole image, run `--input full`.
Model weights are saved under `outputs/checkpoints/`, which is git-ignored.
The Grad-CAM figure goes to `outputs/figures/`.

If Colab disconnects, rerun the same command. Finished folds are reloaded, not retrained.

## Step 7 - Compare models (cross-validation)

```bash
python src/06_compare_models.py
```

Writes `outputs/results/comparison_cv.md` plus ROC/PR curves and a calibration plot.
Use the cross-validation numbers to make every decision: which features, which input,
which hyperparameters. All of these can change freely at this stage.

## Step 8 - Score the test set ONCE (end of November, after everything is frozen)

Only when the team agrees nothing else will change:

```bash
python src/03_baseline.py --final
python src/04_classical_ml.py --final
python src/05_cnn.py --final --gradcam
python src/06_compare_models.py --final
```

`comparison_test.md` is your manuscript's results table.
If you change a model after seeing test results, say so honestly in the manuscript.
Otherwise the test score is no longer an unbiased estimate, which is the leakage the spec penalises.

---

## Design choices you will need to explain in the report

| Choice | Why | Where |
|---|---|---|
| Normal cases excluded | Binary task is about characterising a lesion; normals have no mask | `01_prepare_data.py` |
| Stratified 80/20 + 5-fold CV | Small dataset (252): CV for tuning, untouched test for the final estimate | `02_make_splits.py` |
| Threshold for 90% sensitivity, chosen on CV | Missing a cancer costs more than an extra biopsy; the threshold is fixed before testing | `common.py` |
| AUROC + AUPRC + bootstrap 95% CIs | Spec requirement for imbalanced data (39% malignant) | `common.py` |
| Class-balanced weights / loss | Counter the 154:98 imbalance | all model scripts |
| BI-RADS **not** used as an input | It is the radiologist's malignancy judgement, so using it would leak the answer | everywhere |
| Lesion crop uses the radiologist's outline | Limitation: a deployed model would need a detector or the user to outline the lesion | `features.py`, `05_cnn.py` |

Two points to state as limitations:

- The cross-validation estimates are slightly optimistic. The same folds pick the hyperparameters (classical models) and the stopping epoch (CNN). The test set is the unbiased number.
- The data come from one country and 4 scanner types. An external test set such as BUS-BRA would show how well the model generalises.

## Who does what (suggested)

- **Data lead:** Steps 1, 3, 4. Writes the data description and limitations.
- **Modelling lead:** Steps 5 and 6. Tunes the feature set and CNN settings on CV.
- **Writing lead:** Turns the printouts and figures into the Data and Methods sections. Keeps the model card up to date.
- **Interface lead (optional bonus):** A Gradio app that loads the best checkpoint and shows the probability and Grad-CAM. It must carry the prototype notice.

## Still to add before the final submission

- `MODEL_CARD.md` for your best model (a template is included)
- `requirements-lock.txt` with exact versions (last cell of the Colab notebook)
- A CRediT contribution statement and a Use of Generative AI statement in the README
- The results table and data download instructions in the README
