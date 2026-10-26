"""Step 2 - Declare the data splits ONCE, before any modelling.

- 20% of lesions are held out as the TEST set, stratified by class.
  It is used exactly once, at the very end (scripts need --final to touch it).
- The remaining 80% is divided into 5 stratified cross-validation folds,
  used for tuning and model selection.

BrEaST has one image per patient, so a lesion-level split is already a
patient-level split: no patient can appear in both training and test.

Output: data/processed/splits.csv  (Case_ID, label, split, fold)
Run:    python src/02_make_splits.py
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import METADATA_CSV, N_FOLDS, SEED, SPLITS_CSV, TEST_FRACTION


def main() -> None:
    if SPLITS_CSV.exists():
        print(f"{SPLITS_CSV} already exists. Splits are fixed for the whole project; "
              "delete the file yourself only if the team agrees to re-split.")
        return

    meta = pd.read_csv(METADATA_CSV)
    if meta["Case_ID"].duplicated().any():
        raise ValueError("Duplicate Case_IDs: check for repeated patients before splitting")

    trainval, test = train_test_split(
        meta, test_size=TEST_FRACTION, stratify=meta["label"], random_state=SEED
    )
    splits = meta[["Case_ID", "label"]].copy()
    splits["split"] = "trainval"
    splits.loc[splits.Case_ID.isin(test.Case_ID), "split"] = "test"
    splits["fold"] = -1

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tv = splits[splits.split == "trainval"]
    for fold, (_, val_idx) in enumerate(skf.split(tv, tv["label"])):
        splits.loc[tv.index[val_idx], "fold"] = fold

    splits.to_csv(SPLITS_CSV, index=False)
    summary = splits.groupby(["split", "fold"])["label"].agg(n="size", malignant="sum")
    print(summary.to_string())
    print(f"\nSaved {SPLITS_CSV}. Commit this file so every team member uses the same split.")


if __name__ == "__main__":
    main()
