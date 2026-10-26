"""Step 3 - Baselines (the floor every other model must beat).

a) Majority class: predicts the malignancy prevalence for everyone (AUROC = 0.5).
b) Logistic regression on two shape measures a radiologist reads off the outline:
   lesion size and the depth/width ("taller than wide") ratio.

Run:  python src/03_baseline.py            (cross-validation only)
      python src/03_baseline.py --final    (also scores the test set - do this ONCE, at the end)
"""
import argparse

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import set_seed
from evaluate import load_table_with_splits, report, sklearn_oof, sklearn_test
from features import load_features

COLS = ["area_px", "depth_width_ratio"]


def majority():
    return DummyClassifier(strategy="prior")


def logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def main(final: bool) -> None:
    set_seed()
    data = load_table_with_splits(load_features())
    prevalence = data.loc[data.split == "trainval", "label"].mean()
    for name, factory in [("baseline_majority", majority), ("baseline_logreg", logistic)]:
        oof = sklearn_oof(factory, data, COLS)
        test = sklearn_test(factory, data, COLS)[1] if final else None
        if name == "baseline_majority":
            # One constant probability for everyone (AUROC exactly 0.5), not a slightly different one per fold.
            oof["prob"] = prevalence
            if test is not None:
                test["prob"] = prevalence
        report(name, oof, test, extra={"features": COLS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="score the held-out test set (once!)")
    main(p.parse_args().final)
