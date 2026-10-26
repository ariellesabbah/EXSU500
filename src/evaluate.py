"""Shared evaluation: turns out-of-fold and test predictions into saved metrics.

Protocol (the same for every model):
1. Cross-validation on the 80% train/validation set gives an out-of-fold (OOF)
   probability for every training lesion. Model choices are made on these only.
2. The operating threshold is chosen on the OOF probabilities (sensitivity >= 90%).
3. Only with --final is the held-out test set scored, once, with that threshold.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, bootstrap_ci, fmt_ci, save_json, threshold_for_sensitivity

TARGET_SENSITIVITY = 0.90


def load_table_with_splits(features: pd.DataFrame) -> pd.DataFrame:
    from common import SPLITS_CSV

    splits = pd.read_csv(SPLITS_CSV)
    return splits.merge(features, on="Case_ID", how="left", validate="one_to_one")


def sklearn_oof(make_model, data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Out-of-fold probabilities on the train/validation set for an sklearn model factory."""
    tv = data[data.split == "trainval"]
    parts = []
    for fold in sorted(tv.fold.unique()):
        tr, va = tv[tv.fold != fold], tv[tv.fold == fold]
        model = make_model().fit(tr[cols], tr.label)
        parts.append(pd.DataFrame({"Case_ID": va.Case_ID, "label": va.label, "fold": fold,
                                   "prob": model.predict_proba(va[cols])[:, 1]}))
    return pd.concat(parts, ignore_index=True)


def sklearn_test(make_model, data: pd.DataFrame, cols: list[str]):
    """Refit on ALL train/validation lesions, then score the held-out test set once."""
    tv, te = data[data.split == "trainval"], data[data.split == "test"]
    model = make_model().fit(tv[cols], tv.label)
    return model, pd.DataFrame({"Case_ID": te.Case_ID, "label": te.label,
                                "prob": model.predict_proba(te[cols])[:, 1]})


def report(model: str, oof: pd.DataFrame, test: pd.DataFrame | None = None, extra: dict | None = None) -> dict:
    """oof/test: DataFrames with columns Case_ID, label, prob (and fold for oof)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(RESULTS_DIR / f"{model}_oof.csv", index=False)

    per_fold = []
    if "fold" in oof:
        for f, g in oof.groupby("fold"):
            if g.label.nunique() == 2:
                per_fold.append(roc_auc_score(g.label, g.prob))
    thr = threshold_for_sensitivity(oof.label, oof.prob, TARGET_SENSITIVITY)
    cv = bootstrap_ci(oof.label.values, oof.prob.values, thr)
    cv["fold_auroc"] = per_fold
    if extra:
        cv["extra"] = extra
    save_json(cv, RESULTS_DIR / f"{model}_cv_metrics.json")

    print(f"\n[{model}] cross-validation (out-of-fold, n={cv['n']})")
    print(f"  AUROC {fmt_ci(cv['auroc'])}   AUPRC {fmt_ci(cv['auprc'])}")
    if per_fold:
        s = pd.Series(per_fold)
        print(f"  per-fold AUROC {s.mean():.3f} ± {s.std():.3f}")
    print(f"  threshold for sensitivity ≥ {TARGET_SENSITIVITY:.0%}: {thr:.3f} → "
          f"sens {cv['sensitivity']['value']:.2f}, spec {cv['specificity']['value']:.2f}")

    result = {"cv": cv}
    if test is not None:
        test.to_csv(RESULTS_DIR / f"{model}_test.csv", index=False)
        te = bootstrap_ci(test.label.values, test.prob.values, thr)
        save_json(te, RESULTS_DIR / f"{model}_test_metrics.json")
        print(f"\n[{model}] HELD-OUT TEST (n={te['n']}, threshold fixed at {thr:.3f})")
        for k in ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "brier"]:
            print(f"  {k:<12}{fmt_ci(te[k])}")
        result["test"] = te
    return result
