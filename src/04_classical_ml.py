"""Step 4 - Classical machine learning on handcrafted shape, intensity and texture features.

Two model families, each tuned by grid search on the 5 cross-validation folds:
- Random forest
- Support vector machine (RBF kernel)

Hyperparameters are picked by mean cross-validated AUROC on the train/validation
set only. Feature importance (permutation, on the folds) is saved as the
interpretability artifact for this model family.

Run:  python src/04_classical_ml.py            (cross-validation only)
      python src/04_classical_ml.py --final    (also scores the test set - ONCE, at the end)
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FIGURES_DIR, RESULTS_DIR, SEED, set_seed
from evaluate import load_table_with_splits, report, sklearn_oof, sklearn_test
from features import load_features

DROP = {"Case_ID", "label", "split", "fold", "area_px"}  # area_px duplicates area_cm2

MODELS = {
    "random_forest": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED))]),
        {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]},
    ),
    "svm": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("scale", StandardScaler()),
                  # SVMs output scores, not probabilities; Platt scaling (sigmoid) turns them into probabilities
                  ("clf", CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                                                 method="sigmoid", cv=3, ensemble=False))]),
        {"clf__estimator__C": [0.1, 1, 10, 100], "clf__estimator__gamma": ["scale", 0.01, 0.1]},
    ),
}


def tune(pipe, grid, tv: pd.DataFrame, cols):
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=PredefinedSplit(tv.fold.values), n_jobs=-1)
    search.fit(tv[cols], tv.label)
    print(f"  best params {search.best_params_}  (mean CV AUROC {search.best_score_:.3f})")
    return search.best_params_


def importance_plot(name, factory, tv, cols):
    """Permutation importance measured on each held-out validation fold, then averaged.

    Measuring it on the lesions the model was trained on would overstate memorised features.
    """
    per_fold = []
    for fold in sorted(tv.fold.unique()):
        tr, va = tv[tv.fold != fold], tv[tv.fold == fold]
        model = factory().fit(tr[cols], tr.label)
        imp = permutation_importance(model, va[cols], va.label, scoring="roc_auc", n_repeats=20, random_state=SEED)
        per_fold.append(imp.importances_mean)
    s = pd.Series(pd.DataFrame(per_fold).mean().values, index=cols).sort_values()
    s.rename("auroc_drop").to_csv(RESULTS_DIR / f"{name}_feature_importance.csv", index_label="feature")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(cols) + 1))
    s.plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Drop in validation AUROC when feature is shuffled")
    ax.set_title(f"{name}: permutation feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_feature_importance.png", dpi=200)
    plt.close(fig)


def main(final: bool) -> None:
    set_seed()
    data = load_table_with_splits(load_features())
    cols = [c for c in data.columns if c not in DROP]
    tv = data[data.split == "trainval"]
    print(f"{len(cols)} features: {', '.join(cols)}")

    for name, (pipe, grid) in MODELS.items():
        print(f"\n== {name}")
        best = tune(pipe, grid, tv, cols)
        factory = lambda: clone(pipe).set_params(**best)  # noqa: E731
        oof = sklearn_oof(factory, data, cols)
        test = sklearn_test(factory, data, cols)[1] if final else None
        importance_plot(name, factory, tv, cols)
        report(name, oof, test, extra={"best_params": best, "features": cols})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="score the held-out test set (once!)")
    main(p.parse_args().final)
