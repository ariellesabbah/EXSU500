"""Step 6 - Model comparison table, ROC/PR curves and calibration (reliability) plot.

Reads every *_oof.csv (cross-validation) and, if present, *_test.csv (held-out test)
from outputs/results and writes:
- outputs/results/comparison_<stage>.md   : the table for the README and the manuscript
- outputs/figures/roc_pr_<stage>.png       : ROC and precision-recall curves
- outputs/figures/calibration_<stage>.png  : reliability curves

Run:  python src/06_compare_models.py            (cross-validation results)
      python src/06_compare_models.py --final    (test results, after every --final run)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from common import FIGURES_DIR, RESULTS_DIR, fmt_ci

ORDER = ["baseline_majority", "baseline_logreg", "random_forest", "svm", "cnn_resnet18_crop", "cnn_resnet18_full"]


def main(final: bool) -> None:
    stage = "test" if final else "cv"
    pred_suffix = "test" if final else "oof"
    models = [m for m in ORDER if (RESULTS_DIR / f"{m}_{pred_suffix}.csv").exists()]
    models += sorted({p.name.rsplit(f"_{pred_suffix}.csv", 1)[0] for p in RESULTS_DIR.glob(f"*_{pred_suffix}.csv")} - set(models))
    if not models:
        raise SystemExit(f"No *_{pred_suffix}.csv files in {RESULTS_DIR}. Run the model scripts first.")

    rows = []
    for m in models:
        met = json.loads((RESULTS_DIR / f"{m}_{stage}_metrics.json").read_text())
        rows.append({"Model": m, "AUROC": fmt_ci(met["auroc"]), "AUPRC": fmt_ci(met["auprc"]),
                     "Sensitivity": fmt_ci(met["sensitivity"]), "Specificity": fmt_ci(met["specificity"]),
                     "Brier": f"{met['brier']['value']:.3f}", "Threshold": f"{met['threshold']:.3f}"})
    table = pd.DataFrame(rows)
    label = "Held-out test set" if final else "Cross-validation (out-of-fold, train/validation set)"
    md = (f"### {label}\n\nValues are point estimates with 95% bootstrap confidence intervals. "
          "Threshold chosen on cross-validation for sensitivity ≥ 0.90.\n\n" + table.to_markdown(index=False))
    (RESULTS_DIR / f"comparison_{stage}.md").write_text(md)
    print(md)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = {m: pd.read_csv(RESULTS_DIR / f"{m}_{pred_suffix}.csv") for m in models}
    prevalence = next(iter(preds.values())).label.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for m, d in preds.items():
        if d.prob.nunique() < 2:
            continue  # a constant score (majority baseline) IS the dotted chance line; drawing it adds a fake diagonal
        fpr, tpr, _ = roc_curve(d.label, d.prob)
        prec, rec, _ = precision_recall_curve(d.label, d.prob)
        auc = json.loads((RESULTS_DIR / f"{m}_{stage}_metrics.json").read_text())
        a1.plot(fpr, tpr, drawstyle="steps-post", label=f"{m} (AUROC {auc['auroc']['value']:.2f})")
        a2.plot(rec, prec, drawstyle="steps-post", label=f"{m} (AUPRC {auc['auprc']['value']:.2f})")
    a1.plot([0, 1], [0, 1], "k:", lw=1, label="chance = majority baseline (0.50)")
    a2.axhline(prevalence, color="k", ls=":", lw=1, label=f"chance = majority baseline ({prevalence:.2f})")
    a1.set(xlabel="1 − specificity", ylabel="Sensitivity", title=f"ROC ({label.split(' (')[0]})")
    a2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision–recall", ylim=(0, 1.02))
    a1.legend(fontsize=7, loc="lower right")
    a2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_pr_{stage}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for m, d in preds.items():
        if d.prob.nunique() < 3:
            continue
        frac, mean_pred = calibration_curve(d.label, d.prob, n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=m)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="Predicted probability of malignancy", ylabel="Observed fraction malignant",
           title="Reliability (calibration)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"calibration_{stage}.png", dpi=200)
    plt.close(fig)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true")
    main(p.parse_args().final)
