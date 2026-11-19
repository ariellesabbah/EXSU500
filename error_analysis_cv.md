# Error analysis on cross-validation predictions (SRS)

Predictions: out-of-fold CV predictions only (n = 201), original committed run
(outputs_original/results). No test-set data used.
Threshold: the CV threshold for sensitivity >= 90% from each model's metrics file
(SVM 0.224, ResNet-18 0.275). A miss rate of roughly 8-10% is therefore expected by design.
Code: Colab cell written for this analysis (to be committed as a script).

## Summary at the CV threshold
- SVM: TP 72, FN 6, FP 59, TN 64 (sensitivity 0.923, specificity 0.520)
- ResNet-18: TP 72, FN 6, FP 50, TN 73 (sensitivity 0.923, specificity 0.593)
- 6 missed cancers per model; 3 were missed by both.

## False positives by radiologist BI-RADS (benign lesions only)
| BI-RADS | ResNet-18 FP | SVM FP |
|---|---|---|
| 2 | 6/26 (23%) | 11/26 (42%) |
| 3 | 8/29 (28%) | 6/29 (21%) |
| 4a | 13/35 (37%) | 16/35 (46%) |
| 4b | 15/24 (63%) | 17/24 (71%) |
| 4c | 4/4 | 4/4 |
| 5 | 4/5 | 5/5 |

Specificity within BI-RADS 4a-5 (68 benign lesions): ResNet-18 32/68 (47%), SVM 26/68 (38%),
versus 59% and 52% overall. All malignant lesions are in BI-RADS 4a-5, so sensitivity is unchanged.

## Missed cancers
Most missed cancers are BI-RADS 4c-5, but most cancers are (65 of 78). Miss rates are similar
across groups (ResNet-18: 5/65 in 4c-5 vs 1/13 in 4a-4b), so no pattern is established.
No clear lesion-size pattern (mask size is in pixels and image sizes vary by scanner).

## Grad-CAM
The saved localisation file holds per-fold summaries, not per-case values, so it could not
be linked to outcomes. [Add the ratio from the official run.]

## Caveats
- Cross-validation (model-selection) predictions, not test results; to be redone on the test set.
- Small counts, descriptive only, no intervals or significance tests.
- Original-run numbers; the CNN rerun gives somewhat different values.

# to add to model card for error analysis 
In cross-validation at the threshold giving ≥90% sensitivity, false positives rose with radiologist-assigned BI-RADS category (ResNet-18: 23% of benign BI-RADS 2 lesions, 63% of 4b, all 4 of BI-RADS 4c), and specificity within BI-RADS 4a–5 (47%) was lower than overall (59%). Grad-CAM mass fell only modestly more inside the lesion outline than expected by chance (ratio [from the official run]).

"""Error analysis on the cross-validation (out-of-fold) predictions.

Uses CV predictions only (no test-set data). For the SVM and ResNet-18 it reports, at each
model's CV threshold for sensitivity >= 90%: the confusion counts, the missed cancers, and
where false positives fall by radiologist-assigned BI-RADS category. It also saves a figure of
the missed cancers (ResNet-18) with the radiologist's outline in red.

Run from the repository root, after scripts 01-06 have produced data/processed/metadata.csv
and the *_oof.csv files:

    python 07_error_analysis_cv.py                      # committed run  (outputs_original/results)
    python 07_error_analysis_cv.py --results outputs/results

The per-case table it saves contains BI-RADS and age for each case: keep it local, do not commit.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve

MODELS = {"SVM": "svm", "ResNet18": "cnn_resnet18_crop"}
RAW = "data/raw/BrEaST-Lesions_USG-images_and_masks"
META = "data/processed/metadata.csv"


def find_threshold(results_dir, stem, df):
    """CV threshold for sensitivity >= 90%: read from the metrics file, else recompute."""
    try:
        with open(f"{results_dir}/{stem}_cv_metrics.json") as f:
            js = json.load(f)

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if "thresh" in k.lower() and isinstance(v, (int, float)):
                        yield float(v)
                    else:
                        yield from walk(v)

        vals = list(walk(js))
        if vals:
            return vals[0], "read from metrics json"
    except Exception as e:
        print(f"  (could not read metrics json: {e})")
    _, tpr, thr = roc_curve(df["label"], df["prob"])
    return float(thr[np.where(tpr >= 0.90)[0][0]]), "recomputed: highest threshold with sensitivity >= 0.90"


def load_oof(results_dir, stem):
    raw = pd.read_csv(f"{results_dir}/{stem}_oof.csv")
    key = raw["Case_ID"].astype(str).str.lower().str.strip()
    df = pd.DataFrame({"label": raw["label"], "prob": raw["prob"]}).groupby(key).agg(
        label=("label", "first"), prob=("prob", "mean"))
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.lower().map(
            {"malignant": 1, "benign": 0, "1": 1, "0": 0})
    df["label"] = df["label"].astype(int)
    return df


def analyse_model(name, stem, results_dir, meta):
    df = load_oof(results_dir, stem)
    thr, how = find_threshold(results_dir, stem, df)
    df["pred"] = (df["prob"] >= thr).astype(int)
    df["outcome"] = np.select(
        [(df.label == 1) & (df.pred == 1), (df.label == 1) & (df.pred == 0),
         (df.label == 0) & (df.pred == 1), (df.label == 0) & (df.pred == 0)],
        ["TP", "FN", "FP", "TN"], default="")
    counts = df["outcome"].value_counts()
    tp, fn, fp, tn = [int(counts.get(k, 0)) for k in ["TP", "FN", "FP", "TN"]]
    print(f"\n=== {name}: threshold {thr:.3f} ({how}) ===")
    print(f"TP {tp}  FN {fn}  FP {fp}  TN {tn}   "
          f"sensitivity {tp / (tp + fn):.3f}  specificity {tn / (tn + fp):.3f}")
    df = df.join(meta[["BIRADS", "Verification", "Age", "Pixel_size", "mask_px"]])

    print("\nMissed cancers (false negatives), lowest predicted probability first:")
    print(df[df.outcome == "FN"][["prob", "BIRADS", "Verification", "Age", "mask_px"]]
          .sort_values("prob").round(3).to_string())
    print("\nBenign lesions by BI-RADS and outcome:")
    print(pd.crosstab(df[df.label == 0]["BIRADS"], df[df.label == 0]["outcome"]))
    print("\nMalignant lesions by BI-RADS and outcome:")
    print(pd.crosstab(df[df.label == 1]["BIRADS"], df[df.label == 1]["outcome"]))

    sub = df[df["BIRADS"].astype(str).str.lower().isin(["4a", "4b", "4c", "5"]) & (df.label == 0)]
    if len(sub):
        print(f"\nSpecificity within BI-RADS 4a-5: {(sub.outcome == 'TN').sum()}/{len(sub)} "
              f"({(sub.outcome == 'TN').mean():.0%})")
    return df


def plot_missed(df, meta, out_png, max_cases=8):
    keys = list(df.index[df.outcome == "FN"])[:max_cases]
    if not keys:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        try:
            img = Image.open(os.path.join(RAW, meta.loc[k, "Image_filename"])).convert("L")
            ax.imshow(img, cmap="gray")
            mask = Image.open(os.path.join(RAW, meta.loc[k, "Mask_tumor_filename"])).convert("L").resize(img.size)
            ax.contour(np.array(mask) > 0, levels=[0.5], colors="red", linewidths=1)
        except Exception as e:
            print(f"could not draw {k}: {e}")
        ax.set_title(f"{k}  p={df.loc[k, 'prob']:.2f}  BI-RADS {meta.loc[k, 'BIRADS']}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="outputs_original/results",
                    help="folder containing the *_oof.csv and *_cv_metrics.json files")
    args = ap.parse_args()
    results_dir = args.results.rstrip("/")

    meta = pd.read_csv(META)
    meta["key"] = meta["Case_ID"].astype(str).str.lower().str.strip()
    meta = meta.set_index("key")

    errors = {name: analyse_model(name, stem, results_dir, meta) for name, stem in MODELS.items()}

    fn = {m: set(d.index[d.outcome == "FN"]) for m, d in errors.items()}
    a, b = list(fn)
    print(f"\nFalse negatives: {a} {len(fn[a])}, {b} {len(fn[b])}, missed by both {len(fn[a] & fn[b])}")

    tag = "original" if "original" in results_dir else "rerun"
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    plot_missed(errors["ResNet18"], meta, f"outputs/figures/missed_cancers_cv_{tag}.png")
    out_csv = f"outputs/results/error_analysis_cv_{tag}.csv"
    pd.concat(errors, names=["model", "case"]).to_csv(out_csv)
    print(f"saved {out_csv} (BI-RADS/age per case: keep local, do not commit)")


if __name__ == "__main__":
    main()
