"""Shared helpers: paths, image/mask loading, metrics with confidence intervals.

Every other script imports from here, so changes to how images are read or how
metrics are computed only need to be made once.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # put the TCIA download here (never committed)
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

SEED = 42
N_FOLDS = 5
TEST_FRACTION = 0.20
POSITIVE_CLASS = "malignant"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- images
def load_gray(path: Path) -> np.ndarray:
    """Load an ultrasound image as a 2-D uint8 array.

    Some BrEaST images (extended field of view) have a transparent border; the
    alpha channel is dropped so that border becomes black, as the dataset paper
    describes.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def load_mask(image_dir: Path, mask_field: str, shape: tuple[int, int]) -> np.ndarray:
    """Union of all tumour masks for a case.

    The spreadsheet joins several mask filenames with '&' when an image has more
    than one annotated tumour.
    """
    mask = np.zeros(shape, dtype=bool)
    if not isinstance(mask_field, str) or not mask_field.strip():
        return mask
    for name in mask_field.split("&"):
        name = name.strip()
        if not name:
            continue
        m = np.asarray(Image.open(image_dir / name).convert("L")) > 127
        if m.shape != shape:
            raise ValueError(f"Mask {name} has shape {m.shape}, image has {shape}")
        mask |= m
    return mask


def lesion_bbox(mask: np.ndarray, margin: float = 0.2) -> tuple[int, int, int, int]:
    """Bounding box (r0, r1, c0, c1) around the mask, enlarged by `margin` on each side."""
    rows, cols = np.where(mask)
    if rows.size == 0:  # no mask: use the whole image
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    dr, dc = int((r1 - r0) * margin), int((c1 - c0) * margin)
    return (
        max(0, r0 - dr),
        min(mask.shape[0], r1 + dr),
        max(0, c0 - dc),
        min(mask.shape[1], c1 + dc),
    )


# ---------------------------------------------------------------- metrics
def threshold_for_sensitivity(y_true, y_prob, target: float = 0.90) -> float:
    """Highest threshold that still reaches the target sensitivity.

    Chosen on VALIDATION predictions only, then applied unchanged to the test set.
    A missed cancer costs more than an extra biopsy, so we fix sensitivity first.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    t = thr[ok[0]] if ok.size else thr[-1]
    return float(min(t, 1.0))


def point_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "brier": brier_score_loss(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, threshold: float, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Point estimate plus 95% percentile bootstrap CI for every metric."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    point = point_metrics(y_true, y_prob, threshold)
    keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "brier"]
    samples = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = point_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    out = {"threshold": threshold, "n": int(n), "n_malignant": int(y_true.sum())}
    for k in keys:
        vals = np.asarray(samples[k], dtype=float)
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5]) if vals.size else (np.nan, np.nan)
        out[k] = {"value": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)}
    for k in ["tp", "fp", "tn", "fn"]:
        out[k] = point[k]
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def fmt_ci(d: dict) -> str:
    return f"{d['value']:.3f} ({d['ci_low']:.3f}–{d['ci_high']:.3f})"
