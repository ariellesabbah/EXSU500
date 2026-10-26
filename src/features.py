"""Handcrafted features from the image and the radiologist's tumour outline.

Three groups, chosen to mirror what radiologists look at under BI-RADS:
- shape    : size, "taller than wide", circularity, solidity (margin irregularity)
- intensity: echogenicity inside the lesion, contrast with the surrounding tissue,
             posterior shadowing/enhancement (tissue directly below the lesion)
- texture  : grey-level co-occurrence (GLCM) statistics inside the lesion box

Run once to cache:  python src/features.py   -> data/processed/features.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

from common import METADATA_CSV, PROCESSED_DIR, load_gray, load_mask

FEATURES_CSV = PROCESSED_DIR / "features.csv"
SHAPE_FEATURES = ["area_cm2", "depth_width_ratio", "circularity", "solidity", "eccentricity", "extent"]


def _pixel_size_cm(value) -> float | None:
    try:
        v = float(str(value).split("&")[0].replace(",", "."))
        return v if v > 0 else None
    except ValueError:
        return None


def extract(img: np.ndarray, mask: np.ndarray, pixel_cm: float | None) -> dict:
    f: dict[str, float] = {}

    # --- shape (largest connected tumour region)
    lab = label(mask)
    region = max(regionprops(lab), key=lambda r: r.area)
    r0, c0, r1, c1 = region.bbox
    area_px = float(mask.sum())
    f["area_cm2"] = area_px * pixel_cm**2 if pixel_cm else np.nan
    f["area_px"] = area_px
    f["depth_width_ratio"] = (r1 - r0) / max(c1 - c0, 1)       # >1 = taller than wide
    f["circularity"] = 4 * np.pi * region.area / max(region.perimeter, 1) ** 2
    f["solidity"] = region.solidity                             # low = lobulated/spiculated
    f["eccentricity"] = region.eccentricity
    f["extent"] = region.extent

    # --- intensity
    inside = img[mask].astype(float)
    ring = ndimage.binary_dilation(mask, iterations=10) & ~mask
    outside = img[ring].astype(float) if ring.any() else inside
    h = r1 - r0
    below = np.zeros_like(mask)
    below[r1 : min(img.shape[0], r1 + h // 2), c0:c1] = True
    below &= ~mask
    under = img[below].astype(float) if below.any() else outside
    f["mean_in"] = inside.mean()
    f["std_in"] = inside.std()
    f["skew_in"] = float(stats.skew(inside))
    f["kurt_in"] = float(stats.kurtosis(inside))
    f["contrast_vs_surround"] = inside.mean() - outside.mean()
    f["posterior_ratio"] = (under.mean() + 1) / (outside.mean() + 1)  # <1 shadowing, >1 enhancement

    # --- texture: GLCM inside the tight lesion box, 32 grey levels
    crop = img[r0:r1, c0:c1]
    q = (crop // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        f[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    return f


def build_feature_table(meta: pd.DataFrame | None = None) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV) if meta is None else meta
    rows = []
    for _, r in meta.iterrows():
        d = Path(r["image_dir"])
        img = load_gray(d / r["Image_filename"])
        mask = load_mask(d, r["Mask_tumor_filename"], img.shape)
        feats = extract(img, mask, _pixel_size_cm(r.get("Pixel_size")))
        rows.append({"Case_ID": r["Case_ID"], **feats})
    return pd.DataFrame(rows)


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        df = build_feature_table()
        df.to_csv(FEATURES_CSV, index=False)
        print(f"Saved {FEATURES_CSV}")
    return pd.read_csv(FEATURES_CSV)


if __name__ == "__main__":
    if FEATURES_CSV.exists():
        FEATURES_CSV.unlink()
    print(load_features().describe().T.round(3).to_string())
