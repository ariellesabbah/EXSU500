"""Step 1 - Read the BrEaST spreadsheet, keep benign/malignant lesions, check files.

Input : data/raw/  (the TCIA .xlsx and the unzipped image folder, anywhere inside)
Output: data/processed/metadata.csv  and a printed dataset summary

Run:  python src/01_prepare_data.py
"""
from pathlib import Path

import pandas as pd

from common import METADATA_CSV, POSITIVE_CLASS, PROCESSED_DIR, RAW_DIR, load_gray, load_mask


def find_one(pattern: str) -> Path:
    hits = sorted(RAW_DIR.rglob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No file matching '{pattern}' under {RAW_DIR}. "
            "Download the dataset from TCIA (see GETTING_STARTED.md) and unzip it there."
        )
    return hits[0]


def main() -> None:
    xlsx = find_one("*.xlsx")
    df = pd.read_excel(xlsx)
    df.columns = [c.strip() for c in df.columns]
    # The TCIA spreadsheet uses "CaseID"; the dataset paper writes "Case_ID".
    df = df.rename(columns={"CaseID": "Case_ID"})
    print(f"Spreadsheet: {xlsx.name}  ({len(df)} rows)")

    needed = ["Case_ID", "Image_filename", "Mask_tumor_filename", "Classification"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from spreadsheet: {missing}. Found: {list(df.columns)}")

    # Images and masks sit together in one folder; locate it from the first image.
    image_dir = find_one(str(df["Image_filename"].iloc[0])).parent
    print(f"Image folder: {image_dir}")

    df["Classification"] = df["Classification"].astype(str).str.strip().str.lower()
    print("\nAll cases by class:\n" + df["Classification"].value_counts().to_string())

    # Binary task: drop normal tissue (no lesion, no mask).
    df = df[df["Classification"].isin(["benign", "malignant"])].copy()
    df["label"] = (df["Classification"] == POSITIVE_CLASS).astype(int)

    # Check every image and mask opens and that masks match image size.
    heights, widths, areas = [], [], []
    for _, row in df.iterrows():
        img = load_gray(image_dir / row["Image_filename"])
        mask = load_mask(image_dir, row["Mask_tumor_filename"], img.shape)
        if not mask.any():
            raise ValueError(f"Case {row['Case_ID']} has no tumour mask")
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        areas.append(int(mask.sum()))
    df["img_h"], df["img_w"], df["mask_px"] = heights, widths, areas
    df["image_dir"] = str(image_dir)

    keep = ["Case_ID", "Image_filename", "Mask_tumor_filename", "Classification", "label",
            "BIRADS", "Verification", "Age", "Pixel_size", "img_h", "img_w", "mask_px", "image_dir"]
    keep = [c for c in keep if c in df.columns]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(METADATA_CSV, index=False)

    # Summary for the Data section of the midterm report.
    print(f"\nKept {len(df)} lesions: {int(df.label.sum())} malignant, {int((1 - df.label).sum())} benign "
          f"({df.label.mean():.1%} malignant)")
    print(f"Image size: height {min(heights)}–{max(heights)} px, width {min(widths)}–{max(widths)} px")
    if "Verification" in df:
        print("\nReference standard:\n" + pd.crosstab(df["Verification"], df["Classification"]).to_string())
    if "BIRADS" in df:
        print("\nBI-RADS vs final class:\n" + pd.crosstab(df["BIRADS"].astype(str), df["Classification"]).to_string())
    if "Age" in df:
        age = pd.to_numeric(df["Age"], errors="coerce")
        print(f"\nAge: median {age.median():.0f}, range {age.min():.0f}–{age.max():.0f}, missing {age.isna().sum()}")
    print(f"\nSaved {METADATA_CSV}")


if __name__ == "__main__":
    main()
