"""Step 5 - Deep learning: fine-tune an ImageNet-pretrained ResNet-18.

Input  : the lesion region cropped from the ultrasound image using the tumour
         outline, with a 20% margin of surrounding tissue (use --input full for the
         whole image instead; compare both if you have time).
Imbalance: positive-class weighting in the loss.
Protocol: one network per cross-validation fold, early-stopped on that fold's
         validation AUROC -> out-of-fold predictions. Repeated for several seeds.
         Test predictions (--final) = average of the 5 fold models, per seed.
Interpretability: Grad-CAM heatmaps on the last convolutional block (--gradcam).

Needs a GPU for reasonable speed (Colab: Runtime > Change runtime type > T4 GPU).

Run:  python src/05_cnn.py                     # train, cross-validation results
      python src/05_cnn.py --gradcam           # + Grad-CAM on validation images
      python src/05_cnn.py --final --gradcam   # reuse saved models, score test ONCE
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (FIGURES_DIR, METADATA_CSV, RESULTS_DIR, ROOT, SPLITS_CSV, bootstrap_ci,
                    lesion_bbox, load_gray, load_mask, save_json, set_seed)
from evaluate import report

CKPT_DIR = ROOT / "outputs" / "checkpoints"
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
class LesionDataset(Dataset):
    def __init__(self, table: pd.DataFrame, train: bool, input_mode: str):
        self.rows = table.reset_index(drop=True)
        self.input_mode = input_mode
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               transforms.ColorJitter(brightness=0.2, contrast=0.2)] if train else []
        self.tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *aug,
                                      transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.cache: dict[int, Image.Image] = {}

    def __len__(self):
        return len(self.rows)

    def raw(self, i: int) -> Image.Image:
        if i not in self.cache:
            r = self.rows.iloc[i]
            d = Path(r.image_dir)
            img = load_gray(d / r.Image_filename)
            if self.input_mode == "crop":
                mask = load_mask(d, r.Mask_tumor_filename, img.shape)
                r0, r1, c0, c1 = lesion_bbox(mask, margin=0.2)
                img = img[r0:r1, c0:c1]
            self.cache[i] = Image.fromarray(img).convert("RGB")
        return self.cache[i]

    def __getitem__(self, i):
        return self.tf(self.raw(i)), torch.tensor(float(self.rows.iloc[i].label))


def build_model() -> nn.Module:
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, 1))
    return net.to(DEVICE)


@torch.no_grad()
def predict(net, loader) -> np.ndarray:
    net.eval()
    out = [torch.sigmoid(net(x.to(DEVICE)).squeeze(1)).cpu().numpy() for x, _ in loader]
    return np.concatenate(out)


# ------------------------------------------------------------------ training
def train_one(train_df, val_df, seed, args) -> tuple[nn.Module, float, int]:
    set_seed(seed)
    tr = DataLoader(LesionDataset(train_df, True, args.input), batch_size=args.batch, shuffle=True,
                    num_workers=2, drop_last=True)
    va = DataLoader(LesionDataset(val_df, False, args.input), batch_size=64, num_workers=2)
    net = build_model()
    pos_weight = torch.tensor([(train_df.label == 0).sum() / max((train_df.label == 1).sum(), 1)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        for x, y in tr:
            opt.zero_grad()
            loss = loss_fn(net(x.to(DEVICE)).squeeze(1), y.to(DEVICE))
            loss.backward()
            opt.step()
        sched.step()
        auc = roc_auc_score(val_df.label, predict(net, va))
        if auc > best_auc:
            best_auc, best_epoch, waited = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    net.load_state_dict(best_state)
    return net, best_auc, best_epoch


def ckpt_path(seed, fold, mode):
    return CKPT_DIR / f"resnet18_{mode}_seed{seed}_fold{fold}.pt"


# ------------------------------------------------------------------ Grad-CAM
def gradcam(net, x: torch.Tensor) -> np.ndarray:
    """Heatmap of where the last conv block's evidence for 'malignant' comes from."""
    acts, grads = {}, {}

    def keep(module, inputs, output):
        acts["a"] = output
        output.register_hook(lambda g: grads.__setitem__("g", g))

    handle = net.layer4.register_forward_hook(keep)
    net.eval()
    net.zero_grad()
    with torch.enable_grad():
        net(x.unsqueeze(0).to(DEVICE)).squeeze().backward()
    handle.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1)).squeeze().detach().cpu().numpy()
    cam = np.array(Image.fromarray(cam).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def save_gradcam_grid(net, table, probs, title, path, input_mode, n=8):
    ds = LesionDataset(table, False, input_mode)
    order = np.argsort(-np.abs(probs - table.label.values))  # most confident errors first...
    picks = list(order[:n // 2]) + list(np.argsort(np.abs(probs - table.label.values))[:n - n // 2])  # ...then best hits
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.6))
    for j, i in enumerate(picks):
        x, y = ds[i]
        base = np.array(ds.raw(i).convert("L").resize((IMG_SIZE, IMG_SIZE)))
        cam = gradcam(net, x)
        axes[0, j].imshow(base, cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(base, cmap="gray"); axes[1, j].imshow(cam, cmap="jet", alpha=0.4); axes[1, j].axis("off")
        truth = "malig" if y.item() == 1 else "benign"
        axes[0, j].set_title(f"case {table.iloc[i].Case_ID}\n{truth}, p={probs[i]:.2f}", fontsize=8)
    fig.suptitle(title + "  (left half: largest errors, right half: most confident correct)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(args) -> None:
    meta = pd.read_csv(METADATA_CSV)
    data = pd.read_csv(SPLITS_CSV).merge(meta.drop(columns="label"), on="Case_ID", validate="one_to_one")
    tv, te = data[data.split == "trainval"], data[data.split == "test"]
    folds = sorted(tv.fold.unique())
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}. Input: {args.input}. Seeds: {args.seeds}")

    name = f"cnn_resnet18_{args.input}"
    oof_all, test_all, log = [], [], []
    for seed in args.seeds:
        oof_parts, test_probs = [], []
        for fold in folds:
            tr, va = tv[tv.fold != fold], tv[tv.fold == fold]
            path = ckpt_path(seed, fold, args.input)
            net = build_model()
            if path.exists() and not args.retrain:
                net.load_state_dict(torch.load(path, map_location=DEVICE))
                print(f"seed {seed} fold {fold}: loaded {path.name}")
            else:
                net, auc, ep = train_one(tr, va, seed, args)
                torch.save(net.state_dict(), path)
                log.append({"seed": seed, "fold": fold, "best_val_auroc": auc, "best_epoch": ep})
                print(f"seed {seed} fold {fold}: val AUROC {auc:.3f} at epoch {ep}")
            vl = DataLoader(LesionDataset(va, False, args.input), batch_size=64)
            p = predict(net, vl)
            oof_parts.append(pd.DataFrame({"Case_ID": va.Case_ID, "label": va.label, "fold": fold,
                                           "seed": seed, "prob": p}))
            if args.gradcam and seed == args.seeds[0] and fold == folds[0]:
                save_gradcam_grid(net, va, p, f"Grad-CAM, validation fold {fold}",
                                  FIGURES_DIR / f"{name}_gradcam_val.png", args.input)
            if args.final:
                test_probs.append(predict(net, DataLoader(LesionDataset(te, False, args.input), batch_size=64)))
                if args.gradcam and seed == args.seeds[0] and fold == folds[0]:
                    save_gradcam_grid(net, te, test_probs[-1], "Grad-CAM, held-out test",
                                      FIGURES_DIR / f"{name}_gradcam_test.png", args.input)
        oof_all.append(pd.concat(oof_parts))
        if args.final:
            test_all.append(pd.DataFrame({"Case_ID": te.Case_ID, "label": te.label, "seed": seed,
                                          "prob": np.mean(test_probs, axis=0)}))
    if log:
        pd.DataFrame(log).to_csv(RESULTS_DIR / f"{name}_training_log.csv", index=False)

    # Seed-to-seed variability (the spec asks for runs across several seeds).
    oof_long = pd.concat(oof_all)
    per_seed = {int(s): roc_auc_score(g.label, g.prob) for s, g in oof_long.groupby("seed")}
    print("\nOOF AUROC per seed:", {k: round(v, 3) for k, v in per_seed.items()},
          f"mean {np.mean(list(per_seed.values())):.3f} ± {np.std(list(per_seed.values())):.3f}")

    # Headline numbers: probabilities averaged over seeds.
    oof = oof_long.groupby(["Case_ID", "label", "fold"], as_index=False).prob.mean()
    test = None
    if args.final:
        test_long = pd.concat(test_all)
        test = test_long.groupby(["Case_ID", "label"], as_index=False).prob.mean()
        per_seed_test = {int(s): roc_auc_score(g.label, g.prob) for s, g in test_long.groupby("seed")}
        save_json(per_seed_test, RESULTS_DIR / f"{name}_test_auroc_per_seed.json")
    report(name, oof, test, extra={"oof_auroc_per_seed": per_seed, "input": args.input})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", choices=["crop", "full"], default="crop")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--retrain", action="store_true", help="ignore saved checkpoints")
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--final", action="store_true", help="score the held-out test set (once!)")
    main(p.parse_args())
