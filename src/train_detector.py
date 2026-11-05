"""
Train the PAD (real vs. synthetic) detector.

Splits real and synthetic images into train/val/test (stratified, seeded)
BEFORE any training happens, and saves the exact split (as lists of file
paths) to disk — so evaluate.py tests on images the detector genuinely
never saw, and the split is auditable later.

Usage:
    python src/train_detector.py \
        --real_dir /path/to/socofing/Real \
        --synthetic_dir data/synthetic_fingerprints \
        --out_dir results/detector_run1 --epochs 15
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import RealSyntheticDataset, list_images
from detector import PADClassifier
from utils import set_seed, get_device, CSVLogger


def stratified_split(paths, train_frac=0.7, val_frac=0.15, seed=42):
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n = len(paths)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:]


def evaluate_split(model, loader, device, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = (out > 0.5).float()
            correct += (preds == labels).sum().item()
            n += imgs.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--synthetic_dir", required=True)
    ap.add_argument("--out_dir", default="results/detector_run")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_paths = list_images(args.real_dir)
    synth_paths = list_images(args.synthetic_dir)
    if len(real_paths) == 0 or len(synth_paths) == 0:
        raise RuntimeError(
            f"Found {len(real_paths)} real and {len(synth_paths)} synthetic images. "
            f"Both must be non-zero. Check --real_dir and --synthetic_dir."
        )
    print(f"Real images: {len(real_paths)} | Synthetic images: {len(synth_paths)}")

    real_tr, real_val, real_te = stratified_split(real_paths, seed=args.seed)
    synth_tr, synth_val, synth_te = stratified_split(synth_paths, seed=args.seed)

    # Save the exact split so evaluate.py (and a reader of the paper) can verify
    # the test set was never seen during training.
    split_record = {
        "real_train": [str(p) for p in real_tr], "real_val": [str(p) for p in real_val], "real_test": [str(p) for p in real_te],
        "synthetic_train": [str(p) for p in synth_tr], "synthetic_val": [str(p) for p in synth_val], "synthetic_test": [str(p) for p in synth_te],
    }
    with open(out_dir / "split.json", "w") as f:
        json.dump(split_record, f, indent=2)

    train_ds = RealSyntheticDataset(real_tr, synth_tr, image_size=args.image_size)
    val_ds = RealSyntheticDataset(real_val, synth_val, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = PADClassifier().to(device)
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    logger = CSVLogger(out_dir / "detector_training_log.csv",
                        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (out > 0.5).float()
            correct += (preds == labels).sum().item()
            n += imgs.size(0)

        train_loss, train_acc = total_loss / n, correct / n
        val_loss, val_acc = evaluate_split(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        logger.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "detector_best.pt")

    torch.save(model.state_dict(), out_dir / "detector_last.pt")
    print(f"Done. Best val_acc={best_val_acc:.4f}. Checkpoints and split.json saved under {out_dir}")
    print("Run evaluate.py next, pointing at this out_dir's split.json, to get honest test-set metrics.")


if __name__ == "__main__":
    main()
