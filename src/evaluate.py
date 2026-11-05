"""
Honest evaluation of the trained PAD detector on its held-out test set,
plus a real FID score between real and synthetic test images.

Every number this script prints and saves comes from running the actual
model on actual image files listed in split.json (written by
train_detector.py) — nothing here is a placeholder or literature value.

Metric definitions (ISO/IEC 30107-3 terms, used throughout the PAD literature):
  APCER (Attack Presentation Classification Error Rate):
      fraction of SYNTHETIC images the detector accepted as real.
      (equivalent to "FAR" in older biometrics terminology)
  BPCER (Bona fide Presentation Classification Error Rate):
      fraction of REAL images the detector rejected as synthetic.
      (equivalent to "FRR")

Usage:
    python src/evaluate.py --split_file results/detector_run1/split.json \
        --checkpoint results/detector_run1/detector_best.pt \
        --out_dir results/detector_run1/eval
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RealSyntheticDataset
from detector import PADClassifier
from utils import get_device


def run_inference(model, dataset, device, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            scores = model(imgs).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_scores), np.array(all_labels)


def compute_fid(real_paths, synth_paths, device, out_dir):
    """Real FID, computed via pytorch-fid's InceptionV3 feature extractor.
    Writes images to temp dirs because pytorch_fid's API operates on paths.

    Two real-data wrinkles handled here:
    1. pytorch-fid globs for lowercase extensions only (*.png, *.bmp, ...).
       On a case-sensitive filesystem that misses files like SOCOFing's
       *.BMP, so we re-save every image as a lowercase .png.
    2. pytorch-fid's own DataLoader stacks a batch into one tensor WITHOUT
       resizing first, so a batch mixing differently-sized source images
       (which real SOCOFing files are -- they're not all the same pixel
       dimensions) crashes on torch.stack. We resize every image to 299x299
       (InceptionV3's native input size, so this isn't a lossy extra step --
       the model would resize to this internally anyway) before saving.
    Neither step changes what FID is measuring -- same pixel content, just
    normalized file format and matched dimensions.
    """
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from PIL import Image as PILImage
    import shutil, tempfile

    if len(real_paths) < 2 or len(synth_paths) < 2:
        print("Not enough images in one of the sets to compute FID (need >= 2). Skipping.")
        return None

    FID_SIZE = (299, 299)
    tmp_real = tempfile.mkdtemp(prefix="fid_real_")
    tmp_synth = tempfile.mkdtemp(prefix="fid_synth_")
    try:
        for i, p in enumerate(real_paths):
            PILImage.open(p).convert("RGB").resize(FID_SIZE).save(Path(tmp_real) / f"{i:06d}.png")
        for i, p in enumerate(synth_paths):
            PILImage.open(p).convert("RGB").resize(FID_SIZE).save(Path(tmp_synth) / f"{i:06d}.png")
        fid_value = calculate_fid_given_paths(
            [tmp_real, tmp_synth], batch_size=50,
            device=device, dims=2048
        )
        return float(fid_value)
    finally:
        shutil.rmtree(tmp_real, ignore_errors=True)
        shutil.rmtree(tmp_synth, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--skip_fid", action="store_true", help="skip FID (it's slow on CPU)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    with open(args.split_file) as f:
        split = json.load(f)
    real_test = split["real_test"]
    synth_test = split["synthetic_test"]
    print(f"Test set: {len(real_test)} real, {len(synth_test)} synthetic images")

    test_ds = RealSyntheticDataset(real_test, synth_test, image_size=args.image_size)

    model = PADClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    scores, labels = run_inference(model, test_ds, device)
    preds = (scores > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])  # rows/cols: 0=real, 1=synthetic
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")  # happens only if a class is entirely absent from the test set

    tn, fp, fn, tp = cm.ravel()
    # tn: real correctly called real | fp: real called synthetic (BPCER numerator)
    # fn: synthetic called real (APCER numerator) | tp: synthetic correctly called synthetic
    apcer = fn / (fn + tp) if (fn + tp) > 0 else float("nan")  # == FAR
    bpcer = fp / (fp + tn) if (fp + tn) > 0 else float("nan")  # == FRR

    results = {
        "n_real_test": len(real_test),
        "n_synthetic_test": len(synth_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "APCER_FAR": apcer,
        "BPCER_FRR": bpcer,
        "confusion_matrix": {"TN_real_as_real": int(tn), "FP_real_as_synthetic": int(fp),
                              "FN_synthetic_as_real": int(fn), "TP_synthetic_as_synthetic": int(tp)},
    }

    if not args.skip_fid:
        print("Computing FID (this calls a real InceptionV3 forward pass on every image)...")
        fid = compute_fid(real_test, synth_test, device, out_dir)
        results["fid"] = fid

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "synthetic"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "synthetic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.title("PAD detector: confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close()

    print(f"\nSaved metrics.json, confusion_matrix.png, roc_curve.png to {out_dir}")


if __name__ == "__main__":
    main()
