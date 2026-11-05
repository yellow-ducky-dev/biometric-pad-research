"""
Replicates evaluate.py's exact dataset/batching (real + synthetic together)
so results match exactly, and prints the raw detector score for every
synthetic test image, sorted so the borderline ones (closest to 0.5) are
easy to spot.
"""
import argparse
import json
from pathlib import Path

import torch

from dataset import RealSyntheticDataset
from detector import PADClassifier
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image_size", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    with open(args.split_file) as f:
        split = json.load(f)
    real_test = split["real_test"]
    synth_test = split["synthetic_test"]

    ds = RealSyntheticDataset(real_test, synth_test, image_size=args.image_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    model = PADClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            scores = model(imgs.to(device)).cpu()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())

    synth_items = [(ds.items[i][0], all_scores[i]) for i in range(len(ds)) if all_labels[i] == 1]
    synth_items.sort(key=lambda x: x[1])  # lowest score = most "real"-looking to the detector

    print("10 synthetic test images with the LOWEST scores (closest to being misclassified as real):")
    for path, score in synth_items[:10]:
        flag = " <-- BELOW 0.5 THRESHOLD (evaded)" if score <= 0.5 else ""
        print(f"  score={score:.6f}  {path}{flag}")


if __name__ == "__main__":
    main()
