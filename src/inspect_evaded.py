"""
Find and save the specific synthetic test images that fooled the detector
(classified as "real" when they were actually synthetic), so we can look
at them directly rather than just trusting an aggregate percentage.

Usage:
    python src/inspect_evaded.py \
        --split_file ../results/detector_fastgan_150ep_run1/split.json \
        --checkpoint ../results/detector_fastgan_150ep_run1/detector_best.pt \
        --out_dir ../results/detector_fastgan_150ep_run1/evaded_samples
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from dataset import RealSyntheticDataset
from detector import PADClassifier
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_file) as f:
        split = json.load(f)
    synth_test = split["synthetic_test"]

    model = PADClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = RealSyntheticDataset([], synth_test, image_size=args.image_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    evaded = []
    idx = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs_dev = imgs.to(device)
            scores = model(imgs_dev).cpu()
            preds = (scores > 0.5).float()
            for j in range(len(labels)):
                if preds[j].item() == 0.0:  # predicted "real" but is actually synthetic
                    path = synth_test[idx + j]
                    score = scores[j].item()
                    evaded.append((path, score))
            idx += len(labels)

    print(f"Found {len(evaded)} synthetic images classified as real (evaded detection):")
    for path, score in evaded:
        print(f"  {path}  (detector score: {score:.4f}, lower = more 'real'-looking to it)")
        dest = out_dir / Path(path).name
        shutil.copy(path, dest)
    print(f"\nCopied evaded images to {out_dir} for visual inspection.")


if __name__ == "__main__":
    main()
