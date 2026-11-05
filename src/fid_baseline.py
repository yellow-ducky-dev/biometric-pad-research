"""
One-off sanity check: FID between two random halves of the REAL test images.
"""
import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from pytorch_fid.fid_score import calculate_fid_given_paths

from utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    with open(args.split_file) as f:
        split = json.load(f)
    real_test = split["real_test"]
    print(f"Splitting {len(real_test)} real test images into two random halves...")

    shuffled = real_test[:]
    random.Random(args.seed).shuffle(shuffled)
    half = len(shuffled) // 2
    half_a, half_b = shuffled[:half], shuffled[half:]
    print(f"Half A: {len(half_a)} images | Half B: {len(half_b)} images")

    FID_SIZE = (299, 299)
    tmp_a = tempfile.mkdtemp(prefix="fid_real_a_")
    tmp_b = tempfile.mkdtemp(prefix="fid_real_b_")
    try:
        for i, p in enumerate(half_a):
            PILImage.open(p).convert("RGB").resize(FID_SIZE).save(Path(tmp_a) / f"{i:06d}.png")
        for i, p in enumerate(half_b):
            PILImage.open(p).convert("RGB").resize(FID_SIZE).save(Path(tmp_b) / f"{i:06d}.png")

        fid_value = calculate_fid_given_paths([tmp_a, tmp_b], batch_size=50, device=device, dims=2048)
        print(f"\nBaseline FID (real vs. real, two random halves): {fid_value:.4f}")
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


if __name__ == "__main__":
    main()
