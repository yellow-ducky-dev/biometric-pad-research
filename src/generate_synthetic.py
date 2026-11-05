"""
Use a trained generator checkpoint to produce N synthetic fingerprint images,
saved as individual PNGs. These are the actual images used for PAD detector
training/evaluation and for FID computation later — generated fresh from the
model's weights each time, not cached example images.

Usage:
    python src/generate_synthetic.py --checkpoint results/gan_run1/checkpoints/generator_last.pt \
        --out_dir data/synthetic_fingerprints --n 2000
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from gan import Generator, NZ
from utils import set_seed, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)  # different seed than training
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netG = Generator().to(device)
    netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
    netG.eval()

    produced = 0
    with torch.no_grad():
        while produced < args.n:
            bs = min(args.batch_size, args.n - produced)
            noise = torch.randn(bs, NZ, 1, 1, device=device)
            fakes = netG(noise).cpu()
            for j in range(bs):
                save_image(fakes[j], out_dir / f"synth_{produced + j:05d}.png", normalize=True)
            produced += bs
            print(f"Generated {produced}/{args.n}")

    print(f"Saved {produced} synthetic images to {out_dir}")


if __name__ == "__main__":
    main()
