"""
Generate synthetic fingerprint images from a trained FastGAN checkpoint.

Same purpose and honesty guarantee as generate_synthetic.py (for the DCGAN
baseline): fresh latent noise each time, nothing cached or cherry-picked.

Usage:
    python src/generate_synthetic_fastgan.py \
        --checkpoint results/fastgan_run1/checkpoints/generator_last.pt \
        --out_dir data/synthetic_fingerprints_fastgan --n 2000 --image_size 128
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from fastgan import FastGenerator, NZ
from utils import set_seed, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=128, choices=[128, 256])
    ap.add_argument("--width_mult", type=float, default=1.0,
                     help="Must match the width_mult used in train_fastgan.py "
                          "for this checkpoint, or loading will fail with a shape mismatch.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    netG = FastGenerator(nz=NZ, image_size=args.image_size, width_mult=args.width_mult).to(device)
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
