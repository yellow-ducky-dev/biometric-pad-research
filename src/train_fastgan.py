"""
Train FastGAN on real fingerprint images at a higher resolution (128x128
by default) than the DCGAN baseline (64x64).

Uses hinge loss (standard for FastGAN/StyleGAN-family training) plus a
reconstruction loss on real images only, which is the discriminator's
self-supervised stabilizing signal (see fastgan.py's module docstring).

Every logged value is read directly from the actual loss tensors computed
during training -- same honesty guarantee as train_gan.py. If this
architecture also fails to close the realism gap, or fails to train
stably, that is itself a reportable result, not something to hide.

Usage:
    python src/train_fastgan.py --data_dir /path/to/socofing/Real \
        --out_dir results/fastgan_run1 --epochs 50 --batch_size 32 --image_size 128
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import FingerprintDataset
from fastgan import FastGenerator, FastDiscriminator, weights_init, NZ
from utils import set_seed, get_device, CSVLogger


def hinge_loss_d(real_logit, fake_logit):
    return (F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean())


def hinge_loss_g(fake_logit):
    return -fake_logit.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="results/fastgan_run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=128, choices=[128, 256])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--recon_weight", type=float, default=1.0)
    ap.add_argument("--width_mult", type=float, default=1.0,
                     help="Channel-width multiplier for both G and D. "
                          "2.0 = roughly 4x the parameters of the default "
                          "('FastGAN-Large'), a direct test of a higher-capacity attacker.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_every", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. FastGAN at this resolution will be "
              "extremely slow on CPU -- this is intended to run on a GPU session.")

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = FingerprintDataset(args.data_dir, image_size=args.image_size, limit=args.limit)
    print(f"Loaded {len(dataset)} real fingerprint images at {args.image_size}x{args.image_size}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, drop_last=True)

    netG = FastGenerator(nz=NZ, image_size=args.image_size, width_mult=args.width_mult).to(device)
    netD = FastDiscriminator(image_size=args.image_size, width_mult=args.width_mult).to(device)
    n_params_g = sum(p.numel() for p in netG.parameters())
    n_params_d = sum(p.numel() for p in netD.parameters())
    print(f"Generator parameters: {n_params_g:,} | Discriminator parameters: {n_params_d:,} "
          f"(width_mult={args.width_mult})")
    netG.apply(weights_init)
    netD.apply(weights_init)

    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(16, NZ, 1, 1, device=device)
    logger = CSVLogger(out_dir / "training_log.csv",
                        fieldnames=["epoch", "batch", "loss_D", "loss_G",
                                    "recon_loss", "D_real_logit", "D_fake_logit", "elapsed_sec"])

    t0 = time.time()
    for epoch in range(args.epochs):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b_size = real.size(0)

            # --- Discriminator step: hinge loss + reconstruction on real images ---
            netD.zero_grad()
            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake = netG(noise)

            real_logit, recon_loss = netD(real, return_recon=True)
            fake_logit = netD(fake.detach())

            lossD = hinge_loss_d(real_logit, fake_logit) + args.recon_weight * recon_loss
            lossD.backward()
            optD.step()

            # --- Generator step ---
            netG.zero_grad()
            fake_logit_for_g = netD(fake)
            lossG = hinge_loss_g(fake_logit_for_g)
            lossG.backward()
            optG.step()

            if i % 10 == 0:
                elapsed = time.time() - t0
                print(f"[{epoch}/{args.epochs}][{i}/{len(loader)}] "
                      f"loss_D={lossD.item():.4f} loss_G={lossG.item():.4f} "
                      f"recon={recon_loss.item():.4f} "
                      f"D(real)={real_logit.mean().item():.3f} D(fake)={fake_logit.mean().item():.3f} "
                      f"elapsed={elapsed:.0f}s")
                logger.log({
                    "epoch": epoch, "batch": i,
                    "loss_D": lossD.item(), "loss_G": lossG.item(),
                    "recon_loss": recon_loss.item(),
                    "D_real_logit": real_logit.mean().item(),
                    "D_fake_logit": fake_logit.mean().item(),
                    "elapsed_sec": round(elapsed, 1),
                })

        if epoch % args.sample_every == 0 or epoch == args.epochs - 1:
            netG.eval()
            with torch.no_grad():
                sample = netG(fixed_noise).detach().cpu()
            netG.train()
            save_image(sample, out_dir / "samples" / f"epoch_{epoch:04d}.png",
                       normalize=True, nrow=4)

        torch.save(netG.state_dict(), out_dir / "checkpoints" / "generator_last.pt")
        torch.save(netD.state_dict(), out_dir / "checkpoints" / "discriminator_last.pt")

    print(f"Done. Checkpoints and logs saved under {out_dir}")


if __name__ == "__main__":
    main()
