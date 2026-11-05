"""
Train the DCGAN on real fingerprint images.

Every loss value logged here is read directly from the loss tensors
produced during training — nothing is estimated or pre-filled. If
training diverges or mode-collapses, that will show up in the CSV and
in the sample grids; report it as-is rather than re-running until a
nicer-looking number appears.

Usage:
    python src/train_gan.py --data_dir /path/to/socofing/Real \
        --out_dir results/gan_run1 --epochs 50 --batch_size 128
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import FingerprintDataset
from gan import Generator, Discriminator, weights_init, NZ
from utils import set_seed, get_device, CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder of real fingerprint images")
    ap.add_argument("--out_dir", default="results/gan_run")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_every", type=int, default=1, help="epochs between saved sample grids")
    ap.add_argument("--limit", type=int, default=None, help="optional cap on number of training images")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = FingerprintDataset(args.data_dir, image_size=args.image_size, limit=args.limit)
    print(f"Loaded {len(dataset)} real fingerprint images from {args.data_dir}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, drop_last=True)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)
    logger = CSVLogger(out_dir / "training_log.csv",
                        fieldnames=["epoch", "batch", "loss_D", "loss_G",
                                    "D_x", "D_G_z1", "D_G_z2", "elapsed_sec"])

    real_label, fake_label = 1.0, 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b_size = real.size(0)

            # --- train Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, NZ, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optD.step()

            # --- train Generator: maximize log(D(G(z))) ---
            netG.zero_grad()
            label.fill_(real_label)  # generator wants discriminator to say "real"
            output = netD(fake)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optG.step()

            if i % 20 == 0:
                elapsed = time.time() - t0
                print(f"[{epoch}/{args.epochs}][{i}/{len(loader)}] "
                      f"loss_D={lossD.item():.4f} loss_G={lossG.item():.4f} "
                      f"D(x)={D_x:.3f} D(G(z))={D_G_z1:.3f}/{D_G_z2:.3f} "
                      f"elapsed={elapsed:.0f}s")
                logger.log({
                    "epoch": epoch, "batch": i,
                    "loss_D": lossD.item(), "loss_G": lossG.item(),
                    "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2,
                    "elapsed_sec": round(elapsed, 1),
                })

        if epoch % args.sample_every == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                fake_grid = netG(fixed_noise).detach().cpu()
            save_image(fake_grid, out_dir / "samples" / f"epoch_{epoch:04d}.png",
                       normalize=True, nrow=8)

        torch.save(netG.state_dict(), out_dir / "checkpoints" / "generator_last.pt")
        torch.save(netD.state_dict(), out_dir / "checkpoints" / "discriminator_last.pt")

    print(f"Done. Checkpoints and logs saved under {out_dir}")


if __name__ == "__main__":
    main()
