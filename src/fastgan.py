"""
A simplified FastGAN / "Lightweight GAN" implementation, following:
  Liu et al., "Towards Faster and Stabilized GAN Training for High-Fidelity
  Few-Shot Image Synthesis," ICLR 2021.

Why this architecture (vs. training StyleGAN2 from scratch): FastGAN was
specifically designed to produce good image quality from a single GPU in
hours, with far fewer training images/steps than StyleGAN2 needs. It is a
real, published, citable architecture -- not a made-up "better" model.

Two simplifications vs. the original paper, both noted here rather than
hidden:
  1. The discriminator's auxiliary reconstruction decoder is implemented in
     a simplified single-branch form (resizes the real image down for the
     L1 target), not the paper's multi-crop dual-decoder setup.
  2. No spectral normalization (BatchNorm is used instead, as in the DCGAN
     baseline) -- a reasonable simplification for a dataset this size.

The core ideas that matter for training speed/stability are kept: GLU
activations, skip-layer excitation (SLE) connecting low- and high-resolution
feature maps in the generator, and the discriminator's self-supervised
reconstruction loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NZ = 256  # latent dimension (FastGAN papers typically use 256, vs. 100 for DCGAN)
NC = 1    # grayscale

# Base channel counts per spatial resolution. Deliberately modest
# ("lightweight") compared to the original paper's defaults, appropriate for
# a dataset of this size (6,000 images) and single-GPU/few-hour training
# budget. make_channels() scales this by width_mult for a "Large" variant --
# a direct, honest way to test "what if the attacker had a more powerful
# generator" without changing architecture family or resolution.
_BASE_CHANNELS = {4: 256, 8: 128, 16: 64, 32: 64, 64: 32, 128: 16, 256: 8}


def make_channels(width_mult=1.0):
    return {res: max(8, int(round(ch * width_mult))) for res, ch in _BASE_CHANNELS.items()}


CHANNELS = make_channels(1.0)  # kept for backward compatibility / default


def init_block(nz, out_ch):
    """Latent vector -> 4x4 feature map."""
    return nn.Sequential(
        nn.ConvTranspose2d(nz, out_ch * 2, 4, 1, 0, bias=False),
        nn.BatchNorm2d(out_ch * 2),
        nn.GLU(dim=1),  # halves channels back to out_ch
    )


def up_block(in_ch, out_ch):
    """Doubles spatial resolution; nearest-upsample + conv + BN + GLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch * 2),
        nn.GLU(dim=1),
    )


class SLEBlock(nn.Module):
    """Skip-Layer Excitation: a low-resolution feature map gates (channel-wise)
    a much higher-resolution one, letting early, coarse structure influence
    fine detail directly -- part of why FastGAN trains stably in few steps."""

    def __init__(self, ch_low, ch_high):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_low, ch_high, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch_high, ch_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_low, feat_high):
        return feat_high * self.main(feat_low)


class FastGenerator(nn.Module):
    """Generates images up to 128x128. (256x256 supported structurally but
    not recommended for this dataset -- see module docstring.)"""

    def __init__(self, nz=NZ, nc=NC, image_size=128, width_mult=1.0):
        super().__init__()
        assert image_size in (128, 256), (
            "image_size must be 128 (recommended for this dataset) or 256. "
            "For 64x64, use the plain DCGAN in gan.py instead."
        )
        self.image_size = image_size
        self.width_mult = width_mult
        ch = make_channels(width_mult)
        self.channels = ch

        self.init = init_block(nz, ch[4])
        self.up_8 = up_block(ch[4], ch[8])
        self.up_16 = up_block(ch[8], ch[16])
        self.up_32 = up_block(ch[16], ch[32])
        self.sle_32 = SLEBlock(ch[4], ch[32])

        self.up_64 = up_block(ch[32], ch[64])
        self.sle_64 = SLEBlock(ch[8], ch[64])

        self.up_128 = up_block(ch[64], ch[128])
        self.sle_128 = SLEBlock(ch[16], ch[128])

        final_ch = ch[128]
        if image_size == 256:
            self.up_256 = up_block(ch[128], ch[256])
            self.sle_256 = SLEBlock(ch[32], ch[256])
            final_ch = ch[256]

        self.to_image = nn.Sequential(
            nn.Conv2d(final_ch, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        feat_4 = self.init(z)
        feat_8 = self.up_8(feat_4)
        feat_16 = self.up_16(feat_8)

        feat_32 = self.up_32(feat_16)
        feat_32 = self.sle_32(feat_4, feat_32)

        feat_64 = self.up_64(feat_32)
        feat_64 = self.sle_64(feat_8, feat_64)

        feat_128 = self.up_128(feat_64)
        feat_128 = self.sle_128(feat_16, feat_128)

        if self.image_size == 256:
            feat_256 = self.up_256(feat_128)
            feat_256 = self.sle_256(feat_32, feat_256)
            return self.to_image(feat_256)

        return self.to_image(feat_128)


def down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class FastDiscriminator(nn.Module):
    """Discriminator with a simplified auxiliary reconstruction decoder
    (see module docstring) that reconstructs a downsampled version of the
    input from an intermediate 16x16 feature map. This reconstruction loss
    (added only for REAL images during training) encourages the
    discriminator's features to stay informative rather than collapsing to
    a trivial real/fake shortcut -- the main stability trick borrowed from
    the FastGAN paper, in simplified form.
    """

    def __init__(self, nc=NC, image_size=128, width_mult=1.0):
        super().__init__()
        self.image_size = image_size
        self.width_mult = width_mult
        ch = make_channels(width_mult)
        self.channels = ch

        # Build the downsampling path generically based on image_size.
        res_order = [128, 64, 32, 16, 8, 4]
        if image_size == 256:
            res_order = [256, 128, 64, 32, 16, 8, 4]
        start_idx = res_order.index(image_size)
        res_order = res_order[start_idx:]

        self.down_blocks = nn.ModuleList()
        in_ch = nc
        feature_res_at_16 = None
        for i in range(len(res_order) - 1):
            out_res = res_order[i + 1]
            out_ch = ch[out_res]
            self.down_blocks.append(down_block(in_ch, out_ch))
            in_ch = out_ch
            if out_res == 16:
                feature_res_at_16 = out_ch

        self.final = nn.Sequential(
            nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False),
        )

        # Reconstruction decoder off the 16x16 feature map -> 64x64 image,
        # compared (L1) against the real input resized to 64x64.
        decoder_in_ch = feature_res_at_16 if feature_res_at_16 is not None else ch[16]
        self.decoder = nn.Sequential(
            up_block(decoder_in_ch, ch[32]),
            up_block(ch[32], ch[64]),
            nn.Conv2d(ch[64], nc, 3, 1, 1),
            nn.Tanh(),
        )
        self._feat16_ch = feature_res_at_16

    def forward(self, x, return_recon=False):
        feat = x
        feat_16 = None
        cur_res = self.image_size
        for block in self.down_blocks:
            feat = block(feat)
            cur_res = cur_res // 2
            if cur_res == 16:
                feat_16 = feat

        logit = self.final(feat).view(-1)

        if return_recon:
            if feat_16 is None:
                raise RuntimeError(
                    "Discriminator never produced a 16x16 feature map for this "
                    "image_size; reconstruction loss unavailable."
                )
            recon = self.decoder(feat_16)
            target = F.interpolate(x, size=recon.shape[-2:], mode="bilinear", align_corners=False)
            recon_loss = F.l1_loss(recon, target)
            return logit, recon_loss

        return logit


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
