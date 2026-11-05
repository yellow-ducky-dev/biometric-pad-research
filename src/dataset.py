"""
Dataset loader for fingerprint images.

Works with any folder of images (recursively globbed) — point it at
SOCOFing's "Real" subfolder, a LivDet folder, or anything else. It does
not assume a specific dataset's directory layout beyond "images live
somewhere under this path."
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FingerprintDataset(Dataset):
    def __init__(self, root, image_size=64, grayscale=True, limit=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {self.root}\n"
                f"Point --data_dir at a real folder of fingerprint images "
                f"(e.g. the 'Real' folder from SOCOFing after download)."
            )
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found under {self.root}. Check the path and file extensions."
            )

        mode = "L" if grayscale else "RGB"
        mean = [0.5] * (1 if grayscale else 3)
        std = [0.5] * (1 if grayscale else 3)

        self.grayscale = grayscale
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),  # -> [-1, 1], matches Tanh output of generator
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")  # convert first, then transform handles grayscale
        img = self.transform(img)
        return img, 0  # label unused for GAN training; kept for API consistency


class RealSyntheticDataset(Dataset):
    """Labeled dataset for PAD detector training/eval: real=0, synthetic=1.

    Pass explicit lists of file paths (produced by a stratified split in
    train_detector.py / evaluate.py) rather than re-globbing here, so that
    train/val/test splits are reproducible and don't leak across runs.
    """

    def __init__(self, real_paths, synthetic_paths, image_size=64):
        self.items = [(p, 0) for p in real_paths] + [(p, 1) for p in synthetic_paths]
        mean, std = [0.5], [0.5]
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
