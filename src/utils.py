"""
Shared utilities: seeding, device selection, CSV logging.
Nothing in this file produces or fabricates numbers — it only helps
record, honestly, whatever the training/eval scripts actually measure.
"""
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CSVLogger:
    """Appends rows to a CSV file, writing the header once.
    Used for every metric this project reports, so that every number
    in the eventual paper can be traced back to a row in one of these files.
    """

    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def count_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)
