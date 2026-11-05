# %% [markdown]
# # Biometric PAD Research — Colab Runner
#
# This notebook is written as a plain Python file with `# %%` cell markers
# (Jupytext-style). Colab, VS Code, and Jupyter can all open it directly as
# a notebook ("Open with" -> Jupytext, or `jupytext --to notebook run_on_colab.py`
# if you want a literal .ipynb file).
#
# **Before running:** Runtime -> Change runtime type -> GPU (T4 is fine).
#
# Every cell below either runs real code against real data, or clearly
# comments that a step is a one-time setup step (installs, dataset download).
# No cell contains a pre-filled "expected output."

# %% [markdown]
# ## 1. Setup

# %%
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# %cd YOUR_REPO
# (Until you've pushed to GitHub, instead upload the project zip to Colab's
#  file browser and run: !unzip biometric-pad-research.zip && %cd biometric-pad-research)

# %%
# !pip install -q -r requirements.txt

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (training will be slow)")

# %% [markdown]
# ## 2. Download the dataset (SOCOFing, from Kaggle)
#
# Kaggle's current API uses a short-lived token string (no kaggle.json
# download). Get yours from kaggle.com -> your profile picture -> Settings
# -> API -> "Generate New Token". It's shown once as a plain string — copy it.

# %%
# !pip install -q -U kagglehub

# %%
import kagglehub
# This prompts you to paste the token string you copied above.
kagglehub.login()

# %%
# Downloads into kagglehub's cache (not into data/) and returns that path.
socofing_path = kagglehub.dataset_download("ruizgara/socofing")
print("Downloaded to:", socofing_path)

# %%
# Confirm the actual folder layout before moving on — it can vary.
# !find {socofing_path} -maxdepth 2 -type d

REAL_DATA_DIR = f"{socofing_path}/SOCOFing/Real"  # adjust if the find above shows a different path

# %% [markdown]
# ## 3. Train the DCGAN on real fingerprints
#
# ~50 epochs on SOCOFing's ~6,000 real images takes roughly 1-2 hours on a
# free Colab T4. Watch `results/gan_run1/samples/epoch_*.png` as it trains —
# synthetic fingerprints should start looking ridge-like by epoch 15-20.
# If loss_D collapses to ~0 early and stays there, that's mode collapse —
# a real, reportable outcome, not a bug to silently work around.

# %%
# REAL_DATA_DIR is an absolute path (from kagglehub's cache), so don't
# prepend "../" to it — only the output dir is relative to src/.
# !cd src && python train_gan.py \
#     --data_dir {REAL_DATA_DIR} \
#     --out_dir ../results/gan_run1 \
#     --epochs 50 \
#     --batch_size 128

# %% [markdown]
# ## 4. Generate a synthetic test corpus from the trained generator

# %%
# !cd src && python generate_synthetic.py \
#     --checkpoint ../results/gan_run1/checkpoints/generator_last.pt \
#     --out_dir ../data/synthetic_fingerprints \
#     --n 3000

# %% [markdown]
# ## 5. Train the PAD detector (real vs. synthetic classifier)

# %%
# !cd src && python train_detector.py \
#     --real_dir {REAL_DATA_DIR} \
#     --synthetic_dir ../data/synthetic_fingerprints \
#     --out_dir ../results/detector_run1 \
#     --epochs 15

# %% [markdown]
# ## 6. Evaluate on the held-out test set (real metrics, real FID)

# %%
# !cd src && python evaluate.py \
#     --split_file ../results/detector_run1/split.json \
#     --checkpoint ../results/detector_run1/detector_best.pt \
#     --out_dir ../results/detector_run1/eval

# %%
import json
with open("results/detector_run1/eval/metrics.json") as f:
    metrics = json.load(f)
metrics

# %% [markdown]
# ## 7. Look at the actual images
#
# Always eyeball these before writing anything in the paper — a classifier
# metric alone won't tell you if the GAN is producing recognizable
# fingerprint structure or noise that happens to fool a small CNN.

# %%
import matplotlib.pyplot as plt
from PIL import Image
import glob

samples = sorted(glob.glob("results/gan_run1/samples/*.png"))[-1]
img = Image.open(samples)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.title(f"Generator output: {samples}")
plt.show()

# %% [markdown]
# ## 8. Download everything back to your machine
#
# `results/` now contains: training_log.csv, detector_training_log.csv,
# split.json, metrics.json, confusion_matrix.png, roc_curve.png, and sample
# image grids across training. This is the evidence base for the paper and
# what gets pushed to GitHub + archived on Zenodo.

# %%
# !zip -r results.zip results/
# from google.colab import files
# files.download("results.zip")
