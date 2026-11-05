# BiometricPAD: GAN-Generated Synthetic Fingerprints vs. a PAD Detector

A small, honest, fully-reproducible research project evaluating whether a
DCGAN trained from scratch on real fingerprints can produce synthetic
samples that fool a CNN-based presentation-attack-detection (PAD) classifier.

**Every number this project produces comes from code in this repo actually
being run on actual data.** There are no placeholder metrics anywhere —
`results/` only contains what the training and evaluation scripts write
after a real run. If you re-run this on different data or a different seed,
your numbers will differ from any numbers quoted in the paper, and that's
expected: report what you actually measured.

## What this is (and isn't)

This is presentation-attack-detection (PAD) research: training a generator
to produce synthetic biometric samples, and a detector to tell them apart
from real ones, in the same spirit as the LivDet competitions and the
broader PAD literature (ISO/IEC 30107). It is **not** an attempt to defeat
any specific commercial product (a phone's fingerprint sensor, a particular
vendor's scanner, etc.) — the "detector" here is a classifier you train
yourself as a stand-in for that class of system, not a real deployed one.

## Project structure

```
biometric-pad-research/
├── src/
│   ├── dataset.py            # image loading (GAN training + labeled real/synthetic)
│   ├── gan.py                # DCGAN generator + discriminator (Radford et al., 2015)
│   ├── train_gan.py          # trains the generator on real fingerprints
│   ├── generate_synthetic.py # produces a synthetic image corpus from a trained generator
│   ├── detector.py           # small CNN: real vs. synthetic classifier (the "PAD" model)
│   ├── train_detector.py     # trains the detector, saves a reproducible train/val/test split
│   ├── evaluate.py           # runs the detector on its held-out test set; computes
│   │                         # accuracy/precision/recall/F1/AUC/APCER/BPCER + real FID
│   └── utils.py              # seeding, device selection, CSV logging
├── notebooks/
│   └── run_on_colab.py       # Jupytext-style notebook: the full pipeline, cell by cell
├── requirements.txt
└── results/                  # created by running the scripts — empty in a fresh clone
```

## Requirements

- A GPU. A free Google Colab T4 or a Kaggle Notebook GPU is enough. Training
  on CPU will technically run but is impractically slow for anything beyond
  a tiny smoke test.
- The [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing)
  (6,000 real fingerprint images, open license) — or any other fingerprint
  image dataset you have legitimate access to (e.g. LivDet, if you've
  requested access). The code doesn't care which; point `--data_dir` /
  `--real_dir` at a folder of images.

## Running it

The easiest path is `notebooks/run_on_colab.py` (open it in Colab via
Jupytext, or just read it top to bottom — every real command is there,
commented out so you can run cells one at a time and inspect output).

The short version, from the command line, once you have a GPU and the
dataset downloaded:

```bash
pip install -r requirements.txt

# 1. Train the GAN on real fingerprints
python src/train_gan.py --data_dir data/SOCOFing/Real \
    --out_dir results/gan_run1 --epochs 50 --batch_size 128

# 2. Generate a synthetic corpus from the trained generator
python src/generate_synthetic.py \
    --checkpoint results/gan_run1/checkpoints/generator_last.pt \
    --out_dir data/synthetic_fingerprints --n 3000

# 3. Train the PAD detector (real vs. synthetic), with a reproducible split
python src/train_detector.py \
    --real_dir data/SOCOFing/Real \
    --synthetic_dir data/synthetic_fingerprints \
    --out_dir results/detector_run1 --epochs 15

# 4. Evaluate on the held-out test set — real metrics + real FID
python src/evaluate.py \
    --split_file results/detector_run1/split.json \
    --checkpoint results/detector_run1/detector_best.pt \
    --out_dir results/detector_run1/eval
```

Expect step 1 (GAN training) to take roughly 1-3 hours on a free Colab T4,
depending on epoch count. Watch `results/gan_run1/samples/epoch_*.png` as it
trains — the images should go from noise to ridge-like patterns. If they
don't, or if training diverges (loss_D collapsing to near-zero early and
staying there is the classic GAN failure mode, "mode collapse"), that's a
real result worth reporting, not something to hide by re-running until a
nicer number shows up.

## What `evaluate.py` actually measures

- **Accuracy / Precision / Recall / F1 / ROC-AUC** — standard binary
  classification metrics, computed with scikit-learn directly from the
  detector's predictions on the test split.
- **APCER / BPCER** (ISO/IEC 30107-3 terms, equivalent to FAR/FRR in older
  biometrics terminology):
  - APCER = fraction of *synthetic* images the detector accepted as real
    (this is the "evasion rate" — how often the attack succeeds).
  - BPCER = fraction of *real* images the detector incorrectly rejected
    as synthetic.
- **FID (Fréchet Inception Distance)** — computed with the `pytorch-fid`
  package's InceptionV3 feature extractor, the same implementation used
  throughout the GAN literature, not a hand-rolled approximation.

## From results to paper

Once you've run the pipeline and have `results/detector_run1/eval/metrics.json`,
tell me the actual numbers (or just share the `results/` folder) and I'll
write the paper's Methods and Results sections directly from that file —
nothing invented, nothing "to be populated later."

## Publishing: GitHub + Zenodo + ACIG

1. **GitHub**: push this repo (code + small result artifacts — CSVs, JSON,
   sample image grids, split.json). Don't commit the raw dataset or large
   `.pt` checkpoints (see `.gitignore`); attach those to a GitHub Release
   instead if you want them archived.
2. **Zenodo**: create a new upload, link it to the GitHub repo (Zenodo's
   GitHub integration will auto-archive each tagged release and mint a
   version-specific DOI under your reserved concept DOI), fill in:
   - Resource type: Software (for the code) — if you're also depositing the
     paper, that's usually a separate Zenodo record of type "Publication/
     Preprint", cross-linked via "Related works".
   - Authors: your name, with your ORCID iD attached.
   - License: whatever you choose for the code (MIT/Apache-2.0 are typical
     for research code); CC-BY 4.0 is standard for the paper text if ACIG
     allows self-archiving.
3. **ACIG submission**: via the editorial system dashboard, with the GitHub
   repo URL and Zenodo DOI cited in the paper's Data/Code Availability
   section.

## Ethical note

Use only datasets you have the legal right to use, and don't point this
pipeline at live, deployed authentication systems. This is research into
detection, not a tool for defeating anyone's actual security.
