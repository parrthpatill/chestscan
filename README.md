# ChestScan

AI-Based Pneumonia Screening from Chest X-Ray Images.

## Project Status

Stage 1 — Data Understanding and Leakage Audit completed. Model training has not started.

## Verified Environment

- Local project Python: 3.11.16 through pyenv
- System Python: 3.12.3
- GPU notebook: Google Colab
- TensorFlow: 2.20.0
- GPU: Verified and used successfully by TensorFlow

## Dataset

Kermany et al. Chest X-Ray Images (Pneumonia), accessed through Kaggle.

Verified extracted dataset structure:

- Train: 5,216 images
  - NORMAL: 1,341
  - PNEUMONIA: 3,875
- Validation: 16 images
  - NORMAL: 8
  - PNEUMONIA: 8
- Test: 624 images
  - NORMAL: 234
  - PNEUMONIA: 390

The original validation set is only 16 images and will not be used for model tuning. A new patient-level train/validation split will be created during Stage 1.

## Dataset Count and Leakage Audit

- Verified extracted image count: 5,856.
- The project brief states 5,863 images; the discrepancy remains unexplained.
- Filename-derived identifiers show 170 train-test overlaps involving 320 test images.
- These identifiers are proxies and have not been independently verified as patient identities.
- The original 624-image test set remains unchanged. No overlapping images have been silently removed.
- The original 16-image validation set will not be used for model tuning.

The combined EDA and leakage-audit notebook is currently stored at
`notebooks/ChestScan_Day3_Day4_Clean_.ipynb`. Its organization will be reconciled
with the planned notebook layout during a later cleanup.

## Medical Safety

ChestScan is a screening aid and is not a diagnostic tool.
It is not intended to replace evaluation by a qualified medical professional.
