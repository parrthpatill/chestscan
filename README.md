# ChestScan

AI-Based Pneumonia Screening from Chest X-Ray Images.

## Project Status

Stage 0 — Setup and Foundations

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

## Medical Safety

ChestScan is a screening aid and is not a diagnostic tool.
It is not intended to replace evaluation by a qualified medical professional.
