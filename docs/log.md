# ChestScan Development Log

## Day 1 — Stage 0: Environment and Project Setup

### Completed
- Verified Ubuntu 24.04 environment.
- Verified Git 2.43.0.
- Verified Node.js v20.19.5 and npm 10.8.2.
- Installed pyenv 2.8.6.
- Installed Python 3.11.16 through pyenv.
- Configured Python 3.11.16 specifically for the `chestscan` project.
- Confirmed Ubuntu system Python remains Python 3.12.3.
- Initialized Git repository on the `main` branch.
- Created initial project directory structure.
- Created `README.md` with project status and medical-safety framing.

### Current Python Setup
- Project Python: 3.11.16
- Project Python selector: `.python-version`
- System Python: 3.12.3

### Day 1 Checklist
- [x] Python 3.10/3.11 environment available
- [x] Git configured
- [x] Git repository initialized
- [x] Main branch created
- [x] Basic project structure created
- [x] `README.md` created
- [x] Medical-safety framing included
- [ ] GitHub remote repository connected
- [ ] Stage 0 learning completed
- [ ] Stage 0 final gate completed

### Notes
Python 3.11.16 is isolated through pyenv and does not replace Ubuntu's system Python 3.12.3.

## Day 2 — Stage 0: GPU Notebook and Dataset Access

### Completed
- Created and verified a Google Colab GPU notebook.
- Verified TensorFlow 2.20.0.
- Verified that TensorFlow detects a GPU.
- Verified that a TensorFlow matrix multiplication executes on `/device:GPU:0`.
- Configured Kaggle API authentication using `KAGGLE_API_TOKEN`.
- Installed the Kaggle client.
- Identified the project dataset:
  `paultimothymooney/chest-xray-pneumonia`
- Downloaded the dataset successfully from Kaggle.
- Extracted the dataset in Colab.
- Verified the `train`, `val`, and `test` directory structure.
- Verified `NORMAL` and `PNEUMONIA` class directories.
- Verified the actual image counts.

### Verified Dataset Counts

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

### GPU Verification

TensorFlow successfully executed a matrix multiplication on:

`/job:localhost/replica:0/task:0/device:GPU:0`

### Day 2 Checklist
- [x] GPU runtime enabled
- [x] TensorFlow imported successfully
- [x] GPU detected by TensorFlow
- [x] TensorFlow operation executed on GPU
- [x] Kaggle authentication configured
- [x] Correct project dataset identified
- [x] Dataset downloaded
- [x] Dataset extracted
- [x] Dataset directory structure verified
- [x] Class folders verified
- [x] Dataset image counts verified
- [ ] Patient-level train/validation split
- [ ] `splits.csv`
- [ ] Baseline model
- [ ] Final model

### Notes
The verified extracted image count is 5,856. The project brief describes the dataset as 5,863 images; this discrepancy has not been investigated yet and will be examined during Stage 1 EDA rather than assumed or silently reconciled.

The original 16-image validation set will not be used for tuning. Patient-level splitting will be performed during Stage 1 to reduce patient-level data leakage risk.
