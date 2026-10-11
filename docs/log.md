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


### Day 3 — Data Understanding & EDA

**Status:** Core EDA completed.

**Dataset counts**

| Split | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

**Observations**
- The training data is class-imbalanced, with approximately 2.89 times as many Pneumonia images as Normal images.
- A visual grid of 16 random training images per class was generated.
- In a sample of 1,000 training images, 968 were grayscale and 32 were RGB.
- Image dimensions vary substantially, supporting the need for resizing during preprocessing.
- Brightness histograms were generated from samples of 100 images per class.
- All 5,856 files in the expected class folders had recognized image extensions and passed the image-readability check.

**Decisions**
- Preserve the original 624-image test set.
- Do not use the original 16-image validation set for model tuning.
- Record measured results rather than assume expected counts.
- Keep the screening-aid-only framing.

### Day 4 — Patient Metadata & Original Split Leakage Audit

**Status:** Metadata extraction and original-split overlap audit completed in Colab.

**Verified findings**
- Total images indexed: 5,856.
- Unique filename-derived patient identifiers: 3,118.
- Images with missing parsed identifiers: 0.
- Patients with more than one image: counted in the metadata analysis.
- Train–validation identifier overlap: 0.
- Train–test identifier overlap: 170.
- Validation–test identifier overlap: 0.
- Test images whose parsed identifiers also appear in training: 320.
- Test images whose parsed identifiers do not appear in training: 304.

**Interpretation and limitations**
- Identifiers were parsed from filenames and are proxies; they have not been independently verified as patient identities.
- The original split has potential patient-level leakage, particularly between train and test.
- The 624 original test images remain unchanged. No model training or new split was performed during this audit.
- The verified dataset count remains 5,856, not the 5,863 stated in the project brief. The discrepancy remains unexplained after EDA and integrity checks.

**Artifacts generated in Colab**
- `image_metadata.csv`
- `original_split_patient_overlap.csv`
- `leakage_audit.md`

These were generated under `/content/chestscan_artifacts/`. They still need to be copied into the appropriate locations in the local repository.

**Next**
Wait for explicit authorization before beginning Day 5. The next planned stage is a patient-level train/validation split with leakage checks and a `splits.csv` manifest.