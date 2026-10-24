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

These Colab artifacts were used as the basis for the local Day 5 split manifest.

**Outcome**
Day 5 split creation is documented below.
### Day 5 — Patient-Level Split Manifest

**Status:** Split manifest generated and validation checks passed locally.

**Verified split counts**

| New split | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Train | 1,147 | 2,912 | 4,059 |
| Validation | 202 | 514 | 716 |
| Excluded overlap | 0 | 457 | 457 |
| Original test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

**Decisions and limitations**
- Preserved all 624 original test images.
- Labelled 304 test images as `test_clean` and all 624 as `test_full`.
- Excluded 457 non-test images whose filename-derived identifiers overlap with the original test set.
- The excluded images are all labelled Pneumonia, changing the class distribution of the eligible training data.
- Filename-derived identifiers remain unverified proxies for patient identity.
- The original 16-image validation set is not used for tuning.
- Training class weights, calculated from the new training split, are `NORMAL: 1.769398` and `PNEUMONIA: 0.696944`.
- No model training was performed.

**Artifact**
- `splits.csv` — 5,856 rows, with the required six columns.
- `scripts/create_splits.py` — split-manifest generation script.

**Validation**
- All paths are unique.
- All labels are recognized.
- The original test set contains exactly 624 images.
- The clean-test cohort contains exactly 304 images.
- Leakage assertions passed for the new train/validation/test partitions, using filename-derived identifiers.

**Next**
Review the final diff and commit the Day 5 artifacts. Do not begin Day 6 without explicit authorization.

## Day 6 — tf.data Pipeline

### Completed
- Implemented `src/data.py` with manifest loading and image-path resolution.
- Decoded JPEG images, converted them to RGB, and resized them to 224x224.
- Implemented training-only augmentation: rotation, zoom, horizontal flip, brightness, and contrast.
- Added training shuffle, batching (32), parallel mapping, and prefetching.
- Built train, validation, test_full, and test_clean datasets.
- Verified image batches have shape (32, 224, 224, 3), labels have shape (32,), and pixel values are within [0, 255].
- Visually compared original and augmented images; augmentation was active and images remained recognizable.
- Confirmed zero filename-derived patient-ID overlap between the new train and validation partitions.

### Limitations
- Filename-derived IDs are proxies, not verified patient identities.
- No model has been trained; performance metrics remain unavailable.
- Augmentation plausibility has only been visually checked on a small sample.
- ChestScan is a screening aid, not a diagnostic tool.

### Next
Begin Day 7: implement and train the baseline CNN.

### Day 7 — Baseline CNN

- **Objective:** Build and train a small CNN from scratch.
- **Dataset:** Existing `splits.csv`; 4,059 training images, 716 validation images, and 624 untouched test images.
- **Architecture:** Four convolutional layers, global average pooling, dropout, a 64-unit dense layer, and sigmoid output.
- **Training:** TensorFlow 2.21.0, GPU enabled, seed 42, batch size 32, class weights, and training-only augmentation.
- **Duration:** 15 epochs.
- **Best validation AUC:** 0.9820 at epoch 12.
- **Checkpoint:** `/content/chestscan_artifacts/models/day7_cnn_best.keras`
- **Training log:** `/content/chestscan_artifacts/day7_training_log.csv`
- **Pending:** Interpret learning curves, compute validation metrics, and record results in `docs/experiments.md`.
- **Limitations:** Filename-derived patient IDs are proxies. This model is experimental and is not a medical diagnostic tool.


## Day 8 — Baseline CNN evaluation

- Plotted baseline training and validation learning curves.
- Verified that the persistent and runtime `splits.csv` files have matching SHA-256 hashes.
- Evaluated the currently saved baseline CNN checkpoint on 716 validation images only.
- Validation accuracy: 0.8953.
- Validation precision: 0.9933.
- Validation recall: 0.8599.
- Validation F1-score: 0.9218.
- Validation AUC: 0.9857.
- Confusion matrix: TN=199, FP=3, FN=72, TP=442.
- Saved validation metrics to `artifacts/metrics/day8_validation_metrics.json`.
- Created `docs/experiments.md` to document experiment results.
- Checkpoint provenance remains uncertain because subsequent unlogged training runs may have overwritten the original Day 7 checkpoint.
- Test cohorts remain untouched. Threshold selection, calibration, and triage band selection remain pending.


## Day 9 — Baseline CNN: Learning-Rate Experiment

- **Change:** Adam learning rate reduced from 1e-3 to 3e-4.
- **Training:** 15 epochs completed; best epoch was 14, selected by validation AUC.
- **Best validation AUC:** 0.9628.
- **Epoch 14 validation metrics:** accuracy 0.9022, precision 0.9684,
  recall 0.8930, loss 0.2282.
- **Diagnosis:** Training loss decreased, but validation metrics fluctuated
  substantially between epochs.
- **Outcome:** Improvement over the historical baseline is not established.
- **Provenance issue:** The Day 9 run overwrote the Day 7 checkpoint and
  training log. Recovery copies are identical to the current files, so the
  original baseline artifacts have not been recovered.
- **Data integrity:** `splits.csv` remains unchanged. Test cohorts were not
  used for model selection.
- **Next:** Begin Day 10 prerequisites for MobileNetV2 transfer learning.

ChestScan is an experimental screening aid, not a diagnostic tool.


### Day 11 — MobileNetV2 Phase 2 Fine-Tuning

- Loaded the Phase 1 checkpoint and fine-tuned the top MobileNetV2 layers using learning rate `1e-5`, keeping Batch Normalization layers frozen.
- Used class weights and patient-level training/validation partitions.
- Early stopping triggered at epoch 12 and restored in-memory weights from epoch 8.
- Best recorded validation recall: **0.9805**.
- Best recorded validation accuracy: **0.9777**.
- Best recorded validation AUC: **0.9973** (epoch 5).
- Lowest recorded validation loss: **0.0614** (epoch 8).
- Checkpoint: `artifacts/models/mobilenetv2_phase2_v1_best.keras`
- Log: `artifacts/logs/mobilenetv2_phase2_v1.csv`
- Test set remains untouched. No final model selected.
- Pending: independent checkpoint and log verification.
- Medical-safety note: ChestScan is a screening aid, not a diagnostic tool.

### Day 17 — Error Analysis and Triage Evaluation

**Status:** Complete

**Work performed**
- Built a case-level audit table from the frozen Day 16 predictions for all 624 test images.
- Inspected false negatives, false positives, and representative uncertain-band cases.
- Calculated uncertain-band error capture separately for `test_full` and `test_clean`.
- Recorded findings and limitations in `artifacts/evaluation/day17_error_analysis.md`.
- Saved the complete case-level table to `artifacts/evaluation/day17_case_audit.csv`.

**Results**
- Full test: 11 false negatives, 50 false positives, 61 total errors.
- Clean test: 1 false negative, 50 false positives, 51 total errors.
- Full-test uncertain-band error capture: 21/61 (34.43%).
- Clean-test uncertain-band error capture: 21/51 (41.18%).
- The uncertain band captured no false negatives in the full-test cohort.
- ECE was 0.1310 on the full test and 0.2876 on the clean subset.

**Limitations**
- The uncertain band does not capture all classification errors.
- High-confidence false positives and false negatives remain.
- The clean cohort is smaller and class-imbalanced.
- Filename-derived identifier overlap indicates a leakage risk but does not prove that leakage caused all observed performance differences.
- ChestScan remains a screening aid, not a diagnostic tool.

**Integrity**
- Model checkpoint, decision threshold (0.905344), and uncertain-band bounds (0.830344–0.980344) remain frozen.
- No retraining or test-driven tuning was performed.

**Next:** Begin Day 18 Grad-CAM explainability as a stretch goal, while protecting the remaining API, UI, deployment, and documentation schedule.

### Day 19 — Grad-CAM Review and Application Handoff

**Status:** Grad-CAM investigation complete; application handoff in progress.

**Findings**
- Grad-CAM maps were finite and non-zero, but did not convincingly
  demonstrate lung-focused attention.
- These visual findings do not prove shortcut learning.
- Findings were documented in
  `artifacts/gradcam_day18_19_findings.md`.
- Local `backend/`, `frontend/`, and `tests/` directories exist but
  contain no files.
- The local repository does not contain the model checkpoint or
  `config.json`; their availability in the model environment must
  be verified before API inference is implemented.

**Integrity**
- Preserve the frozen checkpoint, decision threshold, and uncertain
  band. No retraining or threshold changes authorized.
- ChestScan remains a screening aid, not a diagnostic tool.

**Next:** Begin Stage 6 prerequisites and verify model access before
implementing the first backend component.

Day 20: Implemented and smoke-tested the frozen MobileNetV2 inference component. Verified checkpoint checksum, configuration, model loading, embedded preprocessing, prediction output, corrupt-image rejection, and syntax compilation. Real-image parity remains unverified; no model or threshold changes.


## Day 21 — Backend API Foundations

**Objective:** Implement a FastAPI backend for ChestScan using the frozen MobileNetV2 model and existing inference pipeline.

### Implementation
- Created `backend/main.py` with FastAPI application and lifecycle management.
- Loaded the frozen model and configuration at startup using `load_config()` and TensorFlow Keras.
- Implemented `GET /health` to report API status and model availability.
- Implemented `POST /predict` accepting JPEG and PNG uploads through the `file` field.
- Added a 5 MB upload limit and validation for empty files, unsupported media types, corrupt images, and mismatches between declared and actual image formats.
- Added CORS configuration for the frontend at `http://localhost:5173`.
- Added structured prediction responses containing the predicted class, pneumonia probability, decision threshold, uncertainty band, inference time, and medical disclaimer.
- Added error handling for invalid uploads and inference failures.

### Verification
- Python syntax compilation and API import checks passed.
- Model loaded successfully during Uvicorn startup.
- `/health` returned HTTP 200 with `{"status":"ok","model_loaded":true}`.
- Synthetic PNG prediction returned HTTP 200 with the expected response fields.
- Unsupported media type returned HTTP 415.
- Corrupt image and mismatched image format returned HTTP 400.
- Oversized upload returned HTTP 413.
- Repeated valid PNG prediction after the validation fix returned HTTP 200.

### Notes and Limitations
- Synthetic images were used for API smoke testing because no local chest X-ray images were available. Their predictions have no clinical meaning.
- Real-image inference parity with the notebook and automated API regression tests remain pending.
- The frozen model, decision threshold (`0.905344`), and uncertainty band (`0.830344`–`0.980344`) were not changed.

**Status:** Day 21 implementation and manual API smoke tests completed. Automated regression testing is planned for Day 22.

**Disclaimer:** Screening aid only. Not a medical diagnosis.

### Day 22 — Backend API Regression Testing

- Verified that the configured checkpoint SHA-256 matches the actual frozen model file.
- Updated `backend/inference.py` preprocessing to match the TensorFlow evaluation pipeline used in the notebook.
- Added `tests/test_inference_parity.py` to compare real-model API predictions against an independently computed TensorFlow reference.
- Ran the complete test suite: **7 passed, 0 failed**.
- Verified probability parity on one sample image within an absolute tolerance of `1e-5`.
- Confirmed the model checkpoint, decision threshold (`0.905344`), and uncertainty band (`0.830344–0.980344`) remain unchanged.
- Updated `.gitignore` to exclude the local sample image and downloaded notebook.
- Existing Starlette/httpx deprecation warning remains non-blocking.
- **Limitations:** Parity was verified on one sample image only; full test-set parity has not been established. Existing evaluation limitations, including potential dataset overlap and unvalidated clinical performance, remain.
- **Status:** Backend regression tests pass. Ready for final change review and commit.