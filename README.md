---
title: HAR Platform
sdk: docker
app_port: 7860
---

# HAR-Platform

Interactive Human Activity Recognition platform (Streamlit + PyTorch + Plotly) for testing wearable-sensor CSV files
from different datasets (UCI HAR, WISDM, MotionSense, HHAR, custom), with cross-dataset domain analysis,
model/adaptation comparison and a leakage-safe research-experiment mode.

**Principles:** no dummy predictions, no fabricated metrics. Without a trained checkpoint the app refuses to predict.
Accuracy / macro-F1 appear only when valid ground-truth labels exist; otherwise only descriptive statistics are shown.

## Install
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # install a CUDA build of torch first if you want a GPU
```

## 1. Train a source model (required once — no weights are shipped)
Download UCI HAR (https://archive.ics.uci.edu/dataset/240) into `datasets/UCI HAR Dataset/`, then:
```bash
python -m models.train --uci-har-dir "datasets/UCI HAR Dataset" --channels acc
```
* `--channels acc` (3 channels, default) works for files without a gyroscope; `acc_gyro` uses 6 channels.
* Subject-wise train/val split of the official training subjects; test on the official held-out test subjects.
* Writes `models/checkpoints/har_source.pt` (weights, classes, channels, **training normalisation statistics**) and a
  `*.reference.npz` sample of training windows used by the domain-analysis module.
* Classes come from the training data: UCI HAR gives walking, upstairs, downstairs, sitting, standing, lying
  (no running). To add running etc., train from other CSVs with `--csv-spec` (see `datasets/example_spec.json`).

## 2. Launch
```bash
streamlit run app.py
```

## Pages
| Page | What it does |
|---|---|
| 1 Upload & preprocess | multi-file upload, preview, auto-detect + manual column mapping (formats A–D, long/sensor-type format, header-less WISDM), units, time unit, fs, label mapping, preprocessing report |
| 2 Signal visualisation | per-axis, combined, gyroscope, magnitude, FFT, window predictions; time-interval selector |
| 3 Activity prediction | per-window class/confidence/probabilities, activity shares, transition timeline, low-confidence windows, ground-truth evaluation (if labels) |
| 4 Multi-CSV comparison | dataset table (accuracy/macro-F1 only when labelled), confusion matrices, per-class accuracy/F1, magnitude & PSD comparisons, cross-dataset prediction differences |
| 5 Domain analysis | MMD + permutation test (hand-crafted and embedding space), layer-wise MMD, PCA / t-SNE / UMAP, domain-shift report |
| 6 Model comparison | scratch / source-only / frozen / full fine-tune / adaptive layer-wise on identical subject-wise test windows |
| 7 Research experiments | source & target roles, fractions 1/5/10/20/100 %, seeds, group k-fold, MLflow + `results/` export |
| 8 Reports | ZIP with predictions, preprocessing stats, evaluation, comparison, domain-shift report |

Upload and process files on page 1 once; every other page (including the experiment pages) picks them from there.

## Preprocessing (identical for every file)
1. map columns → 2. drop NaN rows (counted) → 3. merge duplicate timestamps, sort → 4. convert to **g** and **rad/s**
(auto-guessed, overridable) → 5. split into segments at subject change, label change or time gap (> max(1 s, 5×dt)) →
6. resample each segment to **50 Hz** (anti-alias low-pass when decimating) → 7. 128-sample windows, 50 % overlap,
**never across segments** → 8. z-score with the statistics stored in the checkpoint → 9. shape/finite validation.

Models expect *total* acceleration (gravity included). Body/linear acceleration columns are combined with detected gravity
columns when present; otherwise a warning is shown.

## Leakage control (experiment mode)
* Splits are by subject (`file::subject`); with < 3 subjects a purged temporal split inside each segment is used (one-window gap).
* Validation windows come from training subjects only. Target fractions subsample train and validation alike.
* Normalisation: pretrained strategies keep source statistics; *scratch* fits statistics on its training subset only.
* All strategies share the same test windows per (seed, fold, fraction). Metrics are computed over the common activities.

## About the "adaptive layer-wise" model
`adaptation/strategies.py::_adaptive_layerwise` is a concrete, documented implementation of the idea: per-layer MMD between
the source reference sample and the target training windows sets each layer's learning rate (between `min_lr_scale` and 1×)
and the strength of an L2-SP anchor to the pretrained weights (stronger for layers with little shift). If your proposed
method differs, replace that function — the rest of the pipeline is unchanged.

## Notes / caveats
* Confidence = max softmax probability; **not calibrated**. ECE is reported when ground truth exists.
* Macro-F1 averages over classes present in the ground truth; windows with labels the model cannot output are excluded and counted.
* Time share per activity = windows × stride / fs (non-overlapping equivalent).
* WISDM "stairs" (no up/down split) is excluded by default; edit the label map if you want to treat it differently.
* Checkpoints are loaded with `torch.load(weights_only=True)`.

## Tests
```bash
pytest                      # torch tests skip automatically if torch is missing
python tests/run_basic.py   # torch-free tests without pytest
```

## Layout
`app.py`, `ui/` (pages), `preprocessing/`, `models/` (CNN, trainer, registry, `train.py`), `adaptation/`, `evaluation/`,
`visualization/`, `experiments/` (splits, pool, runner, tracking), `datasets/` (loaders), `config/`, `tests/`, `uploads/`, `results/`.
