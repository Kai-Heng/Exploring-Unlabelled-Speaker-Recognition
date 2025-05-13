# Exploring-Unlabelled-Speaker-Recognition

> **Refer to the full documentation (`Exploring Unlabelled Speaker Recognition Documentation.pdf`) if you need more detailed explanations.**

---

## 1 Overview

This repository proposes and outlines a **fully‑unsupervised, high‑accuracy pipeline** for identifying *≈200* unique speakers in an unlabeled collection of WAV recordings.
The core idea:

1. **Pre‑process** audio (resample, VAD, normalise).
2. **Embed** each utterance with a **pre‑trained ECAPA‑TDNN** (or x‑vector) network.
3. **Cluster** the embedding vectors (HDBSCAN / Spectral Clustering) to form speaker IDs.
4. **Validate** clusters using internal metrics & manual spot‑checks.

Most of the heavy lifting is done by the powerful speaker‑embedding model, allowing accurate separation without labelled data.

---

## 2 Data Exploration & Analysis

| Step                                 | Purpose                               |
| ------------------------------------ | ------------------------------------- |
| Listen to samples                    | Gauge voice diversity & noise levels  |
| Waveform plots                       | Detect silence, clipping, long pauses |
| Spectrograms                         | Visualise pitch & formant patterns    |
| Basic statistics (pitch, MFCC means) | Spot broad clusters (e.g. gender)     |

Challenges discovered:

* Varying recording quality & background noise.
* Possible near‑identical voices.
* No ground‑truth labels for evaluation.

---

## 3 Proposed Solution & Justification

| Component           | Choice                            | Rationale                                             |
| ------------------- | --------------------------------- | ----------------------------------------------------- |
| **Embeddings**      | ECAPA‑TDNN (192‑D)                | SOTA robustness; trained to separate speakers         |
| **Distance Metric** | Cosine                            | Invariant to loudness; aligns with embedding training |
| **Clustering**      | HDBSCAN –or– Spectral             | Handles unequal cluster sizes; auto‑detects outliers  |
| **Evaluation**      | Silhouette, DBI, manual listening | Only viable when labels are absent                    |

Why it works: embeddings compress speaker identity into a compact vector; clustering then groups vectors that are naturally close in this space.

---

## 4 Conceptual Implementation Strategy

```text
preprocess.py
  - resample 16 kHz mono
  - voice‑activity‑detect & trim
  - loudness normalise

extract_embeddings.py
  - load ECAPA model (SpeechBrain)
  - for each .wav → 192‑D vector → save to embeddings.npy

cluster_embeddings.py
  - load embeddings.npy
  - run HDBSCAN(min_cluster_size=2)
  - save cluster_labels.csv

evaluate.py
  - silhouette_score(embeddings, labels)
  - flag low‑silhouette recordings for manual review
```

All code is **conceptual/pseudocode** and can be converted to runnable Python with minimal effort.

---

## 5 Challenges & Mitigations

| Challenge                 | Mitigation                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------- |
| Similar‑sounding speakers | High‑resolution ECAPA embeddings; iterative re‑clustering of suspect large clusters |
| Noise / channel mismatch  | VAD + light spectral denoising; robustness of ECAPA (trained with augmentation)     |
| No ground truth           | Internal metrics; targeted manual listening; verification‑score cross‑checks        |
| Over‑ / under‑clustering  | Compare HDBSCAN vs. fixed‑k AHC, inspect cluster size distribution                  |


---

## 6 Repository Layout

```
/README.md              ← THIS FILE
/Exploring Unlabelled Speaker Recognition Documentation.pdf      ← Detailed write‑up (all questions answered)
/code/
   preprocess.py*       ← audio cleaning (conceptual)
   extract_embeddings.py*
   cluster_embeddings.py*
   evaluate.py*
```

*(starred files are high‑level pseudocode – edit into real scripts as you iterate)*

---

## 7 Quick‑Start (Conceptual)

```bash
# 1. Prepare env
conda create -n speaker-dia python=3.10
conda activate speaker-dia
pip install speechbrain hdbscan librosa matplotlib

# 2. Run pipeline
python code/preprocess.py   # cleans /data/*.wav → /data/clean/*.wav
python code/extract_embeddings.py  # → embeddings.npy
python code/cluster_embeddings.py  # → cluster_labels.csv
python code/evaluate.py     # prints silhouette & saves plots
```

---

## 8 Next Steps

1. Swap ECAPA for any newer embedding model if desired.
2. Iterate clustering parameters to hit exactly 200 clusters.
3. Build a small web dashboard to audition clusters and merge/split interactively.

---

*Project bootstrapped May 2025 – innovation in progress 🚧*
