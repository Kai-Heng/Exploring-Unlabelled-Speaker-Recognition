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

| **Exploration Step**                 | **Purpose**                                                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Listening to audio samples           | Understand voice diversity, accents, noise, and potential overlap between speakers                  |
| Plotting waveform diagrams           | Identify amplitude variations, silence regions, clipping, or speech rhythm anomalies                |
| Spectrogram visualization            | Observe pitch, formant structures, and frequency distribution across time                           |
| Statistical feature extraction       | Analyze MFCCs, pitch, and speaking rate to detect broad groupings (e.g., by gender or vocal traits) |
| Silence and speech segmentation      | Detect and isolate active voice regions using VAD; flag recordings with multiple speakers           |
| Feature space projection (PCA/t-SNE) | Visualize embedding clusters to assess speaker separability and clustering potential                |


Challenges discovered:

* Varying recording quality & background noise.
* Possible near‑identical voices.
* No ground‑truth labels for evaluation.

---

## 3 Proposed Solution & Justification

| **Component**            | **Selected Method**                                                     | **Justification**                                                                       |
| ------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Speaker Embeddings**   | ECAPA‑TDNN (192‑D)                                                      | State-of-the-art model with strong robustness to noise and speaker variability          |
| **Similarity Metric**    | Cosine similarity                                                       | Scale-invariant; matches the training objective of modern speaker embeddings            |
| **Clustering Algorithm** | HDBSCAN or Spectral Clustering                                          | Handles variable cluster sizes; identifies outliers; no need to predefine cluster count |
| **Evaluation Strategy**  | Silhouette score, Davies-Bouldin Index (DBI), manual audio verification | Effective for unsupervised validation when no ground truth is available                 |


Why it works: embeddings compress speaker identity into a compact vector; clustering then groups vectors that are naturally close in this space.

---

## 4 Conceptual Implementation Strategy

```text
src/
├── preprocess.py         # 1.  Audio cleaning
│   • resample to 16 kHz mono
│   • apply VAD → trim silence / noise
│   • loudness‑normalise → data/clean/
│
├── embed.py              # 2.  Speaker‑vector extraction
│   • load SpeechBrain ECAPA‑TDNN
│   • each WAV → 192‑D embedding
│   • stack → embeddings.npy  +  filenames.txt
│
├── cluster.py            # 3.  Unsupervised grouping
│   • load embeddings.npy  (L2‑normalised)
│   • run HDBSCAN(min_cluster_size=2)
│   • output cluster_map.json  {filename: cluster_id}
│
└── evaluate.py           # 4.  Quality checks
    • compute Silhouette & Davies‑Bouldin indices
    • plot cluster‑size histogram
    • list recordings with low silhouette for manual review
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
Exploring-Unlabelled-Speaker-Recognition/
├── data/
│   ├── raw/          # <‑‑ 200 original WAVs go here
│   ├── clean/        # auto‑generated VAD‑trimmed clips
│   └── embeddings/   # auto‑generated .npy vectors
├── src/
│   ├── preprocess.py
│   ├── embed.py
│   ├── cluster.py
│   └── evaluate.py
├── Exploring Unlabelled Speaker Recognition Documentation.pdf   # <‑‑ Detailed write‑up
├── README.md
└── requirements.txt
```
---

## 7 Quick‑Start (Conceptual)

```bash
# 1. Prepare env
python -m venv AINgineer && source AINgineer/bin/activate  # Windows: AINgineer\Scripts\activate
pip install -r requirements.txt

# 2. Run pipeline
python src/preprocess.py   # cleans /data/*.wav → /data/clean/*.wav
python src/embed.py        # → embeddings.npy
python src/cluster.py      # → cluster_map.json
python src/evaluate.py     # prints silhouette & saves plots
```

---

## 8 Next Steps

1. Swap ECAPA for any newer embedding model if desired.
2. Iterate clustering parameters to hit exactly 200 clusters.
3. Build a small web dashboard to audition clusters and merge/split interactively.

---

*Project bootstrapped May 2025 – innovation in progress 🚧*
