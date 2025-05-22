# Exploring-Unlabeled-Speaker-Recognition

> **Refer to the full documentation (`Exploring Unlabeled Speaker Recognition Documentation.pdf`) if you need more detailed explanations.**

---

## Overview

This repository proposes and outlines a **fully‑unsupervised, high‑accuracy pipeline** for identifying *≈200* unique speakers in an unlabeled collection of WAV recordings.
The core idea:

1. **Pre‑process** audio (resample, VAD, normalise).
2. **Embed** each utterance with a **pre‑trained ECAPA‑TDNN** (or x‑vector) network.
3. **Cluster** the embedding vectors (HDBSCAN / Spectral Clustering) to form speaker IDs.
4. **Validate** clusters using internal metrics & manual spot‑checks.

Most of the heavy lifting is done by the powerful speaker‑embedding model, allowing accurate separation without labelled data.

---

## 🔍 Data Exploration & Analysis

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

## 💡 Proposed Solution & Justification

| **Component**            | **Selected Method**                                                     | **Justification**                                                                       |
| ------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Speaker Embeddings**   | ECAPA‑TDNN (192‑D)                                                      | State-of-the-art model with strong robustness to noise and speaker variability          |
| **Similarity Metric**    | Cosine similarity                                                       | Scale-invariant; matches the training objective of modern speaker embeddings            |
| **Clustering Algorithm** | HDBSCAN                                          | Handles variable cluster sizes; identifies outliers; no need to predefine cluster count |
| **Evaluation Strategy**  | Silhouette score, Davies-Bouldin Index (DBI), manual audio verification | Effective for unsupervised validation when no ground truth is available                 |


Why it works: embeddings compress speaker identity into a compact vector; clustering then groups vectors that are naturally close in this space.

---

## 🛠️ Conceptual Implementation Strategy

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

---

## ⚠️ Challenges & Mitigations

| Challenge                 | Mitigation                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------- |
| Similar‑sounding speakers | High‑resolution ECAPA embeddings; iterative re‑clustering of suspect large clusters |
| Noise / channel mismatch  | VAD + light spectral denoising; robustness of ECAPA (trained with augmentation)     |
| No ground truth           | Internal metrics; targeted manual listening; verification‑score cross‑checks        |
| Over‑ / under‑clustering  | Compare HDBSCAN vs. fixed‑k AHC, inspect cluster size distribution                  |


---

## 🗂️ Repository Layout

```
Exploring-Unlabeled-Speaker-Recognition/
├── data/
│   ├── raw/          # <-- original WAVs go here
│   ├── combined/     # <-- concatenated WAVs go here (Optional)
│   ├── clean/        # auto‑generated VAD‑trimmed clips
│   └── embeddings/   # auto‑generated .npy vectors
├── src/
│   ├── extract_recordings.py # copy Audio‑MNIST --> data/raw/
│   ├── combine_audio.py # concatenate digit recordings --> data/combined/
│   ├── preprocess.py
│   ├── embed.py
│   ├── cluster.py
│   └── evaluate.py
├── Exploring Unlabeled Speaker Recognition Documentation.pdf   # <-- Detailed write‑up
├── README.md
└── requirements.txt
```
---

## 🚀 Quick‑Start

```bash
# 0) Clone repository
git clone https://github.com/Kai-Heng/Exploring-Unlabelled-Speaker-Recognition.git
cd Exploring-Unlabelled-Speaker-Recognition

# 1) Set up Python 3.12 environment
python3.12 -m venv AINgineer
source AINgineer/bin/activate          # (Windows → AINgineer\Scripts\activate)
pip install -r requirements.txt

# 2) Prepare data  
# • If you already have WAVs in  data/raw/  → **skip this step.**
# • Otherwise, populate it from the Audio‑MNIST download (≈ 30 000 files):
python src/extract_recordings.py --src /path/to/AudioMNIST/data/
python src/combine_audio.py

# 2. Run pipeline
python src/preprocess.py   # cleans /data/*.wav → /data/clean/*.wav
python src/embed.py        # → embeddings.npy
python src/cluster.py      # → cluster_map.json
python src/evaluate.py     # prints silhouette & saves plots
```

---

## 📊 Clustering Evaluation Results (Experiment with 60 speakers)

| Algorithm                                            | Silhouette Score<sup>†</sup> | Davies‑Bouldin Index<sup>‡</sup> | Observations                                                                                                       |
| ---------------------------------------------------- | ---------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **HDBSCAN**                                          | **0.510**                    | **0.824**                        | Same overall quality as K‑Means on this dataset, while automatically discovering cluster count and flagging noise. |
| **K‑Means** (k = 60, n\_init = 20, seed = 42)        | **0.510**                    | **0.824**                        | Fast and simple; matched HDBSCAN’s scores but required pre‑setting *k* and forces every point into a cluster.      |
| **Spectral Clustering** (nearest‑neighbors affinity) | −0.024                       | 2.614                            | Performed poorly — negative silhouette and high DB index indicate ill‑formed clusters for this embedding.          |
| **Spectral Clustering** (radial basis function) | **0.510**                       | **0.824**                            | Same overall quality as K‑Means on this dataset while using rdf to construct affinity matrix          |

<sup>†</sup> **Silhouette score** ∈ \[−1, 1]  (+1 = well‑separated,  0 ≈ overlap,  −1 = mis‑clustered. Higher is better.)

<sup>‡</sup> **Davies‑Bouldin index** ≥ 0  (0 = perfectly compact/isolated clusters. Lower is better.)


Complete JSON mapping of recording → cluster is in ```data/embeddings/cluster_map.json```.

---

### 📈 Metric Ranges & Why They Matter

* **Silhouette Score (S)**

  * Range **−1 → 1**.
  * *Interpretation*: S ≳ 0.5 is generally considered good; S < 0 suggests points are assigned to the wrong clusters.

* **Davies‑Bouldin Index (DBI)**

  * Range **0 → ∞**.
  * *Interpretation*: DBI ≲ 1 indicates compact, well‑separated clusters; values ≫ 1 mean high overlap.

These complementary metrics help avoid relying on a single view of cluster quality.

---

### Quick Takeaways

* **HDBSCAN** matched K‑Means’ quantitative scores **without** requiring you to guess *k* and **flagged noise points** automatically — valuable for speaker‑embedding spaces that may contain outliers.
* **Spectral Clustering** under‑performed on the same embeddings, suggesting either a poor affinity choice or that the speaker manifold is not well captured by a graph‑based approach here.
* Given equal scores, you might prefer **HDBSCAN** for its flexibility and practical benefits (automatic cluster count, noise handling).

---

## 🎧 About the Dataset — *AudioMNIST (Combined)*

| Property                | Value                                       |
| ----------------------- | ------------------------------------------- |
| **Total clips**         | 30 000 WAV files                            |
| **Speakers**            | 60 (one folder per speaker)                 |
| **Digits**              | 0 – 9 (spoken)                              |
| **Samples per speaker** | 500 (≈ 50 clips per digit before combining) |
| **Metadata**            | `audioMNIST_meta.txt` (gender, age, etc.)   |
| **Source**              | Kaggle dataset “Audio MNIST”        |

For this task, the raw digit recordings for each speaker were **concatenated across digits** — that is, each combined clip contains a sequence of digits from 0 to 9 spoken in order. This process was repeated 50 times per speaker, resulting in **60 speakers × 50 combined recordings = 3 000 utterances**. These longer clips were created to ensure greater phonetic variability per recording, enabling the CAPA‑TDNN model to better capture speaker-specific features for clustering.

---

## 🧠 Future Work

While this pipeline demonstrates solid performance on a structured dataset like AudioMNIST, several areas remain open for extension and experimentation:

* **Generalization to real-world data:** Test on more varied datasets with spontaneous speech, background noise, and cross-channel recordings.
* **Dynamic clustering techniques:** Investigate semi-supervised refinement or pseudo-label bootstrapping using high-confidence clusters.
* **Incremental speaker addition:** Explore how the model handles unseen speakers and how embeddings generalize to new voices.
* **Embedding model fine-tuning:** Fine-tune ECAPA-TDNN on unlabeled target domain recordings using self-supervised objectives to better capture domain-specific features.
* **Post-processing with centroid modeling:** Use cluster centroids as speaker prototypes and apply few-shot classification for new recordings.

---
