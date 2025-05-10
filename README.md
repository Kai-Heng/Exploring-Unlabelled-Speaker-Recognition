# Exploring-Unlabelled-Speaker-Recognition
This repository contains the initial scaffolding for a proof-of-concept speaker clustering system developed for the AIngineer take-home assignment. The goal is to group speech segments from a collection of unlabeled audio recordings by speaker identity, using fully unsupervised methods.

## 🔍 Problem Overview

Given a dataset of microphone recordings with no labels, the task is to identify and cluster speech segments by speaker. The output should assign consistent IDs to each speaker such that future segments can be matched to their respective clusters.

## 🧱 Planned Architecture

The project will be developed in modular stages:

1. **Preprocessing**
   - Resample audio to 16 kHz mono
   - Apply Voice Activity Detection (VAD)
   - Extract speech-only segments

2. **Embedding Extraction**
   - Use pre-trained models (e.g., ECAPA-TDNN) to convert segments into fixed-length speaker embeddings

3. **Clustering**
   - Apply unsupervised clustering (e.g., HDBSCAN, AHC) to group similar speaker embeddings

4. **Post-processing**
   - Refine cluster assignments and label stability
   - Prepare system for inference on unseen segments
   
---

*Project bootstrapped May 2025 – innovation in progress 🚧*
