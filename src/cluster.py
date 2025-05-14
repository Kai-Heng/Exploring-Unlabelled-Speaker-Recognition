#!/usr/bin/env python
"""Cluster ECAPA embeddings with HDBSCAN."""
import os, argparse, numpy as np, hdbscan, json
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans

def main(emb_dir, out_json, min_cluster_size=2):
    X = np.load(os.path.join(emb_dir, "embeddings.npy"))
    X = normalize(X)                 # L2‑normalise → cosine ≈ euclidean
    print(f"Clustering {X.shape[0]} embeddings …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                metric="euclidean")
    labels = clusterer.fit_predict(X)

    # Spectral (needs n_clusters)
    # clusterer = SpectralClustering(n_clusters=200, affinity='nearest_neighbors',
    #                             assign_labels='kmeans')
    # labels = clusterer.fit_predict(X)

    # k-Means
    # clusterer = KMeans(n_clusters=200, n_init=20, random_state=42)
    # labels = clusterer.fit_predict(X)
    
    # Save mapping filename → cluster_id
    files = np.loadtxt(os.path.join(emb_dir, "filenames.txt"), dtype=str).tolist()
    mapping = {fname: int(lbl) for fname, lbl in zip(files, labels)}
    with open(out_json, "w") as f:
        json.dump(mapping, f, indent=2)
    # Quick stats
    n_clusters = len(set(labels) - {-1})
    n_outliers = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_outliers} outliers.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/embeddings")
    ap.add_argument("--out_json", default="data/embeddings/cluster_map.json")
    ap.add_argument("--min_cluster_size", type=int, default=2)
    args = ap.parse_args()
    main(args.emb_dir, args.out_json, args.min_cluster_size)
