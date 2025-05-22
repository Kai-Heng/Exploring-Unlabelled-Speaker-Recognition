"""Cluster ECAPA embeddings with HDBSCAN."""
import os, argparse, numpy as np, hdbscan, json
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans

def main(emb_dir, out_json, min_cluster_size=2):
    X = np.load(os.path.join(emb_dir, "embeddings.npy"))
    X = normalize(X)                 # L2‑normalise → cosine ≈ euclidean
    print(f"Clustering {X.shape[0]} embeddings …")

    # Decent Clustering Method HDBSCAN & K-means
    # HDBSCAN [Silhouette = 0.510  |  Davies‑Bouldin = 0.824]
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
    #                             metric="euclidean")
    # labels = clusterer.fit_predict(X)

    # k-Means [Silhouette = 0.510  |  Davies‑Bouldin = 0.824]
    # clusterer = KMeans(n_clusters=60, n_init=20, random_state=42)
    # labels = clusterer.fit_predict(X)

    # Spectral [Silhouette = -0.024  |  Davies‑Bouldin = 2.614]
    # 'nearest_neighbors': construct the affinity matrix by computing a graph of nearest neighbors.
    # 'rbf': construct the affinity matrix using a radial basis function (RBF) kernel.
    clusterer = SpectralClustering(n_clusters=60, affinity='rbf',
                                assign_labels='kmeans')
    labels = clusterer.fit_predict(X)

    # Save mapping filename → cluster_id
    with open(os.path.join(emb_dir, "filenames.txt")) as f:
        files = [line.strip() for line in f]
    print(f"[DEBUG] # files: {len(files)}, # labels: {len(labels)}")
    if len(files) != len(labels):
        raise ValueError("Mismatch between # filenames and # cluster labels")
    mapping = dict(zip(files, map(int, labels)))
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
