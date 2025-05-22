"""Silhouette score, Davies‑Bouldin Index, and size histogram."""
import os, argparse, json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

def main(emb_dir, map_json):
    X = np.load(os.path.join(emb_dir, "embeddings.npy"))
    labels = np.array(list(json.load(open(map_json)).values()))
    mask = labels != -1                     # exclude outliers for metrics
    X = normalize(X[mask])
    y = labels[mask]
    print(labels)
    if len(set(y)) < 2:
        print("Need ≥2 clusters for metrics.")
        return
    sil = silhouette_score(X, y, metric="euclidean")
    dbi = davies_bouldin_score(X, y)
    cal = calinski_harabasz_score(X, y)
    print(f"Silhouette = {sil:.3f}  |  Davies‑Bouldin = {dbi:.3f} | Calinski-Harabasz = {cal:.3f}")

    # Histogram plot
    uniq, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.bar(range(len(counts)), counts, edgecolor="k")
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster #")
    plt.ylabel("# recordings")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/embeddings")
    ap.add_argument("--map_json", default="data/embeddings/cluster_map.json")
    args = ap.parse_args()
    main(args.emb_dir, args.map_json)
