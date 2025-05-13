import hdbscan, numpy as np
X = np.load("embeddings.npy")
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
labels = clusterer.fit_predict(X)  