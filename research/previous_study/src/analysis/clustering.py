import hdbscan
import numpy as np


def cluster_hdbscan(embedding, min_cluster_size=15):
    """
    Cluster low-dimensional embeddings using HDBSCAN.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size
    )

    labels = clusterer.fit_predict(embedding)
    return labels
