import umap
import numpy as np


def project_umap(features, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Project high-dimensional features into low-dimensional space using UMAP.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )

    embedding = reducer.fit_transform(features)
    return embedding
