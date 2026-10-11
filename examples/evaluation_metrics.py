"""Representation evaluation metrics used by ACE examples.

These metrics are adapted from the mouse droplet analysis workflow. They measure
whether the salient latent space is age-ordered while mixing non-age cell traits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import entropy, pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors


def _corr_stat(res):
    return getattr(res, "statistic", getattr(res, "correlation", res[0] if isinstance(res, tuple) else np.nan))


def _corr_pvalue(res):
    return getattr(res, "pvalue", res[1] if isinstance(res, tuple) and len(res) > 1 else np.nan)


def _sample_cross_age_pairs(age, n_pairs=5000, seed=0):
    age = np.asarray(age, dtype=float)
    if age.size < 2 or np.unique(age).size < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    sampled_i = []
    sampled_j = []
    total_sampled = 0
    batch_size = max(n_pairs * 2, 10000)
    while total_sampled < n_pairs:
        i = rng.integers(0, len(age), size=batch_size)
        j = rng.integers(0, len(age), size=batch_size)
        mask = (i != j) & (~np.isclose(age[i], age[j]))
        valid_i = i[mask]
        valid_j = j[mask]
        sampled_i.append(valid_i)
        sampled_j.append(valid_j)
        total_sampled += valid_i.size
    return np.concatenate(sampled_i)[:n_pairs], np.concatenate(sampled_j)[:n_pairs]


def distance_age_correlation(latent, age, n_pairs=5000, seed=0):
    i, j = _sample_cross_age_pairs(age, n_pairs=n_pairs, seed=seed)
    if i.size == 0:
        return {
            "distance_age_n_pairs": 0,
            "distance_age_spearman_r": np.nan,
            "distance_age_spearman_p": np.nan,
            "distance_age_pearson_r": np.nan,
            "distance_age_pearson_p": np.nan,
        }
    dist = np.linalg.norm(latent[i] - latent[j], axis=1)
    age_diff = np.abs(age[i] - age[j])
    spearman_res = spearmanr(dist, age_diff)
    pearson_res = pearsonr(dist, age_diff)
    return {
        "distance_age_n_pairs": len(i),
        "distance_age_spearman_r": _corr_stat(spearman_res),
        "distance_age_spearman_p": _corr_pvalue(spearman_res),
        "distance_age_pearson_r": _corr_stat(pearson_res),
        "distance_age_pearson_p": _corr_pvalue(pearson_res),
    }


def age_bin_centroid_monotonicity(latent, age):
    age = np.asarray(age, dtype=float)
    centroids = []
    valid_ages = []
    for age_bin in np.sort(np.unique(age)):
        mask = np.isclose(age, age_bin)
        if np.any(mask):
            centroids.append(latent[mask].mean(axis=0))
            valid_ages.append(age_bin)
    if len(centroids) < 2:
        return {"age_bin_centroid_monotonicity_spearman_r": np.nan}
    centroids = np.vstack(centroids)
    centroid_distances = []
    age_bin_differences = []
    for i in range(len(valid_ages)):
        for j in range(i + 1, len(valid_ages)):
            centroid_distances.append(np.linalg.norm(centroids[i] - centroids[j]))
            age_bin_differences.append(abs(valid_ages[j] - valid_ages[i]))
    res = spearmanr(centroid_distances, age_bin_differences)
    return {"age_bin_centroid_monotonicity_spearman_r": _corr_stat(res)}


def _build_weighted_knn_graph(latent, n_neighbors=30):
    n_cells = latent.shape[0]
    if n_cells < 2:
        return csr_matrix((n_cells, n_cells)), np.empty((0, 2), dtype=int)
    n_neighbors = min(n_neighbors, n_cells - 1)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(latent)
    knn_distances, knn_indices = neighbors.kneighbors(latent)
    knn_distances = knn_distances[:, 1:]
    knn_indices = knn_indices[:, 1:]
    edge_to_weight = {}
    for source_idx in range(n_cells):
        for target_idx, distance in zip(knn_indices[source_idx], knn_distances[source_idx]):
            edge = tuple(sorted((source_idx, int(target_idx))))
            prev_distance = edge_to_weight.get(edge)
            if prev_distance is None or float(distance) < prev_distance:
                edge_to_weight[edge] = float(distance)
    if not edge_to_weight:
        return csr_matrix((n_cells, n_cells)), np.empty((0, 2), dtype=int)
    edges = np.array(list(edge_to_weight.keys()), dtype=int)
    weights = np.array(list(edge_to_weight.values()), dtype=float)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = csr_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n_cells, n_cells))
    return graph, edges


def _knn_connectivity_graph(latent, n_neighbors=30):
    n_cells = latent.shape[0]
    if n_cells < 2:
        return csr_matrix((n_cells, n_cells))
    n_neighbors = min(n_neighbors, n_cells - 1)
    indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent).kneighbors(
        latent, return_distance=False
    )[:, 1:]
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.ravel()
    data = np.ones(rows.shape[0], dtype=float)
    graph = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    graph = graph.maximum(graph.T)
    return graph


def morans_i(connectivities, values):
    values = np.asarray(values, dtype=float)
    z = values - values.mean()
    denom = np.sum(z**2)
    s0 = connectivities.sum()
    if denom == 0 or s0 == 0:
        return np.nan
    return float((len(values) / s0) * (z @ connectivities @ z) / denom)


def gearys_c(connectivities, values):
    values = np.asarray(values, dtype=float)
    z = values - values.mean()
    denom = np.sum(z**2)
    s0 = connectivities.sum()
    if denom == 0 or s0 == 0:
        return np.nan
    rows, cols = connectivities.nonzero()
    weights = np.asarray(connectivities[rows, cols]).ravel()
    numerator = np.sum(weights * (values[rows] - values[cols]) ** 2)
    return float(((len(values) - 1) / (2 * s0)) * numerator / denom)


def geodesic_distance_age_correlation(latent, age, n_neighbors=30, n_pairs=1000, seed=0, graph=None):
    age = np.asarray(age, dtype=float)
    if latent.shape[0] < 2 or np.unique(age).size < 2:
        return {"geodesic_distance_age_spearman_r": np.nan}
    if graph is None:
        graph, _ = _build_weighted_knn_graph(latent, n_neighbors=n_neighbors)
    i, j = _sample_cross_age_pairs(age, n_pairs=n_pairs, seed=seed)
    if i.size == 0:
        return {"geodesic_distance_age_spearman_r": np.nan}
    unique_sources, inverse = np.unique(i, return_inverse=True)
    path_lengths = shortest_path(graph, directed=False, indices=unique_sources)
    geodesic_distances = path_lengths[inverse, j]
    valid_mask = np.isfinite(geodesic_distances)
    if valid_mask.sum() < min(n_pairs, max(100, n_pairs // 10)):
        return {"geodesic_distance_age_spearman_r": np.nan}
    age_diff = np.abs(age[i[valid_mask]] - age[j[valid_mask]])
    res = spearmanr(geodesic_distances[valid_mask], age_diff)
    return {"geodesic_distance_age_spearman_r": _corr_stat(res)}


def adjacent_age_connectivity(latent, age, n_neighbors=30, edges=None):
    age = np.asarray(age, dtype=float)
    if latent.shape[0] < 2:
        return {"adjacent_age_connectivity_ratio": np.nan, "adjacent_age_connectivity_ratio_all": np.nan}
    age_order = np.sort(np.unique(age))
    age_to_index = {float(age_bin): idx for idx, age_bin in enumerate(age_order)}
    age_indices = np.array([age_to_index[float(age_bin)] for age_bin in age], dtype=int)
    if edges is None:
        _, edges = _build_weighted_knn_graph(latent, n_neighbors=n_neighbors)
    if edges.shape[0] == 0:
        return {"adjacent_age_connectivity_ratio": np.nan, "adjacent_age_connectivity_ratio_all": np.nan}
    age_differences = np.abs(age_indices[edges[:, 0]] - age_indices[edges[:, 1]])
    cross_age_mask = age_differences > 0
    n_cross_age_edges = np.sum(cross_age_mask)
    if n_cross_age_edges == 0:
        return {"adjacent_age_connectivity_ratio": np.nan, "adjacent_age_connectivity_ratio_all": np.nan}
    n_adjacent_age_edges = np.sum(age_differences == 1)
    return {
        "adjacent_age_connectivity_ratio": n_adjacent_age_edges / n_cross_age_edges,
        "adjacent_age_connectivity_ratio_all": n_adjacent_age_edges / len(age_differences),
    }


def entropy_batch_mixing(latent, labels, n_neighbors=30, n_pools=20, n_samples_per_pool=50, seed=0):
    labels = pd.Series(labels).astype("category").cat.codes.to_numpy()
    n_neighbors = min(n_neighbors, latent.shape[0] - 1)
    if n_neighbors < 1:
        return np.nan
    rng = np.random.default_rng(seed)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
    neighbor_labels = labels[indices]
    entropies = np.apply_along_axis(
        lambda x: entropy(np.unique(x, return_counts=True)[1].astype(np.int32)),
        axis=1,
        arr=neighbor_labels,
    )
    if n_pools == 1:
        return float(np.mean(entropies))
    pool_size = min(n_samples_per_pool, len(entropies))
    return float(np.mean([np.mean(entropies[rng.choice(len(entropies), size=pool_size)]) for _ in range(n_pools)]))


def evaluate_salient_representation(
    adata_embed,
    age_key="age_float",
    trait_keys=("cell_type", "tissue", "sex"),
    n_neighbors=30,
    n_pairs=5000,
    seed=0,
):
    adata = adata_embed.copy()
    latent = np.asarray(adata.X)
    age = adata.obs[age_key].astype(float).to_numpy()
    n_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))
    graph, edges = _build_weighted_knn_graph(latent, n_neighbors=n_neighbors)
    connectivities = _knn_connectivity_graph(latent, n_neighbors=n_neighbors)

    metrics = {
        "gearys_c": gearys_c(connectivities, age),
        "morans_i": morans_i(connectivities, age),
    }
    metrics.update(distance_age_correlation(latent, age, n_pairs=n_pairs, seed=seed))
    metrics.update(age_bin_centroid_monotonicity(latent, age))
    metrics.update(
        geodesic_distance_age_correlation(
            latent, age, n_neighbors=n_neighbors, n_pairs=max(500, n_pairs // 5), seed=seed, graph=graph
        )
    )
    metrics.update(adjacent_age_connectivity(latent, age, n_neighbors=n_neighbors, edges=edges))
    for key in trait_keys:
        if key in adata.obs:
            metrics[f"{key}_AllAge_entropy_batch_mixing"] = entropy_batch_mixing(
                latent, adata.obs[key], n_neighbors=n_neighbors, seed=seed
            )
    return metrics


def save_metrics(metrics: dict[str, float], path):
    pd.Series(metrics, name="value").to_frame().to_csv(path)
