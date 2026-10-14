"""
Prototype selection helper used by step 2.
Converts graph embeddings to stationary distributions and selects representative
training graphs with k-medoids using a configurable pairwise distance.

find_prototype_kmedoids_for2.py

Selects the most representative graphs from a set of training graphs using:
  1. GAE embedding → random walk stationary distribution (per graph)
  2. Pairwise kl_symmetric (Jeffreys) distance matrix
  3. k-medoids with k=1 to find the single medoid graph with improved initialisation
  4. Select N graphs with shortest distance to the medoid

Usage:
    from find_prototype_kmedoids_for2 import find_typical_graph_kmedoids

    prototype_graphs, info = find_typical_graph_kmedoids(
        training_graphs, gae, n=72
    )
"""

from unittest import result
#import tensorflow as tf
import numpy as np
#import kmedoids
from util import inner_transition, fast_markov_vector_robust, ReLu


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------

def _to_strict_simplex(X, eps=1e-12, smooth=1e-5):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    X = np.clip(X, 0.0, None)
    X = X + smooth
    s = X.sum(axis=1, keepdims=True)
    return X / np.clip(s, eps, None)


def _random_walk_in_latent_space(emb, active=ReLu):
    transition = inner_transition(emb, active)
    return fast_markov_vector_robust(transition.T)


def _jeffreys_vec(p, q):
    """Jeffreys (symmetric KL) divergence between two probability vectors.

    Equivalent to KL(p||q) + KL(q||p) = sum((p - q) * log(p / q)).
    Requires p, q > 0 everywhere (guaranteed by _to_strict_simplex).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sum((p - q) * np.log(p / q)))


# ------------------------------------------------------
# Pairwise distance matrix (kl_symmetric / l2 / cosine)
# ------------------------------------------------------

def _pairwise_kl_symmetric(vecs_simplex, distance_method):
    """Build a symmetric N×N distance matrix."""
    N = len(vecs_simplex)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            if distance_method == 'kl_symmetric':
                d = _jeffreys_vec(vecs_simplex[i], vecs_simplex[j])
            elif distance_method == 'l2':
                d = float(np.linalg.norm(vecs_simplex[i] - vecs_simplex[j]))
            elif distance_method == 'cosine':
                vi, vj = vecs_simplex[i], vecs_simplex[j]
                denom = np.linalg.norm(vi) * np.linalg.norm(vj)
                d = 1.0 - float(np.dot(vi, vj) / denom) if denom > 1e-12 else 1.0
            else:
                raise ValueError(f"Unsupported distance_method: {distance_method!r}. "
                                 f"Choose from 'kl_symmetric', 'l2', 'cosine'.")

            d = max(d, 0.0)  # numerical safety
            D[i, j] = d
            D[j, i] = d

    return D


# ------------------------------------------------------
# Main function
# ------------------------------------------------------

def find_typical_graph_kmedoids(training_graphs, gae, n=None,
                                distance_method='kl_symmetric'):
    """
    Select the N most representative graphs from training_graphs.

    Steps
    -----
    1. Embed every training graph via GAE.
    2. Compute random walk stationary distribution in latent space.
    3. Convert to probability simplex.
    4. Build a symmetric N×N pairwise distance matrix (kl_symmetric by default).
    5. Run k-medoids with k=1 to find the single medoid graph.
    6. Rank all graphs by distance to the medoid; return the top-n.

    Parameters
    ----------
    training_graphs : list[nx.Graph]
    gae : GAE
        A trained Graph Autoencoder instance.
    n : int or None
        Number of graphs to return (closest to medoid).
        If None, returns all graphs ranked by distance to medoid.
    distance_method : str
        'kl_symmetric' — Jeffreys divergence directly between simplex vectors (default).
        'l2'           — Euclidean distance between simplex vectors.
        'cosine'       — cosine distance between simplex vectors.

    Returns
    -------
    typical_graphs : list[nx.Graph]
        Top-n graphs closest to the medoid (medoid is always first).
    info : dict
        'medoid_index'     — index of medoid in training_graphs.
        'selected_indices' — indices of returned graphs (medoid first).
        'dist_to_medoid'   — distances from medoid to every training graph.
        'distance_matrix'  — full N×N pairwise distance array.
        'vecs_simplex'     — simplex vector per training graph.
        'embeddings'       — raw GAE embeddings per training graph (reuse to avoid recomputing).
        'distance_method'  — metric used.
    """
    N = len(training_graphs)
    n = N if n is None else min(n, N)

    # Step 1: embed all training graphs
    print("  Embedding training graphs...")
    embeddings = []
    for i, G in enumerate(training_graphs):
        embeddings.append(gae.embed(G))
        if (i + 1) % 100 == 0:
            print(f"  Embedded {i + 1}/{N}")

    # Step 2: random walk distributions in latent space
    print("  Computing random walk distributions...")
    vecs = [_random_walk_in_latent_space(emb) for emb in embeddings]
    vecs_simplex = [_to_strict_simplex(v.reshape(1, -1))[0] for v in vecs]

    # Step 3: symmetric pairwise distance matrix
    print(f"  Building {N}×{N} pairwise distance matrix (distance_method={distance_method})...")
    dist_matrix = _pairwise_kl_symmetric(vecs_simplex, distance_method)
    print("  Distance matrix built.")

    # Step 5: k=1 medoid — point with minimum total distance to all others
    print("  Finding medoid (k=1)...")
    medoid_idx = int(np.argmin(dist_matrix.sum(axis=1)))
    print(f"  Medoid found: index {medoid_idx}")

    # Step 6: rank by distance to medoid, take top-n (medoid is always rank 0)
    dist_to_medoid = dist_matrix[medoid_idx].copy()
    dist_to_medoid[medoid_idx] = 0.0
    ranked = np.argsort(dist_to_medoid)   # ascending: closest first
    selected_indices = ranked[:n].tolist()  # n-1 most similar
    #selected_indices = [medoid_idx] + selected_indices

    print(f"  Selected {n} graphs (medoid + {n - 1} most closest).")

    return [training_graphs[i] for i in selected_indices], {
        "medoid_index": medoid_idx,
        "selected_indices": selected_indices,
        "dist_to_medoid": dist_to_medoid.tolist(),
        "distance_matrix": dist_matrix,
        "vecs_simplex": vecs_simplex,
        "embeddings": embeddings,          # list[np.array], indexed by training graph position
        "distance_method": distance_method,
    }
