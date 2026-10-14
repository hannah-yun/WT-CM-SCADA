"""
Step 2: Select one graph threshold for each candidate window size.
Reads the training graphs saved by step 1, embeds them with a graph autoencoder,
and selects the threshold maximizing the sum of symmetric KL distances to a medoid.
Writes best_thresholds.csv and loss tables; it does not select a single final window size.

2_select_wds_ths.py

For each window size, loops over all threshold folders (from 1_save_cosine_graphs_with_sigmoid.py),
trains a GAE on the medoid graph, computes kmedoids loss in GAE-embedded space
(same method as find_typical_graph_kmedoids / Step 3), and selects the best threshold.

Criterion: maximise sum of KL-symmetric distances from all training graphs to their
medoid IN THE GAE-EMBEDDED LATENT SPACE. 


Outputs (saved to --base_dir):
  - best_thresholds.csv             : one row per window_size with the selected threshold
  - all_threshold_losses.csv        : full (ws, thresh, kmedoids_loss, mean_degree) table
  - w{ws}/threshold_losses.csv      : per-window table (threshold, kmedoids_loss, mean_degree)
"""

import os
import sys
import pickle
import argparse

import numpy as np
import tensorflow as tf
#import kmedoids
import networkx as nx
import pandas as pd
from sklearn_extra.cluster import KMedoids

from gae.embedding import GAE
from util import norm_transition, fast_markov_vector_robust
from find_prototype_kmedoids_for2 import find_typical_graph_kmedoids

np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# HELPERS
# ----------------------------

def _to_strict_simplex(v, eps=1e-12, smooth=1e-5):
    v = np.asarray(v, dtype=float)
    v[~np.isfinite(v)] = 0.0
    v = np.clip(v, 0.0, None) + smooth
    return v / np.clip(v.sum(), eps, None)


def _jeffreys(p, q):
    """Jeffreys (symmetric KL) divergence between two simplex vectors."""
    return float(np.sum((p - q) * np.log(p / q)))


# ----------------------------
# CORE FUNCTION
# ----------------------------

def compute_kmedoids_loss_gae(training_graphs, epoch=200, hidden1=64, hidden2=32):
    """
    1. Find medoid via KL-symmetric on original adjacency RW (lightweight, no GAE needed).
    2. Train GAE on the medoid graph.
    3. Call find_typical_graph_kmedoids (n=None) to embed all training graphs
       and compute kmedoids loss in GAE-embedded latent space.
    4. Return (loss, mean_degree).

    Maximising loss across thresholds selects the threshold where training graphs
    are most spread in latent space — same criterion as Step 3.
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    # Step 1: find medoid in original graph space (kl_symmetric on RW vectors)
    vecs = []
    for G in training_graphs:
        adj        = nx.adjacency_matrix(G)
        transition = norm_transition(adj.toarray())
        vec        = fast_markov_vector_robust(transition.T)
        vecs.append(_to_strict_simplex(vec))

    N    = len(vecs)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d          = max(_jeffreys(vecs[i], vecs[j]), 0.0)
            dist[i, j] = d
            dist[j, i] = d

    km = KMedoids(n_clusters=1, init='build', random_state=42)
    km.fit(dist)
    medoid_idx = int(km.medoid_indices_[0])
    medoid_graph = training_graphs[medoid_idx]

    # Step 2: train GAE on medoid
    num_nodes = medoid_graph.number_of_nodes()
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    gae = GAE(num_nodes, epoch=epoch, hidden1=hidden1, hidden2=hidden2)
    sys.argv = original_argv
    gae.fit(medoid_graph)

    # Step 3: embed all training graphs, compute kmedoids loss in latent space
    _, info = find_typical_graph_kmedoids(
        training_graphs, gae, n=None, distance_method='kl_symmetric'
    )
    loss = float(np.sum(info['dist_to_medoid']))

    mean_deg = float(np.mean([
        np.mean([d for _, d in G.degree()])
        for G in training_graphs
    ]))

    return loss, mean_deg


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select best threshold per window size by maximising kmedoids loss "
                    "in GAE-embedded latent space (same method as Step 3)"
    )

    parser.add_argument("--ws_start",  type=int,   default=24)
    parser.add_argument("--ws_end",    type=int,   default=96)
    parser.add_argument("--ws_step",   type=int,   default=1)
    parser.add_argument("--thr_start", type=float, default=0.60)
    parser.add_argument("--thr_end",   type=float, default=0.95)
    parser.add_argument("--thr_step",  type=float, default=0.005)
    parser.add_argument("--base_dir",  type=str,   default="cosine_graphs")
    parser.add_argument("--epoch",     type=int,   default=200)
    parser.add_argument("--hidden1",   type=int,   default=64)
    parser.add_argument("--hidden2",   type=int,   default=32)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip window sizes whose threshold_losses.csv already exists")

    args = parser.parse_args()

    window_sizes = range(args.ws_start, args.ws_end + 1, args.ws_step)
    n_thr        = round((args.thr_end - args.thr_start) / args.thr_step) + 1
    thresholds   = [round(args.thr_start + i * args.thr_step, 4) for i in range(n_thr)]

    best_rows = []   # one row per window_size
    all_rows  = []   # one row per (window_size, threshold)

    print(f"Threshold selection (GAE-embedded): {len(window_sizes)} window sizes x {len(thresholds)} thresholds")
    print(f"Criterion: maximise kmedoids loss in GAE latent space (kl_symmetric)")
    print(f"GAE: epoch={args.epoch}, hidden1={args.hidden1}, hidden2={args.hidden2}")
    print(f"Base dir: {args.base_dir}/\n")

    for ws in window_sizes:
        print(f"--- window_size={ws} ---")
        ws_csv_path = os.path.join(args.base_dir, f"w{ws}", "threshold_losses.csv")

        if args.skip_existing and os.path.exists(ws_csv_path):
            print(f"  skip (threshold_losses.csv exists)")
            ws_df = pd.read_csv(ws_csv_path)
            for _, r in ws_df.iterrows():
                all_rows.append({"window_size": ws, "threshold": r["threshold"],
                                  "kmedoids_loss": r["kmedoids_loss"], "mean_degree": r["mean_degree"]})
            valid = ws_df.dropna(subset=["kmedoids_loss"])
            if not valid.empty:
                best_idx    = valid["kmedoids_loss"].idxmax()
                best_thresh = float(valid.loc[best_idx, "threshold"])
                best_loss   = float(valid.loc[best_idx, "kmedoids_loss"])
                print(f"  => best threshold: t{best_thresh:g}  (kmedoids_loss={best_loss:.4f})")
                best_rows.append({"window_size": ws, "best_threshold": best_thresh,
                                   "kmedoids_loss": best_loss})
            else:
                print(f"  => no valid threshold found for w{ws}")
            continue

        best_thresh = None
        best_loss   = -np.inf

        for thresh in thresholds:
            thresh_label = f"t{thresh:g}"
            train_path   = os.path.join(args.base_dir, f"w{ws}", thresh_label, "train", "graphs.pkl")

            if not os.path.exists(train_path):
                print(f"  {thresh_label}: missing — skip")
                all_rows.append({"window_size": ws, "threshold": thresh,
                                  "kmedoids_loss": None, "mean_degree": None})
                continue

            with open(train_path, "rb") as f:
                training_graphs = pickle.load(f)

            try:
                loss, mean_deg = compute_kmedoids_loss_gae(
                    training_graphs,
                    epoch=args.epoch, hidden1=args.hidden1, hidden2=args.hidden2,
                )
                all_rows.append({"window_size": ws, "threshold": thresh,
                                  "kmedoids_loss": loss, "mean_degree": mean_deg})
                print(f"  {thresh_label}: kmedoids_loss={loss:.4f}  mean_degree={mean_deg:.2f}")

                if loss > best_loss:
                    best_loss   = loss
                    best_thresh = thresh

            except Exception as e:
                print(f"  {thresh_label}: ERROR — {e}")
                all_rows.append({"window_size": ws, "threshold": thresh,
                                  "kmedoids_loss": None, "mean_degree": None})

        # Save per-window CSV: {base_dir}/w{ws}/threshold_losses.csv
        os.makedirs(os.path.dirname(ws_csv_path), exist_ok=True)
        pd.DataFrame(
            [{"threshold": r["threshold"], "kmedoids_loss": r["kmedoids_loss"],
              "mean_degree": r["mean_degree"]}
             for r in all_rows if r["window_size"] == ws]
        ).to_csv(ws_csv_path, index=False)
        print(f"  Saved -> {ws_csv_path}")

        if best_thresh is not None:
            print(f"  => best threshold: t{best_thresh:g}  (kmedoids_loss={best_loss:.4f})")
            best_rows.append({
                "window_size":    ws,
                "best_threshold": best_thresh,
                "kmedoids_loss":  best_loss,
            })
        else:
            print(f"  => no valid threshold found for w{ws}")

    # Save outputs
    best_path = os.path.join(args.base_dir, "best_thresholds.csv")
    all_path  = os.path.join(args.base_dir, "all_threshold_losses.csv")

    pd.DataFrame(best_rows).to_csv(best_path, index=False)
    pd.DataFrame(all_rows).to_csv(all_path,  index=False)

    print(f"\nBest thresholds saved -> {best_path}")
    print(f"All losses saved      -> {all_path}")
    print(f"\nDone: {len(best_rows)}/{len(window_sizes)} window sizes processed.")
