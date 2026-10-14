"""
Step 3: Score and evaluate graphs for a supplied window size and graph threshold.

Loads the training and test graphs saved in step 1. Choose a window size and its
threshold from the results of step 2. The script creates graph-autoencoder and
baseline representations, then tries collection sizes h from --h_start to
--h_end in increments of --h_step.

For each h, it groups consecutive training graphs into collections of h graphs:
    n_collections = floor(N_train / h)
The score_worker_old module calculates KL and SMDL anomaly scores for each
collection and takes their median. Different h values run in parallel.

To evaluate the scores, provide a labels CSV containing Timestamp_clean and
health columns. Use --eval_start to set where evaluation begins. The script
saves the results and plots for the best collection size for each metric.

Outputs under --out_dir:
  h{h}/distance_median_all.csv : anomaly scores for each collection size
  h{h}/metrics.txt             : evaluation metrics for each collection size
  summary.csv                 : results across h values and four score metrics
  best_by_roc_auc.csv          : best results ranked by ROC AUC
  best_by_f1.csv               : best results ranked by F1
  best_by_pr_auc.csv           : best results ranked by PR AUC
  best_h{h}_{metric}_results/  : plots and metrics for the best h by ROC AUC
  timing.csv                  : runtime measurements
"""

import os
import sys
import pickle
import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    auc,
)

from gae.embedding import GAE
from util import (
    norm_transition,
    fast_markov_vector_robust,
    inner_transition,
    ReLu,
)

from pa_rpa_metrics import compute_pa_rpa_metrics
from score_worker_old import score_one_h


METRICS = ["kl_embedding", "kl_baseline", "smdl_embedding", "smdl_baseline"]

DISTANCE_TYPES = ["median_all"]

REPORT_ROWS = [
    'window_size', 'h', 'threshold', 'n_collections', 'distance_type',
    'roc_auc', 'pr_auc', 'avg_precision',
    'f1', 'precision', 'recall', 'specificity', 'accuracy',
    'pw_precision', 'pw_recall', 'pw_f1',
    'pa_precision', 'pa_recall', 'pa_f1',
    'rpa_precision', 'rpa_recall', 'rpa_f1',
    'opt_threshold',
]

DISTANCE_CSV_NAME = {
    "median_all": "distance_median_all.csv",
}

TIMING_PHASE_METRIC = [
    ('info',  'seed'),
    ('train', 'train_time_s'),
    ('train', 'train_emb_time_s'),
    ('train', 'train_base_time_s'),
    ('test',  'test_embed_time_s'),
    ('test',  'test_embed_emb_time_s'),
    ('test',  'test_embed_base_time_s'),
    ('test',  'scoring_time_s'),
    ('test',  'h20_scoring_time_s'),
    ('test',  'kl_embedding_scoring_time_s'),
    ('test',  'kl_baseline_scoring_time_s'),
    ('test',  'smdl_embedding_scoring_time_s'),
    ('test',  'smdl_baseline_scoring_time_s'),
    ('test',  'kl_embedding_test_time_s'),
    ('test',  'kl_baseline_test_time_s'),
    ('test',  'smdl_embedding_test_time_s'),
    ('test',  'smdl_baseline_test_time_s'),
    ('test',  'total_test_time_s'),
    ('total', 'total_time_s'),
]


# ----------------------------
# HELPERS
# ----------------------------

def to_strict_simplex(X, eps=1e-12, smooth=1e-3):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    X = np.clip(X, 0.0, None)
    X = X + smooth
    s = X.sum(axis=1, keepdims=True)
    return X / np.clip(s, eps, None)


def random_walk_in_latent_space(emb):
    transition = inner_transition(emb, ReLu)
    return fast_markov_vector_robust(transition.T)


def random_walk_in_network(adj):
    transition = norm_transition(adj.toarray())
    return fast_markov_vector_robust(transition.T)


# ----------------------------
# EMBED ONCE (shared across all h values)
# ----------------------------

def embed_all(training_graphs, testing_graphs, epoch, hidden1, hidden2, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    t_train = time.time()

    num_nodes = training_graphs[0].number_of_nodes()
    if not all(G.number_of_nodes() == num_nodes for G in training_graphs + testing_graphs):
        raise ValueError("All graphs must have the same number of nodes.")

    N = len(training_graphs)

    # Train GAE on a randomly selected training graph

    random_idx = np.random.randint(N)
    print(f"  Training GAE on random graph index {random_idx}...")

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    gae = GAE(num_nodes, epoch=epoch, hidden1=hidden1, hidden2=hidden2, seed=seed)
    sys.argv = original_argv
    gae.fit(training_graphs[random_idx])
    print("  GAE training complete.")

    # Embed all training graphs — embedding path (GAE-based)
    t_train_emb = time.time()
    all_train_vecs_simplex = []
    for G in training_graphs:
        emb = gae.embed(G)
        vec = random_walk_in_latent_space(emb)
        all_train_vecs_simplex.append(to_strict_simplex(vec.reshape(1, -1))[0])
    train_emb_time = time.time() - t_train_emb

    # Baseline path (network random walk, no GAE)
    t_train_base = time.time()
    all_train_vecs_org_simplex = []
    for G in training_graphs:
        adj = nx.adjacency_matrix(G)
        vec_org = random_walk_in_network(adj)
        all_train_vecs_org_simplex.append(to_strict_simplex(vec_org.reshape(1, -1))[0])
    train_base_time = time.time() - t_train_base

    train_time = time.time() - t_train
    print(f"  Train embeddings done. ({train_time:.1f}s  "
          f"emb={train_emb_time:.1f}s  base={train_base_time:.1f}s)")

    # Embed test graphs — time GAE path and org path separately
    print("  Embedding test graphs...")
    test_vecs_simplex = []
    test_vecs_org_simplex = []
    test_embed_emb_time = 0.0
    test_embed_base_time = 0.0
    for i, G in enumerate(testing_graphs):
        _t = time.time()
        emb = gae.embed(G)
        vec = random_walk_in_latent_space(emb)
        test_vecs_simplex.append(to_strict_simplex(vec.reshape(1, -1))[0])
        test_embed_emb_time += time.time() - _t

        _t = time.time()
        adj = nx.adjacency_matrix(G)
        vec_org = random_walk_in_network(adj)
        test_vecs_org_simplex.append(to_strict_simplex(vec_org.reshape(1, -1))[0])
        test_embed_base_time += time.time() - _t

        if (i + 1) % 50 == 0:
            print(f"    Embedded {i + 1}/{len(testing_graphs)} test graphs")

    test_time = test_embed_emb_time + test_embed_base_time
    print(f"  Test embeddings done. ({test_time:.1f}s  "
          f"emb={test_embed_emb_time:.1f}s  base={test_embed_base_time:.1f}s)")

    return {
        'all_train_vecs_simplex':     all_train_vecs_simplex,
        'all_train_vecs_org_simplex': all_train_vecs_org_simplex,
        'test_vecs_simplex':          test_vecs_simplex,
        'test_vecs_org_simplex':      test_vecs_org_simplex,
        'num_nodes':                  num_nodes,
        'N_train':                    N,
        'train_time':                 train_time,
        'train_emb_time':             train_emb_time,
        'train_base_time':            train_base_time,
        'test_time':                  test_time,
        'test_embed_emb_time':        test_embed_emb_time,
        'test_embed_base_time':       test_embed_base_time,
    }


# ----------------------------
# COMPUTE METRICS
# ----------------------------

def compute_metrics(distance_csv, labels_csv, window_size, threshold, h,
                    n_collections, distance_type, distance_metric, eval_start):
    distance_df = pd.read_csv(distance_csv)
    labels_df   = pd.read_csv(labels_csv)
    labels_df['timestamp'] = pd.to_datetime(labels_df['Timestamp_clean'], format='mixed')

    min_length          = min(len(distance_df), len(labels_df) - window_size)
    distance_eval_start = eval_start - window_size
    if min_length <= 0 or distance_eval_start >= min_length or distance_eval_start < 0:
        return None

    distances   = distance_df[distance_metric].values[distance_eval_start:min_length]
    true_labels = labels_df['health'].values[eval_start:window_size + min_length]

    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(true_labels, distances)
    f1_scores   = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    opt_thresh  = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else thresholds_pr[-1]
    predictions = (distances >= opt_thresh).astype(int)

    roc_auc  = roc_auc_score(true_labels, distances)
    pr_auc   = auc(recall_arr, precision_arr)
    avg_prec = average_precision_score(true_labels, distances)
    accuracy = accuracy_score(true_labels, predictions)
    prec     = precision_score(true_labels, predictions, zero_division=0)
    rec      = recall_score(true_labels, predictions, zero_division=0)
    f1       = f1_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pa_rpa   = compute_pa_rpa_metrics(true_labels, predictions)

    return {
        'window_size':     window_size,
        'h':               h,
        'n_collections':   n_collections,
        'threshold':       threshold,
        'distance_type':   distance_type,
        'distance_metric': distance_metric,
        'roc_auc':         roc_auc,
        'pr_auc':          pr_auc,
        'avg_precision':   avg_prec,
        'f1':              f1,
        'precision':       prec,
        'recall':          rec,
        'specificity':     specificity,
        'accuracy':        accuracy,
        'opt_threshold':   opt_thresh,
        **pa_rpa,
    }


# ----------------------------
# SAVE METRICS TXT PER h
# ----------------------------

def save_h_metrics_txt(h_results, h, n_collections, window_size, threshold, out_dir):
    path = os.path.join(out_dir, f"h{h}", "metrics.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"h={h}  |  n_collections={n_collections}  |  "
                f"window_size={window_size}  |  threshold={threshold}\n")
        f.write("=" * 60 + "\n")
        for dt in DISTANCE_TYPES:
            f.write(f"\n[distance_type: {dt}]\n")
            for r in h_results:
                if r['distance_type'] != dt:
                    continue
                f.write(f"  {r['distance_metric']}:\n")
                f.write(f"    ROC AUC:       {r['roc_auc']:.10f}\n")
                f.write(f"    F1:            {r['f1']:.10f}\n")
                f.write(f"    PR AUC:        {r['pr_auc']:.10f}\n")
                f.write(f"    Avg Precision: {r['avg_precision']:.10f}\n")
                f.write(f"    Precision:     {r['precision']:.10f}\n")
                f.write(f"    Recall:        {r['recall']:.10f}\n")
                f.write(f"    Specificity:   {r['specificity']:.10f}\n")
                f.write(f"    Opt threshold: {r['opt_threshold']:.6f}\n")
                f.write(f"    PW  — P:{r['pw_precision']:.10f}  R:{r['pw_recall']:.10f}  F1:{r['pw_f1']:.10f}\n")
                f.write(f"    PA  — P:{r['pa_precision']:.10f}  R:{r['pa_recall']:.10f}  F1:{r['pa_f1']:.10f}\n")
                f.write(f"    RPA — P:{r['rpa_precision']:.10f}  R:{r['rpa_recall']:.10f}  F1:{r['rpa_f1']:.10f}\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"  Saved metrics -> {path}")


# ----------------------------
# SAVE BEST RESULTS CSV
# ----------------------------

def save_best_results_csv(summary_df, out_dir):
    for criterion in ('roc_auc', 'f1', 'pr_auc'):
        records = {}
        for metric in METRICS:
            sub = summary_df[summary_df['distance_metric'] == metric]
            if sub.empty:
                continue
            best = sub.loc[sub[criterion].idxmax()]
            records[metric] = {row: best.get(row, None) for row in REPORT_ROWS}
        if not records:
            continue
        df = pd.DataFrame(records)
        df.index.name = 'metric'
        path = os.path.join(out_dir, f"best_by_{criterion}.csv")
        df.to_csv(path)
        print(f"Saved -> {path}")


# ----------------------------
# PLOT BEST h
# ----------------------------

def plot_best(distance_csv, labels_csv, window_size, threshold, h, n_collections,
              distance_type, distance_metric, eval_start, out_dir):
    distance_df = pd.read_csv(distance_csv)
    labels_df   = pd.read_csv(labels_csv)
    labels_df['timestamp'] = pd.to_datetime(labels_df['Timestamp_clean'], format='mixed')

    min_length          = min(len(distance_df), len(labels_df) - window_size)
    distance_eval_start = eval_start - window_size
    distances   = distance_df[distance_metric].values[distance_eval_start:min_length]
    true_labels = labels_df['health'].values[eval_start:window_size + min_length]
    timestamps  = labels_df['timestamp'].values[eval_start:window_size + min_length]

    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(true_labels, distances)
    f1_scores   = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    opt_thresh  = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else thresholds_pr[-1]
    predictions = (distances >= opt_thresh).astype(int)

    roc_auc  = roc_auc_score(true_labels, distances)
    pr_auc   = auc(recall_arr, precision_arr)
    f1       = f1_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pa_rpa   = compute_pa_rpa_metrics(true_labels, predictions)

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"plot_{distance_metric}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Evaluation Metrics at Optimal Threshold\n")
        f.write("=" * 50 + "\n")
        f.write(f"Window size:       {window_size}\n")
        f.write(f"Threshold:         {threshold}\n")
        f.write(f"h:                 {h}\n")
        f.write(f"N collections:     {n_collections}\n")
        f.write(f"Distance type:     {distance_type}\n")
        f.write(f"Distance Metric:   {distance_metric}\n")
        f.write(f"Optimal Threshold: {opt_thresh:.6f}\n\n")
        f.write(f"ROC AUC:           {roc_auc:.10f}\n")
        f.write(f"PR AUC:            {pr_auc:.10f}\n")
        f.write(f"F1 Score:          {f1:.10f}\n")
        f.write(f"Specificity:       {specificity:.10f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN={tn}  FP={fp}\n")
        f.write(f"  FN={fn}  TP={tp}\n\n")
        f.write(f"Point-Adjusted (PA):\n")
        f.write(f"  Precision: {pa_rpa['pa_precision']:.10f}  "
                f"Recall: {pa_rpa['pa_recall']:.10f}  F1: {pa_rpa['pa_f1']:.10f}\n")
        f.write(f"Revised Point-Adjusted (RPA):\n")
        f.write(f"  Precision: {pa_rpa['rpa_precision']:.10f}  "
                f"Recall: {pa_rpa['rpa_recall']:.10f}  F1: {pa_rpa['rpa_f1']:.10f}\n")
        f.write(f"Pointwise (PW):\n")
        f.write(f"  Precision: {pa_rpa['pw_precision']:.10f}  "
                f"Recall: {pa_rpa['pw_recall']:.10f}  F1: {pa_rpa['pw_f1']:.10f}\n")
        f.write("=" * 50 + "\n")

    time_data = pd.to_datetime(timestamps)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1.plot(time_data, distances, 'b-', linewidth=1.5,
             label=distance_metric.replace('_', ' ').title())
    ax1.axhline(y=opt_thresh, color='g', linestyle='--', linewidth=2,
                label=f'Optimal Threshold ({opt_thresh:.10f})')

    anomaly_indices = np.where(true_labels == 1)[0]
    if len(anomaly_indices) > 0:
        regions, s = [], anomaly_indices[0]
        for k in range(1, len(anomaly_indices)):
            if anomaly_indices[k] != anomaly_indices[k - 1] + 1:
                regions.append((s, anomaly_indices[k - 1]))
                s = anomaly_indices[k]
        regions.append((s, anomaly_indices[-1]))
        for start_i, end_i in regions:
            ax1.axvspan(time_data[start_i], time_data[end_i], alpha=0.3, color='red',
                        label='True Anomaly' if start_i == regions[0][0] else '')

    ax1.set_ylabel('Distance Score', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'w={window_size}, t={threshold}, h={h} ({n_collections} collections, {distance_type}) — '
        f'{distance_metric.replace("_", " ").title()}  '
        f'(ROC AUC={roc_auc:.3f}, F1={f1:.3f})',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(time_data, 0, predictions, alpha=0.5, color='blue',
                     label='Predicted Anomalies', step='post')
    ax2.fill_between(time_data, 0, true_labels, alpha=0.3, color='red',
                     label='True Anomalies', step='post')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Anomaly Flag', fontsize=12, fontweight='bold')
    ax2.set_title('Predicted vs True Anomalies', fontsize=13, fontweight='bold')
    ax2.set_ylim([-0.1, 1.3])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=46, ha='right')
    plt.tight_layout()
    output_plot = os.path.join(out_dir, f"plot_{distance_metric}.png")
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved plot -> {output_plot}")


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide training graphs sequentially into collections of size h, "
                    "score test windows using median distance across all collections."
    )

    parser.add_argument("--labels_csv",     required=True,
                        help="CSV with 'Timestamp_clean' and 'health' columns")
    parser.add_argument("--train_path",     required=True,  help="Path to train/graphs.pkl")
    parser.add_argument("--test_path",      required=True,  help="Path to test/graphs.pkl")
    parser.add_argument("--window_size",    type=int,   required=True)
    parser.add_argument("--best_threshold", type=float, required=True)
    parser.add_argument("--eval_start",     type=int,   required=True)

    # h loop
    parser.add_argument("--h_start",  type=int, default=20,
                        help="Minimum collection size h (default: 20)")
    parser.add_argument("--h_end",    type=int, required=True,
                        help="Maximum collection size h")
    parser.add_argument("--h_step",   type=int, default=1,
                        help="Step for h loop (default: 1)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip h values where the distance CSV already exists")

    # Output
    parser.add_argument("--out_dir", type=str, default="apply_fixed_results_voting_three")

    # GAE options
    parser.add_argument("--epoch",   type=int, default=500)
    parser.add_argument("--hidden1", type=int, default=24)
    parser.add_argument("--hidden2", type=int, default=12)
    parser.add_argument("--seed",    type=int, default=42)

    # Scoring options
    parser.add_argument("--estimate_method", type=str, default="meanprecision",
                        choices=["meanprecision", "mean", "fixedpoint"])

    # Parallelism for the h-loop (does not change results, only wall time)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Worker processes for the h-loop scoring phase. "
                             "Defaults to $SLURM_CPUS_PER_TASK (or all CPUs if unset).")
    args = parser.parse_args()
    total_start = time.time()

    with open(args.train_path, "rb") as f:
        training_graphs = pickle.load(f)
    with open(args.test_path, "rb") as f:
        testing_graphs = pickle.load(f)

    N_train  = len(training_graphs)
    h_values = list(range(args.h_start, args.h_end + 1, args.h_step))

    if args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    n_workers = max(1, min(n_workers, len(h_values)))

    print(f"Fixed parameters : window_size={args.window_size}, best_threshold={args.best_threshold}")
    print(f"Training graphs  : {N_train}")
    print(f"Testing graphs   : {len(testing_graphs)}")
    print(f"h range          : {args.h_start} to {args.h_end} (step {args.h_step})")
    print(f"Seed             : {args.seed}")
    print(f"Output root      : {args.out_dir}")
    print(f"h-loop workers   : {n_workers}\n")

    # Embed once, reuse for all h values
    print("=" * 60)
    print("Embedding phase (done once)")
    print("=" * 60)
    embedded = embed_all(
        training_graphs = training_graphs,
        testing_graphs  = testing_graphs,
        epoch           = args.epoch,
        hidden1         = args.hidden1,
        hidden2         = args.hidden2,
        seed            = args.seed,
    )
    print(f"  Train time : {embedded['train_time']:.1f}s")
    print(f"  Test embed : {embedded['test_time']:.1f}s\n")

    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows  = []
    total_score_time = 0.0
    total_method_times = {m: 0.0 for m in METRICS}

    print("=" * 60)
    print(f"Scoring phase: {len(h_values)} h values x {len(METRICS)} metrics "
          f"(pool of {n_workers} worker process(es))")
    print("=" * 60)

    # score_one_h() per h is independent (each h writes its own out_dir/h{h}/
    # folder), so run them across a process pool. "spawn" avoids forking a
    # process that has already opened a TF/GPU session in embed_all() above;
    # score_worker_old.py itself never imports tensorflow, so spawned workers
    # stay CPU-only.
    mp_context = multiprocessing.get_context("spawn")
    h_task_results = {}
    h_wall_times = {}
    scoring_phase_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
        future_to_h = {
            pool.submit(
                score_one_h,
                h,
                embedded,
                args.estimate_method,
                args.out_dir,
                args.skip_existing,
            ): h
            for h in h_values
        }

        for fut in as_completed(future_to_h):
            h = future_to_h[fut]
            csv_paths, method_times, h_wall_time = fut.result()
            # h_wall_time is timed inside the worker itself (see score_worker.
            # score_one_h), so it excludes time spent queued for a free
            # worker — a clean per-h figure even though h's run concurrently.
            h_task_results[h] = (csv_paths, method_times)
            h_wall_times[h] = h_wall_time

    total_score_time = time.time() - scoring_phase_start
    h20_scoring_time = h_wall_times.get(20)
    if h20_scoring_time is not None:
        print(f"  h=20 scoring time (self-timed, excludes queue wait): {h20_scoring_time:.1f}s")

    # Process results in original h order, so summary.csv row order matches
    # the sequential version exactly.
    for done, h in enumerate(h_values, 1):
        n_collections = N_train // h
        if h not in h_task_results:
            continue
        csv_paths, method_times = h_task_results[h]
        print(f"\n[{done}/{len(h_values)}] h={h}  ({n_collections} collections)")
        if not csv_paths:
            continue
        for m in METRICS:
            total_method_times[m] += method_times[m]

        h_results = []
        for dt, csv_path in csv_paths.items():
            if not os.path.exists(csv_path):
                continue
            for metric in METRICS:
                try:
                    result = compute_metrics(
                        distance_csv    = csv_path,
                        labels_csv      = args.labels_csv,
                        window_size     = args.window_size,
                        threshold       = args.best_threshold,
                        h               = h,
                        n_collections   = n_collections,
                        distance_type   = dt,
                        distance_metric = metric,
                        eval_start      = args.eval_start,
                    )
                    if result:
                        summary_rows.append(result)
                        h_results.append(result)
                except Exception as e:
                    print(f"  h={h} [{dt}] {metric}: ERROR — {e}")

        if h_results:
            h_df = pd.DataFrame(h_results)
            for metric in METRICS:
                sub = h_df[h_df['distance_metric'] == metric]
                if not sub.empty:
                    best = sub.loc[sub['roc_auc'].idxmax()]
                    print(f"  {metric}: ROC AUC={best['roc_auc']:.10f}, F1={best['f1']:.10f}")

            save_h_metrics_txt(h_results, h, n_collections,
                               args.window_size, args.best_threshold, args.out_dir)

    if not summary_rows:
        print("No results found.")
        exit(1)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")

    for criterion, label in [('roc_auc', 'ROC AUC'), ('f1', 'F1')]:
        print(f"\nBest h per metric (by {label}):")
        for metric in METRICS:
            sub = summary_df[summary_df['distance_metric'] == metric]
            if not sub.empty:
                best = sub.loc[sub[criterion].idxmax()]
                print(f"  {metric}: ROC AUC={best['roc_auc']:.10f}, F1={best['f1']:.10f}, "
                      f"h={int(best['h'])}, n_collections={int(best['n_collections'])}")

    # Pass 2: plot best h per metric (by ROC AUC)
    print(f"\nPass 2: plotting best h per metric (by ROC AUC)...")

    for metric in METRICS:
        sub = summary_df[summary_df['distance_metric'] == metric]
        if sub.empty:
            continue

        best_row  = sub.loc[sub['roc_auc'].idxmax()]
        best_h    = int(best_row['h'])
        best_nc   = int(best_row['n_collections'])
        csv_path  = os.path.join(args.out_dir, f"h{best_h}", DISTANCE_CSV_NAME["median_all"])
        plot_dir  = os.path.join(args.out_dir, f"best_h{best_h}_{metric}_results")

        print(f"  {metric}: best h={best_h} ({best_nc} collections), "
              f"ROC AUC={best_row['roc_auc']:.10f}, F1={best_row['f1']:.10f}")

        try:
            plot_best(
                distance_csv    = csv_path,
                labels_csv      = args.labels_csv,
                window_size     = args.window_size,
                threshold       = args.best_threshold,
                h               = best_h,
                n_collections   = best_nc,
                distance_type   = "median_all",
                distance_metric = metric,
                eval_start      = args.eval_start,
                out_dir         = plot_dir,
            )
        except Exception as e:
            print(f"    ERROR — {e}")

    save_best_results_csv(summary_df, args.out_dir)

    total_test_time = embedded['test_time'] + total_score_time
    total_elapsed   = time.time() - total_start
    h_total = int(total_elapsed // 3600)
    h_min   = int((total_elapsed % 3600) // 60)
    h_sec   = total_elapsed % 60
    print(f"\nTrain time : {embedded['train_time']:.1f}s")
    print(f"Test  time : {total_test_time:.1f}s  "
          f"(embed {embedded['test_time']:.1f}s + scoring {total_score_time:.1f}s)")
    emb_share  = embedded['test_embed_emb_time']
    base_share = embedded['test_embed_base_time']
    print(f"  Per-method test time (embed + scoring):")
    for m in METRICS:
        embed_part = emb_share if 'baseline' not in m else base_share
        total_m = embed_part + total_method_times[m]
        print(f"    {m}: {total_m:.1f}s  (embed {embed_part:.1f}s + scoring {total_method_times[m]:.1f}s)")
    print(f"Total time : {h_total:02d}h {h_min:02d}m {h_sec:05.2f}s ({total_elapsed:.1f}s)")
    print(f"All done. Results saved under: {args.out_dir}/")

    # Save timing — transposed: rows=metrics, one column per seed run
    emb_time  = embedded['test_embed_emb_time']
    base_time = embedded['test_embed_base_time']
    timing_values = {
        'seed':                          args.seed,
        'train_time_s':                  round(embedded['train_time'],      4),
        'train_emb_time_s':              round(embedded['train_emb_time'],  4),
        'train_base_time_s':             round(embedded['train_base_time'], 4),
        'test_embed_time_s':             round(embedded['test_time'],       4),
        'test_embed_emb_time_s':         round(emb_time,               4),
        'test_embed_base_time_s':        round(base_time,              4),
        'scoring_time_s':                round(total_score_time,       4),
        'h20_scoring_time_s':            (round(h20_scoring_time, 4)
                                           if h20_scoring_time is not None else np.nan),
        'kl_embedding_scoring_time_s':   round(total_method_times['kl_embedding'],   4),
        'kl_baseline_scoring_time_s':    round(total_method_times['kl_baseline'],    4),
        'smdl_embedding_scoring_time_s': round(total_method_times['smdl_embedding'], 4),
        'smdl_baseline_scoring_time_s':  round(total_method_times['smdl_baseline'],  4),
        'kl_embedding_test_time_s':      round(emb_time  + total_method_times['kl_embedding'],   4),
        'kl_baseline_test_time_s':       round(base_time + total_method_times['kl_baseline'],    4),
        'smdl_embedding_test_time_s':    round(emb_time  + total_method_times['smdl_embedding'], 4),
        'smdl_baseline_test_time_s':     round(base_time + total_method_times['smdl_baseline'],  4),
        'total_test_time_s':             round(total_test_time,        4),
        'total_time_s':                  round(total_elapsed,          4),
    }
    timing_csv = os.path.join(args.out_dir, "timing.csv")
    col_name   = f"seed_{args.seed}"
    if os.path.exists(timing_csv):
        timing_df = pd.read_csv(timing_csv, index_col=['phase', 'metric'])
    else:
        idx       = pd.MultiIndex.from_tuples(TIMING_PHASE_METRIC, names=['phase', 'metric'])
        timing_df = pd.DataFrame(index=idx)
    timing_df[col_name] = [timing_values[m] for _, m in TIMING_PHASE_METRIC]
    timing_df.to_csv(timing_csv)
    print(f"Timing saved -> {timing_csv}")
