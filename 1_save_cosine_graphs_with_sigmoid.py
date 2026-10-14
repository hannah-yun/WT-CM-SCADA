"""
Step 1: Generate and save time-series graphs for candidate window sizes and thresholds.

Reads training and test CSVs, maps cosine similarity through sigmoid(2 * S),
and saves NetworkX graphs and window data under --base_dir/w{size}/t{threshold}.
Supports column-wise or row-wise similarity and optional train-based standardization.

Generate overlapping sliding-window NetworkX graphs from TRAIN/TEST multivariate time series CSVs
using cosine similarity with sigmoid mapping.

method="cosine"     → (n, n) variable/column-wise similarity graph
method="cosine_row" → (w, w) time-step/row-wise similarity graph

Outputs (in --out_dir):
  - train/graphs.pkl, train/windows.pkl, train/columns.pkl
  - test/graphs.pkl,  test/windows.pkl,  test/columns.pkl
  - standardize_stats.pkl (if --standardize)
"""

import os
import pickle
import argparse
import time
from datetime import datetime
import numpy as np
import networkx as nx
import pandas as pd


# ------------------------------------------------------
# Similarity matrix — cosine only (with sigmoid mapping)
# ------------------------------------------------------
def similarity_matrix(Xw: np.ndarray, method: str) -> np.ndarray:
    """
    Xw : (w, n) window slice
    Returns:
      cosine     → (n, n) variable-wise similarity
      cosine_row → (w, w) time-step-wise similarity
    Both are mapped through sigmoid: 1 / (1 + exp(-2 * S))
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=1e10, neginf=-1e10)
    Xw = np.clip(Xw, -1e10, 1e10)

    if method == "cosine":
        norms = np.sqrt(np.sum(Xw * Xw, axis=0, keepdims=True))
        norms = np.maximum(norms, 1e-10)
        Xn = Xw / norms
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        n = Xn.shape[1]
        S = np.zeros((n, n), dtype=np.float64)
        chunk_size = 100
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            S[i:i_end, :] = Xn[:, i:i_end].T @ Xn
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.clip(S, -1.0, 1.0)
        return 1.0 / (1.0 + np.exp(-2.0 * S))

    if method == "cosine_row":
        norms = np.sqrt(np.sum(Xw * Xw, axis=1, keepdims=True))
        norms = np.maximum(norms, 1e-10)
        Xn = Xw / norms
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        w = Xn.shape[0]
        S = np.zeros((w, w), dtype=np.float64)
        chunk_size = 100
        for i in range(0, w, chunk_size):
            i_end = min(i + chunk_size, w)
            S[i:i_end, :] = Xn[i:i_end, :] @ Xn.T
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S = np.clip(S, -1.0, 1.0)
        return 1.0 / (1.0 + np.exp(-2.0 * S))

    raise ValueError("method must be 'cosine' or 'cosine_row'")


# ------------------------------------------------------
# Graph builders
# ------------------------------------------------------
def _build_graph_from_similarity(S: np.ndarray, node_count: int,
                                  threshold: float, weighted: bool) -> nx.Graph:
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    for i in range(node_count):
        for j in range(i + 1, node_count):
            s = float(S[i, j])
            if s >= threshold:
                if weighted:
                    G.add_edge(i, j, weight=s)
                else:
                    G.add_edge(i, j)
    return G


def build_window_graphs(X: np.ndarray, window_size: int, step_size: int,
                        method: str, threshold: float, weighted: bool):
    """Overlapping sliding windows (used for test set)."""
    m, n = X.shape
    if window_size <= 1 or window_size > m:
        raise ValueError(f"window_size must be in [2, {m}], got {window_size}")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, got {step_size}")

    node_count = window_size if method == "cosine_row" else n
    graphs, windows = [], []
    for start in range(0, m - window_size + 1, step_size):
        end = start + window_size
        S = similarity_matrix(X[start:end, :], method)
        graphs.append(_build_graph_from_similarity(S, node_count, threshold, weighted))
        windows.append((start, end))
    return graphs, windows


def build_nonoverlapping_window_graphs(X: np.ndarray, window_size: int,
                                        method: str, threshold: float, weighted: bool):
    """Non-overlapping windows (used for train set). Skips leading rows if needed."""
    m, n = X.shape
    n_windows = m // window_size
    if n_windows == 0:
        raise ValueError(f"window_size ({window_size}) > train length ({m}); no windows possible.")
    offset = m % window_size

    node_count = window_size if method == "cosine_row" else n
    graphs, windows = [], []
    for i in range(n_windows):
        start = offset + i * window_size
        end = start + window_size
        S = similarity_matrix(X[start:end, :], method)
        graphs.append(_build_graph_from_similarity(S, node_count, threshold, weighted))
        windows.append((start, end))

    msg = f"skipped first {offset} rows, " if offset > 0 else ""
    print(f"[train] Non-overlapping windows: {msg}{n_windows} windows.")
    return graphs, windows


# ------------------------------------------------------
# Standardization (fit on TRAIN only)
# ------------------------------------------------------
def fit_standardizer(X_train: np.ndarray):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-12
    return mu, sd


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


# ------------------------------------------------------
# Load CSV
# ------------------------------------------------------
def load_csv_timeseries(path: str, drop_cols: list[str],
                         use_col_range: list[int] | None,
                         start_row: int, n_rows: int | None):
    df = pd.read_csv(path)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    if use_col_range is not None and len(use_col_range) == 2:
        start, end = use_col_range
        if start < 1 or end < start:
            raise ValueError(f"--use_cols must be 'start end' with start>=1 and end>=start, got {use_col_range}")
        df = df.iloc[:, start - 1:end]
    if start_row > 0:
        df = df.iloc[start_row:, :]
    if n_rows is not None and n_rows > 0:
        df = df.iloc[:n_rows, :]
    df = df.select_dtypes(include=["number"])
    if df.shape[1] == 0:
        raise ValueError("No numeric columns found after filtering.")
    df = df.ffill().bfill().fillna(0.0)
    return df.to_numpy(dtype=float), list(df.columns)


# ------------------------------------------------------
# Save & validate
# ------------------------------------------------------
def validate_graphs(graphs, split_name: str):
    if not graphs:
        print(f"[{split_name}] WARNING: No graphs generated!")
        return
    edge_counts = [g.number_of_edges() for g in graphs]
    empty = sum(1 for e in edge_counts if e == 0)
    avg   = sum(edge_counts) / len(edge_counts)
    print(f"[{split_name}] {len(graphs)} graphs | nodes={graphs[0].number_of_nodes()} "
          f"| edges avg={avg:.1f} min={min(edge_counts)} max={max(edge_counts)} "
          f"| empty={empty}/{len(graphs)}")
    if empty > 0:
        print(f"  WARNING: {empty} empty graphs — consider lowering --threshold")


def save_split(out_dir: str, split_name: str, graphs, windows, columns):
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "graphs.pkl"),  "wb") as f:
        pickle.dump(graphs, f)
    with open(os.path.join(split_dir, "windows.pkl"), "wb") as f:
        pickle.dump(windows, f)
    if columns is not None:
        with open(os.path.join(split_dir, "columns.pkl"), "wb") as f:
            pickle.dump(columns, f)
    print(f"[{split_name}] Saved {len(graphs)} graphs -> {split_dir}")
    validate_graphs(graphs, split_name)


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    X_train, cols_train = load_csv_timeseries(
        args.train_csv, args.drop_cols, args.use_cols, args.start_row, args.n_rows_train)
    X_test, cols_test = load_csv_timeseries(
        args.test_csv,  args.drop_cols, args.use_cols, args.start_row, args.n_rows_test)

    if cols_train != cols_test:
        raise ValueError(
            "TRAIN and TEST numeric columns do not match. "
            "Use --drop_cols / --use_cols to align them.")

    if args.standardize:
        mu, sd = fit_standardizer(X_train)
        X_train = apply_standardizer(X_train, mu, sd)
        X_test  = apply_standardizer(X_test,  mu, sd)
        stats_path = os.path.join(args.out_dir, "standardize_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump({"mean": mu, "std": sd}, f)
        print(f"[standardize] Saved stats -> {stats_path}")

    t_train = time.time()
    train_graphs, train_windows = build_nonoverlapping_window_graphs(
        X=X_train, window_size=args.window_size,
        method=args.method, threshold=args.threshold,
        weighted=not args.unweighted,
    )
    train_graph_time = time.time() - t_train

    t_test = time.time()
    test_graphs, test_windows = build_window_graphs(
        X=X_test, window_size=args.window_size, step_size=args.step_size,
        method=args.method, threshold=args.threshold,
        weighted=not args.unweighted,
    )
    test_graph_time = time.time() - t_test

    print(f"  Graph build time — train: {train_graph_time:.2f}s  test: {test_graph_time:.2f}s")

    save_split(args.out_dir, "train", train_graphs, train_windows, cols_train)
    save_split(args.out_dir, "test",  test_graphs,  test_windows,  cols_test)

    return train_graph_time, test_graph_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cosine-similarity graphs (sigmoid-mapped) for all window_size x threshold combos."
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv",  required=True)

    # Loop ranges
    parser.add_argument("--ws_start",  type=int,   default=24,    help="Window size start (inclusive)")
    parser.add_argument("--ws_end",    type=int,   default=96,    help="Window size end (inclusive)")
    parser.add_argument("--ws_step",   type=int,   default=1,     help="Window size step")
    parser.add_argument("--thr_start", type=float, default=0.60,  help="Threshold start (inclusive)")
    parser.add_argument("--thr_end",   type=float, default=0.95,  help="Threshold end (inclusive)")
    parser.add_argument("--thr_step",  type=float, default=0.005, help="Threshold step")
    parser.add_argument("--base_dir",  type=str,   default="cosine_graphs",
                        help="Root output directory (base_dir/w{ws}/t{thresh}/)")

    # Per-run options
    parser.add_argument("--method",    choices=["cosine", "cosine_row"], default="cosine_row",
                        help="cosine=column-wise (n×n), cosine_row=row/time-step-wise (w×w)")
    parser.add_argument("--step_size", type=int,   default=1)
    parser.add_argument("--unweighted",   action="store_true")
    parser.add_argument("--standardize",  action="store_true", help="Z-score normalise using TRAIN stats only")
    parser.add_argument("--drop_cols",    nargs="*", default=["Timestamp", "timestamp", "Time", "time"])
    parser.add_argument("--use_cols",     nargs=2, type=int, default=None)
    parser.add_argument("--start_row",    type=int, default=0)
    parser.add_argument("--n_rows_train", type=int, default=None)
    parser.add_argument("--n_rows_test",  type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip combos whose output folders already exist")

    args = parser.parse_args()

    window_sizes = range(args.ws_start, args.ws_end + 1, args.ws_step)
    n_thr        = round((args.thr_end - args.thr_start) / args.thr_step) + 1
    thresholds   = [round(args.thr_start + i * args.thr_step, 4) for i in range(n_thr)]
    total        = len(window_sizes) * len(thresholds)
    done         = 0
    total_train_graph_time = 0.0
    total_test_graph_time  = 0.0

    start_dt  = datetime.now()
    t_start   = time.time()
    print(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"method={args.method}  |  {len(window_sizes)} window sizes x {len(thresholds)} thresholds = {total} combos")
    print(f"Output root: {args.base_dir}/\n")

    for ws in window_sizes:
        for thresh in thresholds:
            done += 1
            thresh_label = f"t{thresh:g}"
            out_dir = os.path.join(args.base_dir, f"w{ws}", thresh_label)

            if args.skip_existing and os.path.exists(os.path.join(out_dir, "train", "graphs.pkl")):
                print(f"[{done}/{total}] Skipping w{ws} {thresh_label} (exists)")
                continue

            print(f"\n[{done}/{total}] window_size={ws}, threshold={thresh:.3f}  ->  {out_dir}")
            args.window_size = ws
            args.threshold   = thresh
            args.out_dir     = out_dir
            try:
                tr_t, te_t = main(args)
                total_train_graph_time += tr_t
                total_test_graph_time  += te_t
            except Exception as e:
                print(f"  ERROR: {e}")

    end_dt        = datetime.now()
    total_elapsed = time.time() - t_start
    print(f"\nEnd              : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train graph time : {total_train_graph_time:.2f}s")
    print(f"Test  graph time : {total_test_graph_time:.2f}s")
    print(f"Total time       : {total_elapsed:.2f}s")
    print(f"All done: {total} combinations under {args.base_dir}/")

    os.makedirs(args.base_dir, exist_ok=True)
    timing_path = os.path.join(args.base_dir, "timing.csv")
    pd.DataFrame([{
        "start_time":          start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time":            end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "train_graph_time_s":  round(total_train_graph_time, 2),
        "test_graph_time_s":   round(total_test_graph_time,  2),
        "total_time_s":        round(total_elapsed, 2),
        "n_combos":            total,
        "method":              args.method,
        "ws_start":            args.ws_start,
        "ws_end":              args.ws_end,
        "thr_start":           args.thr_start,
        "thr_end":             args.thr_end,
    }]).to_csv(timing_path, index=False)
    print(f"Timing saved -> {timing_path}")
