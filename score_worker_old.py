"""
CPU scoring worker used by step 3.
Fits Dirichlet models to sequential collections of graph representations and
computes median KL/SMDL test scores for each collection size h.
Writes per-h score CSVs without importing TensorFlow.

score_worker_old.py

TF-free scoring logic used by 3_run_graphs.py's h-loop.

Kept in its own module (importing only numpy/pandas/dirichlet/util) so it can
be run in separate worker processes without pulling in tensorflow/gae —
gae.embedding opens a TF session at import time, and every worker reimporting
that chain would otherwise each try to grab the GPU for no reason.
"""

import os
import time

import numpy as np
import pandas as pd

from dirichlet import dirichlet  # type: ignore
from util import kl_dirichlet, nml_code_length

METRICS = ["kl_embedding", "kl_baseline", "smdl_embedding", "smdl_baseline"]

DISTANCE_CSV_NAME = {
    "median_all": "distance_median_all.csv",
}


def to_strict_simplex(X, eps=1e-12, smooth=1e-3):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    X = np.clip(X, 0.0, None)
    X = X + smooth
    s = X.sum(axis=1, keepdims=True)
    return X / np.clip(s, eps, None)


def mle_safe(D, method="meanprecision", maxiter=200, tol=1e-7, cap_s=1e4):
    try:
        a = dirichlet.mle(D, method=method, maxiter=maxiter, tol=tol)
        s = float(np.sum(a))
        if not np.isfinite(s) or s > cap_s:
            m = np.mean(D, axis=0)
            m = np.clip(m, 1e-12, None)
            m = m / m.sum()
            a = m * cap_s
        return a
    except Exception:
        m = np.mean(D, axis=0)
        m = np.clip(m, 1e-12, None)
        m = m / m.sum()
        return m * 100.0


def _normalize_series(arr):
    mx = np.max(arr)
    return arr / mx if mx > 0 else arr


def score_one_h(h, embedded, estimate_method, out_dir, skip_existing):
    """
    Divide training graphs sequentially into groups of h, compute median distance
    across all collections for each test window. Saves one CSV per h value.

    Returns (csv_paths, method_times, wall_time_s). wall_time_s is timed from
    the moment this function starts *running* inside its worker process, not
    from when the task was submitted — so it excludes any time spent waiting
    in the process pool's queue for a free worker, giving a clean per-h figure
    even when many h values are scored concurrently.
    """
    t_h_start = time.time()
    h_dir    = os.path.join(out_dir, f"h{h}")
    csv_path = os.path.join(h_dir, DISTANCE_CSV_NAME["median_all"])

    zero_times = {m: 0.0 for m in METRICS}
    if skip_existing and os.path.exists(csv_path):
        print(f"  h={h}: skip (CSV exists)")
        return {"median_all": csv_path}, zero_times, time.time() - t_h_start

    all_train_vecs_simplex     = embedded['all_train_vecs_simplex']
    all_train_vecs_org_simplex = embedded['all_train_vecs_org_simplex']
    test_vecs_simplex          = embedded['test_vecs_simplex']
    test_vecs_org_simplex      = embedded['test_vecs_org_simplex']
    N_train                    = embedded['N_train']

    n_collections = N_train // h
    if n_collections < 1:
        print(f"  h={h}: skip (n_collections=0)")
        return {}, zero_times, time.time() - t_h_start

    N_test = len(test_vecs_simplex)
    if N_test < h:
        print(f"  h={h}: skip (not enough test graphs: {N_test} < {h})")
        return {}, zero_times, time.time() - t_h_start

    # Build collections sequentially (no ranking)
    collections_vecs_simplex     = []
    collections_vecs_org_simplex = []
    collections_alpha            = []
    collections_alpha_org        = []

    for k in range(n_collections):
        idx      = list(range(k * h, (k + 1) * h))
        vecs     = to_strict_simplex(np.array([all_train_vecs_simplex[i]     for i in idx]))
        vecs_org = to_strict_simplex(np.array([all_train_vecs_org_simplex[i] for i in idx]))
        collections_vecs_simplex.append(vecs)
        collections_vecs_org_simplex.append(vecs_org)
        collections_alpha.append(mle_safe(vecs,     method=estimate_method))
        collections_alpha_org.append(mle_safe(vecs_org, method=estimate_method))

    print(f"  h={h}: {n_collections} collections, {N_train % h} training graphs discarded")

    # Sliding window scoring — median of all collections
    n_scores = N_test - h + 1
    kl_all = []; kl_org_all = []; smdl_all = []; smdl_org_all = []
    t_kl_emb = t_kl_base = t_smdl_emb = t_smdl_base = 0.0

    for i in range(n_scores):
        W     = to_strict_simplex(np.array(test_vecs_simplex[i:i + h]))
        W_org = to_strict_simplex(np.array(test_vecs_org_simplex[i:i + h]))

        current_alpha     = mle_safe(W,     method=estimate_method)
        current_alpha_org = mle_safe(W_org, method=estimate_method)

        nml_current     = nml_code_length(current_alpha,     h)
        nml_current_org = nml_code_length(current_alpha_org, h)

        kl_k = []; kl_org_k = []; smdl_k = []; smdl_org_k = []

        for k in range(n_collections):
            alpha_k     = collections_alpha[k]
            alpha_org_k = collections_alpha_org[k]

            _t = time.time()
            kl_k.append(kl_dirichlet(current_alpha, alpha_k))
            t_kl_emb += time.time() - _t

            _t = time.time()
            kl_org_k.append(kl_dirichlet(current_alpha_org, alpha_org_k))
            t_kl_base += time.time() - _t

            _t = time.time()
            combined       = np.vstack([collections_vecs_simplex[k], W])
            combined_alpha = mle_safe(combined, method=estimate_method)
            smdl_k.append(
                nml_code_length(combined_alpha, 2 * h)
                - nml_code_length(alpha_k, h)
                - nml_current
            )
            t_smdl_emb += time.time() - _t

            _t = time.time()
            combined_org       = np.vstack([collections_vecs_org_simplex[k], W_org])
            combined_alpha_org = mle_safe(combined_org, method=estimate_method)
            smdl_org_k.append(
                nml_code_length(combined_alpha_org, 2 * h)
                - nml_code_length(alpha_org_k, h)
                - nml_current_org
            )
            t_smdl_base += time.time() - _t

        kl_all.append(float(np.median(kl_k)))
        kl_org_all.append(float(np.median(kl_org_k)))
        smdl_all.append(float(np.median(smdl_k)))
        smdl_org_all.append(float(np.median(smdl_org_k)))

    padding      = np.zeros(h - 1)
    kl_all       = _normalize_series(np.array(kl_all))
    kl_org_all   = _normalize_series(np.array(kl_org_all))
    smdl_all     = _normalize_series(np.array(smdl_all))
    smdl_org_all = _normalize_series(np.array(smdl_org_all))

    os.makedirs(h_dir, exist_ok=True)
    pd.DataFrame({
        'kl_embedding':   np.concatenate([padding, kl_all]),
        'kl_baseline':    np.concatenate([padding, kl_org_all]),
        'smdl_embedding': np.concatenate([padding, smdl_all]),
        'smdl_baseline':  np.concatenate([padding, smdl_org_all]),
    }).to_csv(csv_path, index=False)

    method_times = {
        'kl_embedding':   round(t_kl_emb,   4),
        'kl_baseline':    round(t_kl_base,   4),
        'smdl_embedding': round(t_smdl_emb,  4),
        'smdl_baseline':  round(t_smdl_base, 4),
    }
    print(f"  h={h}: saved CSV -> {h_dir}/")
    return {"median_all": csv_path}, method_times, time.time() - t_h_start
