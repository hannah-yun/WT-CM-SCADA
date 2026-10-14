"""
Optional label-aware scoring worker; step 3 does not currently import it.
Computes Dirichlet KL/SMDL scores and searches lower and upper anomaly thresholds
using evaluation labels. The resulting maximum-F1 thresholds are oracle estimates
for analysis, rather than thresholds learned independently of test labels.

score_worker_twoside.py

Label-aware variant of score_worker.py.

score_worker.py only ever sees embedded[...] vectors and writes raw score
CSVs -- it has no notion of ground-truth labels or evaluation metrics. This
module keeps that file untouched and instead adds, on top of the same
sliding-window KL/SMDL Dirichlet scoring:

  - evaluation_start: an ORIGINAL test-sample index (independent of h) from
    which metrics are computed. h is assumed to be well below
    evaluation_start, so the warm-up padding (the first h-1 samples, which
    have no completed window yet) is always fully inside the dropped range.
  - the best two-sided decision region [lo, hi] (anomaly iff score < lo or
    score > hi), found by exact search over the evaluated range, per metric
    (kl_embedding, kl_baseline, smdl_embedding, smdl_baseline).

The two-sided search maximises F1 on the labels it is given, so it is an
oracle upper bound, not a deployable threshold -- same caveat as
evaluate_anomaly_detection_new.py's find_best_two_sided_threshold.
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
THRESHOLD_CSV_NAME = "threshold_two_sided.csv"


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


def find_best_two_sided_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    Best two-sided decision region: anomaly iff score < lo or score > hi.

    Searches every (lo, hi) pair exactly, via sorting plus prefix sums, and
    keeps the pair with the highest F1. Useful when anomalous samples sit on
    both tails of the score distribution -- a one-sided rule then has to
    sacrifice one of them.

    This maximises F1 *on the labels it is given*, so it is an oracle upper
    bound, not a deployable threshold.

    Args:
        labels: Ground truth binary labels
        scores: Anomaly scores

    Returns:
        Dict with lo, hi, f1, precision, recall, alarm_rate
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order].astype(np.int64)
    n = len(s_sorted)

    cum = np.concatenate([[0], np.cumsum(y_sorted)])
    total_pos = cum[-1]

    best = {'f1': -1.0}
    for i in range(n + 1):                       # flag the i lowest scores
        j = np.arange(i, n + 1)                  # flag the (n - j) highest
        tp = cum[i] + (total_pos - cum[j])
        flagged = i + (n - j)
        fp = flagged - tp
        fn = total_pos - tp
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)

        k = int(np.argmax(f1))
        if f1[k] > best['f1']:
            jj = int(j[k])
            best = {
                'f1': float(f1[k]),
                'lo': float(s_sorted[i - 1]) if i > 0 else float('-inf'),
                'hi': float(s_sorted[jj]) if jj < n else float('inf'),
                'precision': float(tp[k] / flagged[k]) if flagged[k] > 0 else 0.0,
                'recall': float(tp[k] / total_pos) if total_pos > 0 else 0.0,
                'alarm_rate': float(flagged[k] / n),
            }

    return best


def apply_two_sided(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Binary predictions for the region outside [lo, hi]."""
    return ((scores < lo) | (scores > hi)).astype(int)


def score_one_h_twoside(h, embedded, test_labels, estimate_method, out_dir,
                         skip_existing, evaluation_start=80):
    """
    Same sliding-window KL/SMDL scoring as score_worker.score_one_h, plus
    label-aware two-sided threshold search on the evaluated range.

    Divides training graphs sequentially into groups of h, computes median
    distance across all collections for each test window (unchanged from
    score_worker.py), then trims every score series to
    scores[evaluation_start:] and finds the best two-sided F1 threshold
    there for each metric. Saves one score CSV and one threshold CSV per h.

    Args:
        h: window/collection size
        embedded: dict with all_train_vecs_simplex, all_train_vecs_org_simplex,
            test_vecs_simplex, test_vecs_org_simplex, N_train (same as
            score_worker.score_one_h)
        test_labels: (N_test,) binary ground-truth labels, aligned 1:1 with
            embedded['test_vecs_simplex'] / ['test_vecs_org_simplex'] by
            original sample index
        estimate_method: passed to dirichlet.mle
        out_dir: base output directory (an h{h}/ subfolder is created inside)
        skip_existing: skip recomputation if both output CSVs already exist
        evaluation_start: original test-sample index from which the
            threshold search and F1/precision/recall are computed. h is
            assumed to be well below this, so the h-1 warm-up samples are
            always dropped along with the rest of the pre-evaluation range.

    Returns:
        (csv_paths, method_times, wall_time_s). csv_paths has "median_all"
        (full score series, unchanged format) and "threshold" (per-metric
        two-sided threshold/F1 results, evaluated range only).
    """
    t_h_start = time.time()
    h_dir          = os.path.join(out_dir, f"h{h}")
    csv_path       = os.path.join(h_dir, DISTANCE_CSV_NAME["median_all"])
    threshold_path = os.path.join(h_dir, THRESHOLD_CSV_NAME)

    zero_times = {m: 0.0 for m in METRICS}
    if skip_existing and os.path.exists(csv_path) and os.path.exists(threshold_path):
        print(f"  h={h}: skip (CSV exists)")
        return {"median_all": csv_path, "threshold": threshold_path}, zero_times, time.time() - t_h_start

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

    test_labels = np.asarray(test_labels)
    if len(test_labels) != N_test:
        raise ValueError(
            f"test_labels length {len(test_labels)} does not match N_test={N_test}"
        )
    if evaluation_start >= N_test:
        raise ValueError(
            f"evaluation_start={evaluation_start} leaves no samples (N_test={N_test})"
        )

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

    # Full series, aligned 1:1 with original test sample indices 0..N_test-1.
    series = {
        'kl_embedding':   np.concatenate([padding, kl_all]),
        'kl_baseline':    np.concatenate([padding, kl_org_all]),
        'smdl_embedding': np.concatenate([padding, smdl_all]),
        'smdl_baseline':  np.concatenate([padding, smdl_org_all]),
    }

    os.makedirs(h_dir, exist_ok=True)
    pd.DataFrame(series).to_csv(csv_path, index=False)

    # Threshold search on the evaluated range only. evaluation_start is an
    # original sample index, independent of h; h is assumed << evaluation_start
    # so the h-1 warm-up (padded) samples never leak into this range.
    eval_labels = test_labels[evaluation_start:]
    threshold_rows = []
    for metric in METRICS:
        eval_scores = series[metric][evaluation_start:]
        result = find_best_two_sided_threshold(eval_labels, eval_scores)
        result['metric'] = metric
        result['h'] = h
        result['evaluation_start'] = evaluation_start
        result['n_evaluated'] = len(eval_scores)
        threshold_rows.append(result)
        print(f"  h={h} {metric}: F1={result['f1']:.10f} "
              f"P={result['precision']:.10f} R={result['recall']:.10f} "
              f"region=outside[{result['lo']:.10f}, {result['hi']:.10f}]")

    pd.DataFrame(threshold_rows)[
        ['metric', 'h', 'evaluation_start', 'n_evaluated',
         'lo', 'hi', 'f1', 'precision', 'recall', 'alarm_rate']
    ].to_csv(threshold_path, index=False)

    method_times = {
        'kl_embedding':   round(t_kl_emb,   4),
        'kl_baseline':    round(t_kl_base,   4),
        'smdl_embedding': round(t_smdl_emb,  4),
        'smdl_baseline':  round(t_smdl_base, 4),
    }
    print(f"  h={h}: saved CSV -> {h_dir}/")
    return {"median_all": csv_path, "threshold": threshold_path}, method_times, time.time() - t_h_start
