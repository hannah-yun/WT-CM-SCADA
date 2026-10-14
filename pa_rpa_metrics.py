"""
Evaluation helper used by step 3.
Computes pointwise, point-adjusted, and revised point-adjusted precision, recall,
and F1 from binary labels and predictions. Optionally uses external Merlion helpers
when available, otherwise uses the NumPy fallback defined here.

pa_rpa_metrics.py

Pointwise (PW), Point-Adjusted (PA), and Revised Point-Adjusted (RPA)
precision / recall / F1 metrics for anomaly detection.

Uses merlion's ScoreAcc if available; falls back to pure-numpy merlion-style
formula (FP always from original pointwise predictions) otherwise.
"""

import os
import importlib.util
import numpy as np


# ── PA / RPA prediction adjustments (no merlion needed) ──────────────────────

def point_adjust(pred):
    pred = np.array(pred)
    pa   = pred.copy()
    in_anom = False
    for i in range(len(pred)):
        if pred[i] == 1 and not in_anom:
            in_anom = True
            pa[i]   = 1
        elif in_anom and pred[i] == 1:
            pa[i] = 0
        elif in_anom and pred[i] == 0:
            in_anom = False
    return pa


def revised_point_adjust(true_label, pred):
    pred       = np.array(pred)
    true_label = np.array(true_label)
    rpa        = np.zeros_like(pred)
    inside     = False
    for i in range(len(pred)):
        if true_label[i] == 1:
            inside = True
            if pred[i] == 1:
                rpa[i] = 1
        else:
            if inside:
                inside = False
            if pred[i] == 1:
                rpa[i] = 1
    return rpa


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ── Try to load merlion + build_scoreacc ─────────────────────────────────────

_MERLION_AVAILABLE = False
_build_scoreacc    = None
_ScoreType         = None

try:
    from merlion.evaluate.anomaly import ScoreType as _ScoreType

    _eval_metrics_path = os.path.join(
        os.path.dirname(__file__), '..', 'AD_ref', 'evaluate_all_metrics.py')
    _spec = importlib.util.spec_from_file_location(
        'evaluate_all_metrics', os.path.abspath(_eval_metrics_path))
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _build_scoreacc    = _mod.build_scoreacc
    _MERLION_AVAILABLE = True
except Exception:
    pass


# ── Main function ─────────────────────────────────────────────────────────────

def compute_pa_rpa_metrics(true_labels, predictions):
    """
    Return PW / PA / RPA precision, recall, F1.
    Uses merlion's ScoreAcc if available, otherwise pure-numpy merlion-style.
    """
    true_labels = np.array(true_labels).astype(int)
    predictions = np.array(predictions).astype(int)
    pa_pred     = point_adjust(predictions)
    rpa_pred    = revised_point_adjust(true_labels, predictions)

    if _MERLION_AVAILABLE:
        acc = _build_scoreacc(
            true_label=true_labels,
            pred_label=predictions,
            pred_pa=pa_pred,
            pred_rpa=rpa_pred,
        )
        return {
            'pw_precision':  float(acc.precision(_ScoreType.Pointwise)),
            'pw_recall':     float(acc.recall(_ScoreType.Pointwise)),
            'pw_f1':         float(acc.f1(_ScoreType.Pointwise)),
            'pa_precision':  float(acc.precision(_ScoreType.PointAdjusted)),
            'pa_recall':     float(acc.recall(_ScoreType.PointAdjusted)),
            'pa_f1':         float(acc.f1(_ScoreType.PointAdjusted)),
            'rpa_precision': float(acc.precision(_ScoreType.RevisedPointAdjusted)),
            'rpa_recall':    float(acc.recall(_ScoreType.RevisedPointAdjusted)),
            'rpa_f1':        float(acc.f1(_ScoreType.RevisedPointAdjusted)),
        }

    # Fallback: merlion-style formula, FP always from original predictions
    fp = int(np.sum((true_labels == 0) & (predictions == 1)))

    tp_pw,  fn_pw  = (int(np.sum((true_labels == 1) & (predictions == 1))),
                      int(np.sum((true_labels == 1) & (predictions == 0))))
    tp_pa,  fn_pa  = (int(np.sum((true_labels == 1) & (pa_pred == 1))),
                      int(np.sum((true_labels == 1) & (pa_pred == 0))))
    tp_rpa, fn_rpa = (int(np.sum((true_labels == 1) & (rpa_pred == 1))),
                      int(np.sum((true_labels == 1) & (rpa_pred == 0))))

    pw_p,  pw_r,  pw_f1  = _prf(tp_pw,  fp, fn_pw)
    pa_p,  pa_r,  pa_f1  = _prf(tp_pa,  fp, fn_pa)
    rpa_p, rpa_r, rpa_f1 = _prf(tp_rpa, fp, fn_rpa)

    return {
        'pw_precision':  pw_p,  'pw_recall':  pw_r,  'pw_f1':  pw_f1,
        'pa_precision':  pa_p,  'pa_recall':  pa_r,  'pa_f1':  pa_f1,
        'rpa_precision': rpa_p, 'rpa_recall': rpa_r, 'rpa_f1': rpa_f1,
    }
