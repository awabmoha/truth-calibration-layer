from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": i,
                "low": lo,
                "high": hi,
                "count": 0,
                "accuracy": None,
                "confidence": None,
                "gap": None,
            })
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        gap = abs(acc - conf)
        weight = count / len(y_true)
        ece += weight * gap
        mce = max(mce, gap)
        rows.append({
            "bin": i,
            "low": lo,
            "high": hi,
            "count": count,
            "accuracy": acc,
            "confidence": conf,
            "gap": gap,
        })

    return float(ece), float(mce), rows


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))
