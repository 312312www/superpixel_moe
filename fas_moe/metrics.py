"""Deterministic sample-level binary FAS metrics and source-domain thresholds."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape or y.size < 2:
        raise ValueError("labels and scores must be equal-length non-trivial vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("labels must be binary and scores must be finite")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("metrics need both live and spoof samples")
    return y, s


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC via tie-aware average ranks; label 1 is live/positive."""

    y, s = _arrays(labels, scores)
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive = y == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    value = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )
    return float(value)


def error_rates(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float]:
    """Compute sample-level rates for live=1, spoof=0, accept-live score>=threshold."""

    y, s = _arrays(labels, scores)
    predicted_live = s >= float(threshold)
    spoof = y == 0
    live = y == 1
    apcer = float(predicted_live[spoof].mean())
    bpcer = float((~predicted_live[live]).mean())
    acer = (apcer + bpcer) / 2.0
    accuracy = float((predicted_live == live).mean())
    return {
        "threshold": float(threshold),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        # Under this binary sample-level definition HTER and ACER coincide.
        "hter": acer,
        "accuracy": accuracy,
    }


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=np.float64).reshape(-1))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    if values.size == 1:
        return np.asarray(
            [np.nextafter(values[0], -np.inf), np.nextafter(values[0], np.inf)]
        )
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2.0
    return np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            midpoints,
            [np.nextafter(values[-1], np.inf)],
        )
    )


def equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, s = _arrays(labels, scores)
    best_difference = float("inf")
    best_value = float("nan")
    for threshold in threshold_candidates(s):
        rates = error_rates(y, s, float(threshold))
        difference = abs(rates["apcer"] - rates["bpcer"])
        value = (rates["apcer"] + rates["bpcer"]) / 2.0
        if difference < best_difference - 1e-12 or (
            abs(difference - best_difference) <= 1e-12 and value < best_value
        ):
            best_difference = difference
            best_value = value
    return float(best_value)


def select_macro_hter_threshold(
    labels: Sequence[int], scores: Sequence[float], domains: Sequence[str]
) -> dict[str, Any]:
    """Select a source-validation threshold by minimum domain-macro HTER.

    Ties within 1e-12 choose the largest threshold, a deterministic and
    security-conservative rule (fewer spoof samples are accepted as live).
    """

    y, s = _arrays(labels, scores)
    d = np.asarray(domains, dtype=str).reshape(-1)
    if d.shape != y.shape:
        raise ValueError("domains must have the same length as labels")
    unique_domains = sorted(np.unique(d).tolist())
    if not unique_domains:
        raise ValueError("at least one domain is required")
    best_threshold = float("nan")
    best_hter = float("inf")
    for threshold in threshold_candidates(s):
        values = [error_rates(y[d == domain], s[d == domain], threshold)["hter"] for domain in unique_domains]
        macro_hter = float(np.mean(values))
        if macro_hter < best_hter - 1e-12 or (
            abs(macro_hter - best_hter) <= 1e-12 and threshold > best_threshold
        ):
            best_hter = macro_hter
            best_threshold = float(threshold)
    return {
        "threshold": best_threshold,
        "macro_hter": best_hter,
        "domains": unique_domains,
        "candidate_count": int(len(threshold_candidates(s))),
        "tie_break": "largest_threshold",
    }


def evaluate_scores(
    labels: Sequence[int],
    scores: Sequence[float],
    domains: Sequence[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Return per-domain and macro sample-level metrics at a locked threshold."""

    y, s = _arrays(labels, scores)
    d = np.asarray(domains, dtype=str).reshape(-1)
    if d.shape != y.shape:
        raise ValueError("domains must have the same length as labels")
    per_domain: dict[str, dict[str, float | int]] = {}
    for domain in sorted(np.unique(d).tolist()):
        mask = d == domain
        rates: dict[str, float | int] = error_rates(y[mask], s[mask], threshold)
        rates["auc"] = roc_auc(y[mask], s[mask])
        rates["eer"] = equal_error_rate(y[mask], s[mask])
        rates["samples"] = int(mask.sum())
        rates["live"] = int((y[mask] == 1).sum())
        rates["spoof"] = int((y[mask] == 0).sum())
        per_domain[domain] = rates
    metric_names = ("auc", "eer", "apcer", "bpcer", "acer", "hter", "accuracy")
    macro = {
        name: float(np.mean([float(values[name]) for values in per_domain.values()]))
        for name in metric_names
    }
    return {
        "evaluation_unit": "one local NPY array row",
        "label_semantics": {"live": 1, "spoof": 0, "score": "P(live)"},
        "threshold": float(threshold),
        "per_domain": per_domain,
        "macro": macro,
    }


__all__ = [
    "equal_error_rate",
    "error_rates",
    "evaluate_scores",
    "roc_auc",
    "select_macro_hter_threshold",
    "threshold_candidates",
]
