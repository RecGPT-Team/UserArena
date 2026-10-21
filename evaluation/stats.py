"""Statistical functions shared by the evaluation metrics."""

from __future__ import annotations
import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    return math.sqrt(sum(((value - average) ** 2 for value in values)) / len(values))


def quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    clean = sorted((value for value in values if value is not None))
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    position = fraction * (len(clean) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(clean[lower])
    weight = position - lower
    return float(clean[lower] * (1 - weight) + clean[upper] * weight)


def distribution(
    values: Iterable[str], support: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    counts = Counter(values)
    labels = list(support) if support is not None else sorted(counts)
    total = sum((counts[label] for label in labels))
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: counts[label] / total for label in labels}


def js_divergence(
    left: Dict[str, float],
    right: Dict[str, float],
    support: Optional[Sequence[str]] = None,
) -> float:
    labels = list(support) if support is not None else sorted(set(left) | set(right))
    if not labels:
        return 0.0
    left_total = sum((max(0.0, left.get(label, 0.0)) for label in labels))
    right_total = sum((max(0.0, right.get(label, 0.0)) for label in labels))
    if left_total <= 0.0 and right_total <= 0.0:
        return 0.0
    if left_total <= 0.0 or right_total <= 0.0:
        return 1.0

    def term(probability: float, mixture: float) -> float:
        if probability <= 0.0 or mixture <= 0.0:
            return 0.0
        return probability * math.log2(probability / mixture)

    total = 0.0
    for label in labels:
        p = max(0.0, left.get(label, 0.0)) / left_total
        q = max(0.0, right.get(label, 0.0)) / right_total
        m = 0.5 * (p + q)
        total += 0.5 * term(p, m) + 0.5 * term(q, m)
    return min(1.0, max(0.0, total))


def total_variation(
    left: Dict[str, float],
    right: Dict[str, float],
    support: Optional[Sequence[str]] = None,
) -> float:
    labels = list(support) if support is not None else sorted(set(left) | set(right))
    return 0.5 * sum(
        (abs(left.get(label, 0.0) - right.get(label, 0.0)) for label in labels)
    )


def survival_curve(
    lengths: Sequence[int],
    horizon: int,
    observed_events: Optional[Sequence[bool]] = None,
) -> List[float]:
    if not lengths or horizon <= 0:
        return [0.0] * max(0, horizon)
    events = (
        list(observed_events) if observed_events is not None else [True] * len(lengths)
    )
    if len(events) != len(lengths):
        raise ValueError("observed_events must align with lengths")
    if any((length < 0 for length in lengths)):
        raise ValueError("survival lengths must be non-negative")
    survival = 1.0 - sum(
        (length == 0 and event for length, event in zip(lengths, events))
    ) / len(lengths)
    curve = []
    for step in range(1, horizon + 1):
        curve.append(survival)
        at_risk = sum((1 for length in lengths if length >= step))
        observed = sum(
            (1 for length, event in zip(lengths, events) if event and length == step)
        )
        if at_risk:
            survival *= 1.0 - observed / at_risk
    return curve


def survival_distance(left: Sequence[float], right: Sequence[float]) -> float:
    horizon = max(len(left), len(right))
    if horizon == 0:
        return 0.0
    total = 0.0
    for index in range(horizon):
        a = left[index] if index < len(left) else 0.0
        b = right[index] if index < len(right) else 0.0
        total += abs(a - b)
    return total / horizon


def quantile_bins(values: Sequence[float], bins: int) -> List[float]:
    clean = sorted((value for value in values if value is not None))
    if not clean or bins < 2:
        return []
    edges: List[float] = []
    for index in range(1, bins):
        edge = quantile(clean, index / bins)
        if edge is not None and (not edges or edge > edges[-1]):
            edges.append(edge)
    return edges


def bin_labels(values: Sequence[Optional[float]], edges: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for value in values:
        if value is None:
            labels.append("unknown")
            continue
        position = 0
        for edge in edges:
            if value > edge:
                position += 1
            else:
                break
        labels.append("b{}".format(position))
    return labels


def bin_support(edges: Sequence[float]) -> List[str]:
    return ["b{}".format(index) for index in range(len(edges) + 1)] + ["unknown"]


def macro_f1(
    pairs: Sequence[Tuple[str, str]], labels: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, dict], float]:
    if labels is None:
        universe = sorted(
            {truth for truth, _ in pairs} | {pred for _, pred in pairs if pred}
        )
    else:
        universe = list(labels)
    per_class: Dict[str, dict] = {}
    total = 0.0
    for label in universe:
        tp = sum((1 for truth, pred in pairs if truth == label and pred == label))
        fp = sum((1 for truth, pred in pairs if truth != label and pred == label))
        fn = sum((1 for truth, pred in pairs if truth == label and pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum((1 for truth, _ in pairs if truth == label)),
        }
        total += f1
    return (per_class, total / len(universe) if universe else 0.0)


def reciprocal_rank(ranking: Sequence[str], target: str) -> float:
    for position, candidate in enumerate(ranking, 1):
        if str(candidate) == str(target):
            return 1.0 / position
    return 0.0


def pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    left_mean = sum((a for a, _ in pairs)) / len(pairs)
    right_mean = sum((b for _, b in pairs)) / len(pairs)
    covariance = sum(((a - left_mean) * (b - right_mean) for a, b in pairs))
    left_var = sum(((a - left_mean) ** 2 for a, _ in pairs))
    right_var = sum(((b - right_mean) ** 2 for _, b in pairs))
    if left_var <= 0 or right_var <= 0:
        return None
    return covariance / math.sqrt(left_var * right_var)


def scaled_abs_diff(real: float, sim: float, scale: float) -> float:
    if scale <= 0:
        return 0.0 if abs(sim - real) < 1e-09 else 1.0
    return min(1.0, abs(sim - real) / scale)
