"""Cohort fitting and profile-conditioned fidelity."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from simulator.actions import ACTIONS
from simulator.data import top_category
from ..stats import (
    bin_labels, bin_support, distribution, js_divergence, mean, pearson, quantile,
    quantile_bins, scaled_abs_diff, stdev,
)
from ..trace import RolloutTrace, UserProfile, session_features


SCHEME_LABELS: Dict[str, List[str]] = {
    "activity": ["low", "medium", "high"],
    "purchase_propensity": ["non_buyer", "occasional", "frequent"],
    "category_breadth": ["narrow", "medium", "wide"],
    "price_preference": ["budget", "mid", "premium", "unknown"],
}
DEFAULT_SCHEMES = ("activity", "purchase_propensity", "category_breadth")


def category_breadth(profile: UserProfile) -> int:
    return len(
        {top_category(value) for value in profile.history_categories if value}
    )


class CohortScheme:
    def __init__(self, name: str, edges: Sequence[float]) -> None:
        self.name = name
        self.edges = list(edges)

    def labels(self) -> List[str]:
        return list(SCHEME_LABELS[self.name])

    def assign(self, profile: UserProfile) -> str:
        if self.name == "activity":
            return self._bucket(float(profile.click_history), ["low", "medium", "high"])
        if self.name == "category_breadth":
            return self._bucket(
                float(category_breadth(profile)), ["narrow", "medium", "wide"]
            )
        if self.name == "purchase_propensity":
            if profile.buy_history <= 0:
                return "non_buyer"
            if not self.edges:
                return "occasional"
            return (
                "frequent"
                if float(profile.buy_history) > self.edges[0]
                else "occasional"
            )
        if self.name == "price_preference":
            if profile.history_price_mean is None:
                return "unknown"
            return self._bucket(
                float(profile.history_price_mean), ["budget", "mid", "premium"]
            )
        raise ValueError("unknown cohort scheme: {}".format(self.name))

    def _bucket(self, value: float, names: Sequence[str]) -> str:
        position = 0
        for edge in self.edges:
            if value > edge:
                position += 1
            else:
                break
        return names[min(position, len(names) - 1)]

    def to_dict(self) -> dict:
        return {"scheme": self.name, "edges": [round(edge, 4) for edge in self.edges]}


def fit_schemes(
    real: Sequence[RolloutTrace], schemes: Optional[Sequence[str]] = None
) -> Dict[str, CohortScheme]:
    names = list(schemes) if schemes else list(DEFAULT_SCHEMES)
    profiles: Dict[str, UserProfile] = {}
    for trace in real:
        key = trace.visitor_id or trace.profile.visitor_id or trace.session_id
        profiles.setdefault(key, trace.profile)
    population = list(profiles.values())
    fitted: Dict[str, CohortScheme] = {}
    for name in names:
        if name == "activity":
            edges = _tertile_edges(
                [float(profile.click_history) for profile in population]
            )
        elif name == "category_breadth":
            edges = _tertile_edges(
                [float(category_breadth(profile)) for profile in population]
            )
        elif name == "purchase_propensity":
            buyers = [
                float(profile.buy_history)
                for profile in population
                if profile.buy_history > 0
            ]
            median = quantile(buyers, 0.5)
            edges = [median] if median is not None else []
        elif name == "price_preference":
            edges = _tertile_edges(
                [
                    float(profile.history_price_mean)
                    for profile in population
                    if profile.history_price_mean is not None
                ]
            )
        else:
            raise ValueError("unknown cohort scheme: {}".format(name))
        fitted[name] = CohortScheme(name, edges)
    return fitted


def _tertile_edges(values: Sequence[float]) -> List[float]:
    lower = quantile(values, 1 / 3)
    upper = quantile(values, 2 / 3)
    edges = [edge for edge in (lower, upper) if edge is not None]
    deduped: List[float] = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def assign(
    traces: Sequence[RolloutTrace], scheme: CohortScheme
) -> Dict[str, List[RolloutTrace]]:
    buckets: Dict[str, List[RolloutTrace]] = {label: [] for label in scheme.labels()}
    for trace in traces:
        buckets[scheme.assign(trace.profile)].append(trace)
    return buckets


def visitor_counts(traces: Sequence[RolloutTrace]) -> Counter:
    return Counter(
        (
            trace.visitor_id or trace.profile.visitor_id or trace.session_id
            for trace in traces
        )
    )


DEFAULT_MIN_COHORT = 30


class FeatureScales:
    def __init__(
        self,
        real_features: Sequence[dict],
        price_bins: int = 5,
        depth_bins: int = 4,
        top_categories: int = 10,
    ) -> None:
        lengths = [float(row["length"]) for row in real_features]
        self.length_scale = stdev(lengths) or (mean(lengths) or 1.0)
        prices = [float(value) for row in real_features for value in row["prices"]]
        self.price_edges = quantile_bins(prices, price_bins)
        self.price_support = bin_support(self.price_edges)
        depths = [float(value) for row in real_features for value in row["info_depths"]]
        self.depth_edges = quantile_bins(depths, depth_bins)
        self.depth_support = bin_support(self.depth_edges)
        counts = Counter(
            (
                top_category(str(value or "").strip() or "unknown")
                for row in real_features
                for value in row["categories"]
            )
        )
        self.categories = [label for label, _ in counts.most_common(top_categories)]
        self.category_support = self.categories + ["other"]
        self.action_support = list(ACTIONS)

    def to_dict(self) -> dict:
        return {
            "length_scale": round(self.length_scale, 3),
            "price_edges": [round(edge, 2) for edge in self.price_edges],
            "depth_edges": [round(edge, 2) for edge in self.depth_edges],
            "categories": list(self.categories),
        }


def _signature(features: Sequence[dict], scales: FeatureScales) -> dict:
    if not features:
        return {}
    actions = [
        action
        for row in features
        for action, count in row["action_counts"].items()
        for _ in range(count)
    ]
    depths = [float(value) for row in features for value in row["info_depths"]]
    prices = [float(value) for row in features for value in row["prices"]]
    categories = [
        top_category(str(value or "").strip() or "unknown") for row in features for value in row["categories"]
    ]
    visit_rows = [row for row in features if row["ipv_visit_count"] > 0]
    return {
        "sessions": len(features),
        "mean_length": mean([float(row["length"]) for row in features]),
        "action_distribution": distribution(actions, scales.action_support),
        "depth_distribution": distribution(
            bin_labels(depths, scales.depth_edges), scales.depth_support
        ),
        "mean_info_depth": mean(depths),
        "ipv_rate": mean([float(row["entered_ipv"]) for row in features]),
        "mean_ipv_visits": mean([float(row["ipv_visit_count"]) for row in features]),
        "return_proportion": mean(
            [float(row["return_proportion"]) for row in visit_rows]
        ),
        "cart_rate": mean([float(row["carted"]) for row in features]),
        "buy_rate": mean([float(row["bought"]) for row in features]),
        "category_distribution": distribution(
            [label if label in scales.categories else "other" for label in categories],
            scales.category_support,
        ),
        "price_distribution": distribution(
            bin_labels(prices, scales.price_edges), scales.price_support
        ),
    }


def _block_distances(real: dict, sim: dict, scales: FeatureScales) -> Dict[str, float]:
    return {
        "session_length": scaled_abs_diff(
            real["mean_length"], sim["mean_length"], scales.length_scale
        ),
        "action_distribution": js_divergence(
            real["action_distribution"],
            sim["action_distribution"],
            scales.action_support,
        ),
        "information_depth": js_divergence(
            real["depth_distribution"], sim["depth_distribution"], scales.depth_support
        ),
        "ipv_visit": abs(real["ipv_rate"] - sim["ipv_rate"]),
        "return_proportion": abs(real["return_proportion"] - sim["return_proportion"]),
        "conversion": 0.5
        * (
            abs(real["cart_rate"] - sim["cart_rate"])
            + abs(real["buy_rate"] - sim["buy_rate"])
        ),
        "item_category": js_divergence(
            real["category_distribution"],
            sim["category_distribution"],
            scales.category_support,
        ),
        "item_price": js_divergence(
            real["price_distribution"], sim["price_distribution"], scales.price_support
        ),
    }


def _gap_vector(left: dict, right: dict, scales: FeatureScales) -> float:
    distances = _block_distances(left, right, scales)
    return mean(list(distances.values()))


def _cohort_gaps(
    signatures: Dict[str, dict], scales: FeatureScales
) -> Dict[str, float]:
    labels = sorted(signatures)
    gaps: Dict[str, float] = {}
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            gaps["{}|{}".format(left, right)] = _gap_vector(
                signatures[left], signatures[right], scales
            )
    return gaps


def cohort_conditioned_fidelity(
    real: Sequence[RolloutTrace],
    sim: Sequence[RolloutTrace],
    scheme: CohortScheme,
    scales: FeatureScales,
    min_cohort: int = DEFAULT_MIN_COHORT,
) -> dict:
    real_buckets = assign(real, scheme)
    sim_buckets = assign(sim, scheme)
    real_signatures: Dict[str, dict] = {}
    sim_signatures: Dict[str, dict] = {}
    cohorts: Dict[str, dict] = {}
    counted: List[float] = []
    for label in scheme.labels():
        real_features = [session_features(trace) for trace in real_buckets[label]]
        sim_features = [session_features(trace) for trace in sim_buckets[label]]
        real_signature = _signature(real_features, scales)
        sim_signature = _signature(sim_features, scales)
        entry: Dict[str, object] = {
            "real_sessions": len(real_features),
            "sim_sessions": len(sim_features),
            "real_visitors": len(visitor_counts(real_buckets[label])),
        }
        if not real_signature or not sim_signature:
            entry["status"] = "empty"
            cohorts[label] = entry
            continue
        distances = _block_distances(real_signature, sim_signature, scales)
        fidelity = 1.0 - mean(list(distances.values()))
        adequate = len(real_features) >= min_cohort and len(sim_features) >= min_cohort
        entry.update(
            {
                "status": "counted" if adequate else "undersized",
                "fidelity": round(fidelity, 4),
                "block_distances": {
                    key: round(value, 4) for key, value in distances.items()
                },
                "real_summary": _readable(real_signature),
                "sim_summary": _readable(sim_signature),
            }
        )
        cohorts[label] = entry
        if adequate:
            real_signatures[label] = real_signature
            sim_signatures[label] = sim_signature
            counted.append(fidelity)
    real_gaps = _cohort_gaps(real_signatures, scales)
    sim_gaps = _cohort_gaps(sim_signatures, scales)
    shared = sorted(set(real_gaps) & set(sim_gaps))
    gap_correlation = pearson(
        [real_gaps[key] for key in shared], [sim_gaps[key] for key in shared]
    )
    real_gap_mean = mean([real_gaps[key] for key in shared])
    sim_gap_mean = mean([sim_gaps[key] for key in shared])
    return {
        "scheme": scheme.to_dict(),
        "min_cohort": min_cohort,
        "counted_cohorts": len(counted),
        "cohort_conditioned_behavioral_fidelity": mean(counted)
        if counted
        else None,
        "cohorts": cohorts,
        "cohort_gap": {
            "real": {key: round(real_gaps[key], 4) for key in shared},
            "sim": {key: round(sim_gaps[key], 4) for key in shared},
            "real_mean": round(real_gap_mean, 4),
            "sim_mean": round(sim_gap_mean, 4),
            "preservation_ratio": round(sim_gap_mean / real_gap_mean, 4)
            if real_gap_mean > 0
            else None,
            "correlation": round(gap_correlation, 4)
            if gap_correlation is not None
            else None,
        },
    }


def _readable(signature: dict) -> dict:
    return {
        "sessions": signature["sessions"],
        "mean_length": round(signature["mean_length"], 2),
        "mean_info_depth": round(signature["mean_info_depth"], 2),
        "ipv_rate": round(signature["ipv_rate"], 4),
        "mean_ipv_visits": round(signature["mean_ipv_visits"], 2),
        "return_proportion": round(signature["return_proportion"], 4),
        "cart_rate": round(signature["cart_rate"], 4),
        "buy_rate": round(signature["buy_rate"], 4),
    }


def evaluate(
    real: Sequence[RolloutTrace],
    sim: Sequence[RolloutTrace],
    schemes: Optional[Sequence[str]] = None,
    min_cohort: int = DEFAULT_MIN_COHORT,
) -> dict:
    real_features = [session_features(trace) for trace in real]
    scales = FeatureScales(real_features)
    fitted = fit_schemes(real, schemes)
    results: Dict[str, dict] = {}
    headline: List[float] = []
    skipped: Dict[str, str] = {}
    for name, scheme in fitted.items():
        entry = cohort_conditioned_fidelity(
            real, sim, scheme, scales, min_cohort=min_cohort
        )
        results[name] = entry
        value = entry["cohort_conditioned_behavioral_fidelity"]
        if value is None:
            skipped[name] = "no cohort met min_cohort={}".format(min_cohort)
        elif entry["counted_cohorts"] < 2:
            skipped[name] = "only 1 cohort met min_cohort={}".format(min_cohort)
        else:
            headline.append(value)
    return {
        "feature_scales": scales.to_dict(),
        "cohort_conditioned_behavioral_fidelity": round(mean(headline), 4)
        if headline
        else None,
        "schemes_with_result": sorted(
            (
                name
                for name, entry in results.items()
                if entry["cohort_conditioned_behavioral_fidelity"] is not None
                and entry["counted_cohorts"] >= 2
            )
        ),
        "skipped_schemes": skipped,
        "by_scheme": results,
    }
