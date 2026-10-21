"""L2: trajectory-level behavior distributions."""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
from evaluation.trace import ipv_visits
from ..stats import (
    bin_labels,
    bin_support,
    distribution,
    js_divergence,
    mean,
    quantile_bins,
    survival_curve,
    survival_distance,
)
from ..trace import RolloutTrace
from simulator.actions import ACTIONS


def _all_actions(traces: Sequence[RolloutTrace]) -> List[str]:
    return [step.action for trace in traces for step in trace.policy_steps]


def _session_equal_distribution(
    traces: Sequence[RolloutTrace], support: Sequence[str]
) -> dict:
    distributions = []
    for trace in traces:
        steps = trace.policy_steps
        if steps:
            distributions.append(distribution([step.action for step in steps], support))
    zero_policy_sessions = len(traces) - len(distributions)
    averaged = {
        action: sum((item[action] for item in distributions)) / len(distributions)
        if distributions
        else 0.0
        for action in support
    }
    return {
        "sessions": len(distributions),
        "zero_policy_action_sessions": zero_policy_sessions,
        "distribution": {
            action: round(value, 4) for action, value in averaged.items() if value
        },
        "raw_distribution": averaged,
    }


def action_distribution_divergence(
    real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace]
) -> dict:
    real_counts = _all_actions(real)
    sim_counts = _all_actions(sim)
    support = list(ACTIONS)
    real_dist = distribution(real_counts, support)
    sim_dist = distribution(sim_counts, support)
    real_session_equal = _session_equal_distribution(real, support)
    sim_session_equal = _session_equal_distribution(sim, support)
    session_equal_js = None
    if real_session_equal["sessions"] or sim_session_equal["sessions"]:
        session_equal_js = round(
            js_divergence(
                real_session_equal["raw_distribution"],
                sim_session_equal["raw_distribution"],
                support,
            ),
            4,
        )
    return {
        "real_steps": len(real_counts),
        "sim_steps": len(sim_counts),
        "js_divergence": round(js_divergence(real_dist, sim_dist, support), 4),
        "real_distribution": {
            action: round(value, 4) for action, value in real_dist.items() if value
        },
        "sim_distribution": {
            action: round(value, 4) for action, value in sim_dist.items() if value
        },
        "session_equal": {
            "js_divergence": session_equal_js,
            "real_sessions": real_session_equal["sessions"],
            "sim_sessions": sim_session_equal["sessions"],
            "real_zero_policy_action_sessions": real_session_equal[
                "zero_policy_action_sessions"
            ],
            "sim_zero_policy_action_sessions": sim_session_equal[
                "zero_policy_action_sessions"
            ],
            "real_distribution": real_session_equal["distribution"],
            "sim_distribution": sim_session_equal["distribution"],
        },
        "unused_actions": [
            action
            for action in support
            if real_dist.get(action, 0.0) > 0 and sim_dist.get(action, 0.0) == 0
        ],
        "hallucinated_actions": [
            action
            for action in support
            if sim_dist.get(action, 0.0) > 0 and real_dist.get(action, 0.0) == 0
        ],
    }


def _observed_termination(trace: RolloutTrace) -> bool:
    return bool(trace.terminated and (not trace.truncated) and (not trace.error))


def _transitions(traces: Sequence[RolloutTrace]) -> Dict[str, Dict[str, Counter]]:
    tables: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for trace in traces:
        steps = trace.policy_steps
        for current, following in zip(steps, steps[1:]):
            tables[current.layer][current.action][following.action] += 1
        if steps and _observed_termination(trace):
            last = steps[-1]
            tables[last.layer][last.action]["<stop>"] += 1
    return {layer: dict(rows) for layer, rows in tables.items()}


def layer_conditioned_transition_divergence(
    real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace], min_support: int = 5
) -> dict:
    real_tables = _transitions(real)
    sim_tables = _transitions(sim)
    support = list(ACTIONS) + ["<stop>"]
    layers: Dict[str, dict] = {}
    weighted_total = 0.0
    weight_sum = 0.0
    for layer in sorted(set(real_tables) | set(sim_tables)):
        real_rows = real_tables.get(layer, {})
        sim_rows = sim_tables.get(layer, {})
        rows: Dict[str, dict] = {}
        layer_weighted = 0.0
        layer_weight = 0.0
        for action in sorted(set(real_rows) | set(sim_rows)):
            real_row = real_rows.get(action, Counter())
            sim_row = sim_rows.get(action, Counter())
            real_n = sum(real_row.values())
            sim_n = sum(sim_row.values())
            divergence = js_divergence(
                distribution(real_row.elements(), support),
                distribution(sim_row.elements(), support),
                support,
            )
            counted = real_n >= min_support
            rows[action] = {
                "real_n": real_n,
                "sim_n": sim_n,
                "js_divergence": round(divergence, 4),
                "counted": counted,
            }
            if counted:
                layer_weighted += divergence * real_n
                layer_weight += real_n
        layer_divergence = layer_weighted / layer_weight if layer_weight else None
        layers[layer] = {
            "transition_divergence": round(layer_divergence, 4)
            if layer_divergence is not None
            else None,
            "counted_rows": sum((1 for row in rows.values() if row["counted"])),
            "rows": rows,
        }
        weighted_total += layer_weighted
        weight_sum += layer_weight
    scored = [
        layers[name]["transition_divergence"]
        for name in ("feed", "ipv")
        if layers.get(name, {}).get("transition_divergence") is not None
    ]
    return {
        "transition_divergence": round(weighted_total / weight_sum, 4)
        if weight_sum
        else None,
        "macro_layer_divergence": round(mean(scored), 4) if scored else None,
        "min_support": min_support,
        "by_layer": layers,
    }


def survival_curve_distance(
    real: Sequence[RolloutTrace],
    sim: Sequence[RolloutTrace],
    horizon: Optional[int] = None,
) -> dict:
    real_lengths = [len(trace.policy_steps) for trace in real]
    sim_lengths = [len(trace.policy_steps) for trace in sim]
    real_events = [_observed_termination(trace) for trace in real]
    sim_events = [_observed_termination(trace) for trace in sim]
    if horizon is None:
        horizon = max(real_lengths) if real_lengths else max(sim_lengths or [0])
    real_survival = survival_curve(real_lengths, horizon, real_events)
    sim_survival = survival_curve(sim_lengths, horizon, sim_events)

    def supported_through(lengths, events):
        if not lengths:
            return 0
        last = max(lengths)
        return (
            last
            if any((n == last and (not event) for n, event in zip(lengths, events)))
            else horizon
        )

    supported_horizon = min(
        horizon,
        supported_through(real_lengths, real_events),
        supported_through(sim_lengths, sim_events),
    )
    reason = None
    if not real_lengths or not sim_lengths or horizon <= 0:
        reason = "no comparable survival observations"
    elif supported_horizon < horizon:
        reason = "censoring leaves no risk-set support beyond step {} of {}".format(
            supported_horizon, horizon
        )
    return {
        "horizon": horizon,
        "supported_horizon": supported_horizon,
        "not_measured": reason,
        "survival_curve_distance": round(
            survival_distance(real_survival, sim_survival), 4
        )
        if reason is None
        else None,
        "trajectory_error": round(abs(mean(real_lengths) - mean(sim_lengths)), 2)
        if real_lengths and sim_lengths else None,
        "real_mean_length": round(mean([float(v) for v in real_lengths]), 2),
        "sim_mean_length": round(mean([float(v) for v in sim_lengths]), 2),
        "real_event_observed_count": sum(real_events),
        "sim_event_observed_count": sum(sim_events),
        "real_event_observed_rate": round(
            mean([float(value) for value in real_events]), 4
        ),
        "sim_event_observed_rate": round(
            mean([float(value) for value in sim_events]), 4
        ),
        "real_censored_count": sum((not value for value in real_events)),
        "sim_censored_count": sum((not value for value in sim_events)),
        "real_censored_rate": round(
            mean([float(not value) for value in real_events]), 4
        ),
        "sim_censored_rate": round(mean([float(not value) for value in sim_events]), 4),
        "real_truncated_rate": round(
            mean([float(trace.truncated) for trace in real]), 4
        ),
        "sim_truncated_rate": round(mean([float(trace.truncated) for trace in sim]), 4),
        "sim_over_horizon_rate": round(
            mean([1.0 if length > horizon else 0.0 for length in sim_lengths]), 4
        ),
        "real_survival": [
            round(value, 4)
            if i < supported_through(real_lengths, real_events)
            else None
            for i, value in enumerate(real_survival)
        ],
        "sim_survival": [
            round(value, 4) if i < supported_through(sim_lengths, sim_events) else None
            for i, value in enumerate(sim_survival)
        ],
    }


def _depth_samples(traces: Sequence[RolloutTrace]) -> Tuple[List[int], List[str], dict]:
    depths: List[int] = []
    decisions: List[str] = []
    channels: Counter = Counter()
    visits = 0
    for trace in traces:
        for visit in ipv_visits(trace):
            visits += 1
            depths.append(visit.info_to_decision)
            decisions.append(visit.decision)
            for channel, count in visit.channel_counts.items():
                channels[channel] += count
    per_visit = {
        channel: channels[channel] / visits if visits else 0.0 for channel in channels
    }
    return (depths, decisions, per_visit)


def information_acquisition_depth_distance(
    real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace], bins: int = 4
) -> dict:
    real_depths, real_decisions, real_channels = _depth_samples(real)
    sim_depths, sim_decisions, sim_channels = _depth_samples(sim)
    edges = quantile_bins([float(value) for value in real_depths], bins)
    support = bin_support(edges)
    depth_divergence = js_divergence(
        distribution(bin_labels([float(v) for v in real_depths], edges), support),
        distribution(bin_labels([float(v) for v in sim_depths], edges), support),
        support,
    )
    decision_support = ["return", "cart", "buy", "end", "open"]
    decision_divergence = js_divergence(
        distribution(real_decisions, decision_support),
        distribution(sim_decisions, decision_support),
        decision_support,
    )
    return {
        "real_visits": len(real_depths),
        "sim_visits": len(sim_depths),
        "information_acquisition_depth_distance": round(depth_divergence, 4),
        "decision_mix_divergence": round(decision_divergence, 4),
        "bin_edges": [round(edge, 2) for edge in edges],
        "real_mean_depth": round(mean([float(v) for v in real_depths]), 2),
        "sim_mean_depth": round(mean([float(v) for v in sim_depths]), 2),
        "real_channel_per_visit": {
            key: round(value, 3) for key, value in sorted(real_channels.items())
        },
        "sim_channel_per_visit": {
            key: round(value, 3) for key, value in sorted(sim_channels.items())
        },
        "real_decision_mix": {
            key: round(value, 4)
            for key, value in distribution(real_decisions, decision_support).items()
        },
        "sim_decision_mix": {
            key: round(value, 4)
            for key, value in distribution(sim_decisions, decision_support).items()
        },
    }


def evaluate(
    real: Sequence[RolloutTrace],
    sim: Sequence[RolloutTrace],
    min_transition_support: int = 5,
    depth_bins: int = 4,
) -> dict:
    return {
        "action_distribution_divergence": action_distribution_divergence(real, sim),
        "layer_conditioned_transition_divergence": layer_conditioned_transition_divergence(
            real, sim, min_support=min_transition_support
        ),
        "survival_curve_distance": survival_curve_distance(real, sim),
        "information_acquisition_depth_distance": information_acquisition_depth_distance(
            real, sim, bins=depth_bins
        ),
    }
