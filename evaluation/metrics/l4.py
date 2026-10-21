"""L4: conversion funnels and terminal outcomes."""

from __future__ import annotations
from typing import Dict, Sequence
from evaluation.trace import TERMINAL_OUTCOMES, ipv_visits, terminal_outcome
from ..stats import distribution, js_divergence, mean
from ..trace import RolloutTrace

_RATE_NAMES = ("ctr", "ipvr", "acr", "cvr")


def _funnel(traces: Sequence[RolloutTrace]) -> dict:
    if not traces:
        return {
            "sessions": 0,
            "incidence": {name: 0.0 for name in _RATE_NAMES},
            "count_rates": {name: 0.0 for name in _RATE_NAMES},
            "counts": {
                name: 0
                for name in ("feed_actions", "clicks", "ipv_visits", "carts", "buys")
            },
        }
    clicked = ipv = cart = buy = 0
    feed_steps = clicks = visits = carts = buys = 0
    for trace in traces:
        actions = trace.actions()
        action_set = set(actions)
        clicked += int("click" in action_set)
        ipv += int(any((step.layer == "ipv" for step in trace.executed_steps)))
        cart += int("ipv_cart" in action_set)
        buy += int("ipv_buy" in action_set)
        feed_steps += sum((1 for step in trace.executed_steps if step.layer == "feed"))
        clicks += sum((1 for action in actions if action == "click"))
        visits += len(ipv_visits(trace))
        carts += sum((1 for action in actions if action == "ipv_cart"))
        buys += sum((1 for action in actions if action == "ipv_buy"))
    total = len(traces)
    return {
        "sessions": total,
        "incidence": {
            "ctr": clicked / total,
            "ipvr": ipv / total,
            "acr": cart / total,
            "cvr": buy / total,
        },
        "count_rates": {
            "ctr": clicks / feed_steps if feed_steps else 0.0,
            "ipvr": visits / clicks if clicks else 0.0,
            "acr": carts / clicks if clicks else 0.0,
            "cvr": buys / clicks if clicks else 0.0,
        },
        "counts": {
            "feed_actions": feed_steps,
            "clicks": clicks,
            "ipv_visits": visits,
            "carts": carts,
            "buys": buys,
        },
    }


def _compare(real: dict, sim: dict) -> Dict[str, dict]:
    return {
        name: {
            "real": round(real[name], 4),
            "sim": round(sim[name], 4),
            "abs_diff": round(abs(sim[name] - real[name]), 4),
            "relative_gap": round(abs(sim[name] - real[name]) / abs(real[name]), 4)
            if real[name] != 0
            else None,
        }
        for name in _RATE_NAMES
    }


def funnel_rates(real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace]) -> dict:
    real_funnel = _funnel(real)
    sim_funnel = _funnel(sim)
    comparison = _compare(real_funnel["incidence"], sim_funnel["incidence"])
    count_comparison = _compare(real_funnel["count_rates"], sim_funnel["count_rates"])
    session_errors = [comparison[name]["abs_diff"] for name in _RATE_NAMES]
    count_errors = [count_comparison[name]["abs_diff"] for name in _RATE_NAMES]
    comparable = bool(real_funnel["sessions"]) and bool(sim_funnel["sessions"])
    return {
        "real_sessions": real_funnel["sessions"],
        "sim_sessions": sim_funnel["sessions"],
        "funnel_mean_abs_error": round(mean(session_errors + count_errors), 4)
        if comparable
        else None,
        "session_funnel_mean_abs_error": round(mean(session_errors), 4)
        if comparable
        else None,
        "count_funnel_mean_abs_error": round(mean(count_errors), 4)
        if comparable
        else None,
        "rates": {**comparison, "click_per_feed_step": count_comparison["ctr"]},
        "count_rates": count_comparison,
        "counts": {"real": real_funnel["counts"], "sim": sim_funnel["counts"]},
    }


def completed_funnel_rates(
    real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace]
) -> dict:
    completed = [trace for trace in sim if not trace.error]
    kept_ids = {trace.session_id for trace in completed}
    lost_ids = {trace.session_id for trace in sim if trace.error} - kept_ids
    matched = [trace for trace in real if trace.session_id not in lost_ids]
    payload = funnel_rates(matched, completed)
    payload["excluded_episodes"] = len(sim) - len(completed)
    payload["excluded_sessions"] = len(lost_ids)
    return payload


def terminal_outcome_divergence(
    real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace]
) -> dict:
    support = list(TERMINAL_OUTCOMES)
    real_dist = distribution([terminal_outcome(trace) for trace in real], support)
    sim_dist = distribution([terminal_outcome(trace) for trace in sim], support)
    return {
        "terminal_outcome_distribution_divergence": round(
            js_divergence(real_dist, sim_dist, support), 4
        ),
        "real_distribution": {key: round(value, 4) for key, value in real_dist.items()},
        "sim_distribution": {key: round(value, 4) for key, value in sim_dist.items()},
    }


def evaluate(real: Sequence[RolloutTrace], sim: Sequence[RolloutTrace]) -> dict:
    return {
        "funnel": funnel_rates(real, sim),
        "funnel_completed": completed_funnel_rates(real, sim),
        "terminal_outcome": terminal_outcome_divergence(real, sim),
    }
