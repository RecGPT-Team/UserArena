"""Evaluate traces and write metric, validity and cost reports."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import stdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from . import reference
from .metrics import l1, l2, l3, l4
from .stats import mean
from .trace import ReplayStep, ReplayTrace, RolloutStep, RolloutTrace, write_rollout_traces


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def validity_summary(steps: Sequence[Union[RolloutStep, ReplayStep]]) -> dict:
    total = len(steps)
    return {
        "outputs": total,
        "parser_success_rate": round(
            _rate(sum((1 for step in steps if step.parse_ok)), total), 4
        ),
        "valid_action_rate": round(
            _rate(sum((1 for step in steps if step.executable)), total), 4
        ),
    }


def _token_summary(input_total, output_total, recorded, outputs, episodes=None, unit="step"):
    total = input_total + output_total
    result = {
        f"token_instrumented_{unit}s": recorded,
        "token_usage_coverage": round(recorded / outputs, 4) if outputs else None,
        "total_tokens": total if recorded else None,
        f"tokens_per_{unit}": round(total / outputs, 1) if recorded and outputs else None,
    }
    if episodes is not None:
        result["tokens_per_episode"] = (
            round(total / episodes, 1) if recorded and episodes else None
        )
    return result


def _step_rows(traces: Sequence[RolloutTrace]) -> List[dict]:
    rows = []
    for trace in traces:
        pending = []
        position = 1
        for index, step in enumerate(trace.steps):
            pending.append(step.usage)
            if not step.executed and index < len(trace.steps) - 1:
                continue
            tokens_recorded = all(u.input_tokens or u.output_tokens for u in pending)
            inputs = sum(u.input_tokens for u in pending)
            outputs = sum(u.output_tokens for u in pending)
            rows.append({
                "position": position,
                "input_tokens": inputs if tokens_recorded else None,
                "output_tokens": outputs if tokens_recorded else None,
                "total_tokens": inputs + outputs if tokens_recorded else None,
                "llm_calls": sum(u.llm_calls for u in pending)
                if all(u.llm_calls for u in pending) else None,
                "latency_ms": sum(u.latency_ms for u in pending)
                if all(u.latency_ms is not None for u in pending) else None,
            })
            pending = []
            position += 1
    return rows


def _profile(rows: Sequence[dict], max_steps: Optional[int] = None) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        if max_steps is None or row["position"] <= max_steps:
            grouped[row["position"]].append(row)
    profile = {}
    for position, bucket in sorted(grouped.items()):
        result = {"n": len(bucket)}
        for field in ("input_tokens", "output_tokens", "total_tokens", "llm_calls", "latency_ms"):
            values = [row[field] for row in bucket if row[field] is not None]
            result[field + "_n"] = len(values)
            result[field + "_mean"] = round(mean(values), 1) if values else None
            result[field + "_std"] = round(stdev(values), 1) if len(values) > 1 else None
        profile[str(position)] = result
    return profile


def rollout_efficiency(
    traces: Sequence[RolloutTrace], max_profile_steps: Optional[int] = None
) -> dict:
    steps = [step for trace in traces for step in trace.steps]
    input_total = sum(step.usage.input_tokens for step in steps)
    output_total = sum(step.usage.output_tokens for step in steps)
    calls = sum(step.usage.llm_calls for step in steps)
    recorded = sum(bool(step.usage.input_tokens or step.usage.output_tokens) for step in steps)
    return {
        **_token_summary(input_total, output_total, recorded, len(steps), len(traces), unit="output"),
        "episodes": len(traces),
        "outputs": len(steps),
        "executed_actions": sum(len(trace.policy_steps) for trace in traces),
        "total_input_tokens": input_total if recorded else None,
        "total_output_tokens": output_total if recorded else None,
        "total_llm_calls": calls or None,
        "profile_unit": "next_executed_action_index",
        "by_step_index": _profile(_step_rows(traces), max_profile_steps),
    }


def replay_efficiency(trace: ReplayTrace) -> dict:
    steps = trace.steps
    episodes = (
        len({step.session_id for step in steps})
        if steps and all((step.session_id for step in steps))
        else None
    )
    latencies = [
        step.usage.latency_ms for step in steps if step.usage.latency_ms is not None
    ]
    input_total = sum((step.usage.input_tokens for step in steps))
    output_total = sum((step.usage.output_tokens for step in steps))
    return {
        **_token_summary(
            input_total,
            output_total,
            sum(
                (
                    bool(step.usage.input_tokens or step.usage.output_tokens)
                    for step in steps
                )
            ),
            len(steps),
            episodes,
        ),
        "episodes": episodes,
        "input_tokens_per_episode": round(input_total / episodes, 1)
        if episodes
        else None,
        "output_tokens_per_episode": round(output_total / episodes, 1)
        if episodes
        else None,
        "llm_calls_per_episode": round(
            sum((step.usage.llm_calls for step in steps)) / episodes, 2
        )
        if episodes
        else None,
        "call_instrumented_steps": sum((step.usage.llm_calls > 0 for step in steps)),
        "wall_clock_ms_per_step": None,
        "wall_clock_ms_per_episode": None,
        "wall_clock_not_measured": "saved trace has no inference start/end timestamps; summed call latency is not wall-clock time",
        "steps": len(steps),
        "instrumented_steps": sum(
            (1 for step in steps if step.usage.input_tokens or step.usage.output_tokens)
        ),
        "total_input_tokens": input_total,
        "total_output_tokens": output_total,
        "total_llm_calls": sum((step.usage.llm_calls for step in steps)),
        "input_tokens_per_step": round(input_total / len(steps), 1) if steps else 0.0,
        "output_tokens_per_step": round(output_total / len(steps), 1) if steps else 0.0,
        "latency_ms_per_step": round(mean(latencies), 1) if latencies else None,
    }


HEADLINE = {
    "replay": (
        ("RVA", "L0", ("l0", "valid_action_rate"), "higher"),
        ("F1", "L1", ("l1", "macro_action_f1", "macro_action_f1"), "higher"),
        ("Acc", "L1", ("l1", "exact_action_accuracy", "non_scroll_macro_exact_action_accuracy"), "higher"),
    ),
    "rollout": (
        ("OVA", "L0", ("l0", "valid_action_rate"), "higher"),
        ("DM-JSD", "L2", ("l2", "information_acquisition_depth_distance", "decision_mix_divergence"), "lower"),
        ("TE", "L2", ("l2", "survival_curve_distance", "trajectory_error"), "lower"),
        ("PF", "L3", ("l3", "cohort_conditioned_behavioral_fidelity"), "higher"),
        ("ΔCTR", "L4", ("l4", "funnel", "count_rates", "ctr", "abs_diff"), "lower"),
        ("ΔACR", "L4", ("l4", "funnel", "count_rates", "acr", "abs_diff"), "lower"),
        ("T-JSD", "L4", ("l4", "terminal_outcome", "terminal_outcome_distribution_divergence"), "lower"),
    ),
}


def headline(payload: dict, paradigm: str) -> dict:
    summary = {}
    for name, layer, path, direction in HEADLINE[paradigm]:
        value = payload
        for key in path:
            value = value[key]
        if (layer == "L0" or paradigm == "replay") and not payload["l0"]["outputs"]:
            value = None
        summary[name] = {"layer": layer, "value": value, "direction": direction}
    return summary


def rollout_report(
    real: Sequence[RolloutTrace],
    sim: Sequence[RolloutTrace],
    min_cohort: int = l3.DEFAULT_MIN_COHORT,
    cohort_schemes: Optional[Sequence[str]] = None,
    max_profile_steps: Optional[int] = None,
) -> dict:
    payload = {
        "metric_settings": {
            "min_cohort": min_cohort,
            "cohort_schemes": cohort_schemes,
            "max_profile_steps": max_profile_steps,
        },
        "paradigm": "rollout",
        "real_episodes": len(real),
        "sim_episodes": len(sim),
        "l0": validity_summary([step for trace in sim for step in trace.steps]),
        "l2": l2.evaluate(real, sim),
        "l3": l3.evaluate(real, sim, schemes=cohort_schemes, min_cohort=min_cohort),
        "l4": l4.evaluate(real, sim),
        "efficiency": rollout_efficiency(sim, max_profile_steps),
    }
    payload["headline"] = headline(payload, "rollout")
    return payload


def replay_report(trace: ReplayTrace, coverage: Optional[dict] = None) -> dict:
    payload = {
        "paradigm": "replay",
        "steps": len(trace.steps),
        "l0": validity_summary(trace.steps),
        "l1": l1.evaluate_replay(trace),
        "efficiency": replay_efficiency(trace),
    }
    if coverage is not None:
        payload["trace_coverage"] = coverage
    payload["headline"] = headline(payload, "replay")
    return payload


def write_report(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ground_truth_audit(
    audit: Dict[str, dict],
    scored: Sequence[RolloutTrace],
    loader_stats: Optional[Dict[str, int]] = None,
) -> dict:
    from .reference import FUNNEL_EVENTS

    ids = {trace.session_id for trace in scored}
    rows = [row for key, row in audit.items() if key in ids]
    excluded = {
        "sessions_excluded_whole": (loader_stats or {}).get("dropped_sessions", 0),
        "sessions_seen_by_loader": (loader_stats or {}).get("sessions", 0),
        "excluded_by_defect": {
            reason: count
            for reason, count in sorted((loader_stats or {}).items())
            if reason not in ("sessions", "dropped_sessions") and count
        },
    }
    if not rows:
        return dict({"sessions": 0}, **excluded)
    dropped: Dict[str, int] = {}
    for row in rows:
        for reason, count in row.get("dropped", {}).items():
            dropped[reason] = dropped.get(reason, 0) + count
    sessions = len(rows)
    funnel_before: Dict[str, float] = {}
    funnel_after: Dict[str, float] = {}
    funnel_shift: Dict[str, float] = {}
    for event in FUNNEL_EVENTS:
        hit_before = sum((1 for row in rows if row["funnel_before"][event]))
        hit_after = sum((1 for row in rows if row["funnel_after"][event]))
        funnel_before[event] = round(hit_before / sessions, 4)
        funnel_after[event] = round(hit_after / sessions, 4)
        funnel_shift[event] = round((hit_after - hit_before) / sessions, 4)
    return dict(
        {
            "sessions": sessions,
            "dropped_steps": sum(dropped.values()),
            "dropped_by_reason": dict(sorted(dropped.items())),
            "raw_steps": sum((row["raw_steps"] for row in rows)),
            "kept_steps": sum((row["kept_steps"] for row in rows)),
            "clicks_before": sum((row["clicks_before"] for row in rows)),
            "clicks_after": sum((row["clicks_after"] for row in rows)),
            "funnel_before": funnel_before,
            "funnel_after": funnel_after,
            "funnel_shift": funnel_shift,
            "ctr_before": funnel_before["click"],
            "ctr_after": funnel_after["click"],
            "ctr_shift": funnel_shift["click"],
            "clicks_without_ipv": sum((row["clicks_without_ipv"] for row in rows)),
            "clicks_without_ipv_via_back_home": sum(
                (row["clicks_without_ipv_via_back_home"] for row in rows)
            ),
            "sessions_without_ipv_step": sum(
                (1 for row in rows if not row["has_ipv_step"])
            ),
        },
        **excluded,
    )


def rollout_failure_audit(sim: Sequence[RolloutTrace]) -> dict:
    failed = [trace for trace in sim if trace.error]
    if not sim:
        return {"episodes": 0, "failed": 0}
    by_error: Dict[str, int] = {}
    for trace in failed:
        by_error[trace.error] = by_error.get(trace.error, 0) + 1
    return {
        "episodes": len(sim),
        "failed": len(failed),
        "failure_rate": round(len(failed) / len(sim), 4),
        "failed_by_error": dict(sorted(by_error.items())),
        "failed_without_steps": sum((1 for trace in failed if not trace.steps)),
    }


def _assign_rollout_indices(traces: Sequence[RolloutTrace]) -> None:
    by_session: Dict[str, List[RolloutTrace]] = defaultdict(list)
    for trace in traces:
        by_session[trace.session_id].append(trace)
    for session_id, session_traces in by_session.items():
        explicit = [
            trace.rollout_index for trace in session_traces if trace.rollout_index != 0
        ]
        if len(explicit) != len(set(explicit)):
            raise ValueError(f"duplicate explicit rollout_index for {session_id!r}")
        used = set(explicit)
        candidate = 0
        for trace in session_traces:
            if trace.rollout_index != 0:
                continue
            while candidate in used:
                candidate += 1
            trace.rollout_index = candidate
            used.add(candidate)
            candidate += 1


def convert_rollout_traces(
    payloads: Iterable[Mapping[str, object]], model: str = ""
) -> List[RolloutTrace]:
    traces: List[RolloutTrace] = []
    for payload in payloads:
        trace = RolloutTrace.from_dict(dict(payload))
        trace.source = "sim"
        if model:
            trace.model = model
        traces.append(trace)
    _assign_rollout_indices(traces)
    return traces


def run_rollout_evaluation(
    simulated_trace_dicts: Iterable[Mapping[str, object]],
    held_out_sessions_path: Path,
    output_dir: Path,
    model: str,
    min_cohort: int = 30,
    cohort_schemes: Optional[Sequence[str]] = None,
    max_profile_steps: Optional[int] = None,
    provenance: Optional[Mapping[str, object]] = None,
) -> dict:
    model_label = model.strip()
    simulated = convert_rollout_traces(simulated_trace_dicts, model=model_label)
    if not simulated:
        raise ValueError("no simulated traces were provided")
    audit: Dict[str, dict] = {}
    loader_stats: Dict[str, int] = {}
    real = list(
        reference.real_traces(
            Path(held_out_sessions_path), audit=audit, loader_stats=loader_stats
        )
    )
    rolled_out_session_ids = {trace.session_id for trace in simulated}
    real_ids = {trace.session_id for trace in real}
    if rolled_out_session_ids - real_ids:
        raise ValueError("Some simulated sessions have no eligible reference")
    reference.validate_cohort_fields(
        held_out_sessions_path, cohort_schemes, rolled_out_session_ids
    )
    scoped_real = [
        trace for trace in real if trace.session_id in rolled_out_session_ids
    ]
    schemes = tuple(cohort_schemes) if cohort_schemes is not None else None
    payload = rollout_report(
        scoped_real,
        simulated,
        min_cohort=min_cohort,
        cohort_schemes=schemes,
        max_profile_steps=max_profile_steps,
    )
    payload["ground_truth_audit"] = ground_truth_audit(
        audit, scoped_real, loader_stats
    )
    payload["rollout_failures"] = rollout_failure_audit(simulated)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_rollout_traces(destination / 'traces.jsonl', list(scoped_real) + simulated)
    write_report(destination / 'report.json', payload)
    write_report(destination / 'headline.json', payload["headline"])
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_label,
        "real_episodes": len(scoped_real),
        "sim_episodes": len(simulated),
        "rolled_out_sessions": len(rolled_out_session_ids),
        "min_cohort": min_cohort,
        "cohort_schemes": list(schemes) if schemes is not None else None,
        "max_profile_steps": max_profile_steps,
        "files": {
            "traces": 'traces.jsonl',
            "report": 'report.json',
            "headline": 'headline.json',
        },
    }
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    write_report(destination / 'manifest.json', manifest)
    return payload
