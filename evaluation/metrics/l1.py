"""L1: decisions at logged replay states."""

from __future__ import annotations
from collections import Counter
from typing import Dict, Sequence
from ..stats import macro_f1, mean, reciprocal_rank
from ..trace import ReplayStep, ReplayTrace

SCROLL_ACTIONS = frozenset(
    ("scroll_down", "scroll_up", "ipv_scroll_down", "ipv_scroll_up")
)


def _valid_prediction(step: ReplayStep) -> bool:
    return bool(
        step.prediction_present and step.executable and (not step.reject_reason)
    )


def _action_label(step: ReplayStep) -> str:
    return step.pred_action or "" if _valid_prediction(step) else ""


def macro_action_f1(steps: Sequence[ReplayStep]) -> dict:
    pairs = [(step.gt_action, _action_label(step)) for step in steps]
    per_class, macro = macro_f1(pairs)
    correct = sum((1 for truth, pred in pairs if truth == pred))
    by_layer: Dict[str, dict] = {}
    for layer in sorted({step.layer for step in steps}):
        subset = [
            (step.gt_action, _action_label(step))
            for step in steps
            if step.layer == layer
        ]
        layer_classes, layer_macro = macro_f1(subset)
        by_layer[layer] = {
            "steps": len(subset),
            "action_type_accuracy": round(
                sum((1 for truth, pred in subset if truth == pred)) / len(subset), 4
            )
            if subset
            else 0.0,
            "macro_action_f1": round(layer_macro, 4),
            "per_class": layer_classes,
        }
    return {
        "steps": len(steps),
        "macro_action_f1": round(macro, 4),
        "action_type_accuracy": round(correct / len(steps), 4) if steps else 0.0,
        "per_class": per_class,
        "by_layer": by_layer,
        "ground_truth_distribution": dict(Counter((truth for truth, _ in pairs))),
        "prediction_distribution": dict(
            Counter((pred or "<unparsed>" for _, pred in pairs))
        ),
    }


def _same_text(left: object, right: object) -> bool:
    left = str(left or "").strip()
    right = str(right or "").strip()
    return bool(left and left == right)


def exact_action_accuracy(steps: Sequence[ReplayStep]) -> dict:
    exact_hits = 0
    class_support = Counter((step.gt_action for step in steps))
    class_hits: Counter[str] = Counter()
    click_id_compared = click_id_matches = 0
    click_title_fallback_compared = click_title_fallback_matches = 0
    comment_alias_compared = comment_alias_matches = 0
    target_missing = type_only_matches = 0
    invalid_predictions = 0
    for step in steps:
        if not _valid_prediction(step):
            invalid_predictions += 1
            continue
        type_match = step.gt_action == (step.pred_action or "")
        if step.gt_action == "click":
            if step.gt_target_item_id and step.pred_target_item_id:
                click_id_compared += 1
                matched = type_match and str(step.gt_target_item_id) == str(
                    step.pred_target_item_id
                )
                click_id_matches += int(matched)
                exact_hits += int(matched)
                class_hits[step.gt_action] += int(matched)
            elif step.gt_target_title and step.pred_target_title:
                click_title_fallback_compared += 1
                matched = type_match and _same_text(
                    step.gt_target_title, step.pred_target_title
                )
                click_title_fallback_matches += int(matched)
                exact_hits += int(matched)
                class_hits[step.gt_action] += int(matched)
            else:
                target_missing += 1
        elif step.gt_action == "ipv_click_comment":
            if step.gt_target_alias and step.pred_target_alias:
                comment_alias_compared += 1
                matched = type_match and str(step.gt_target_alias) == str(
                    step.pred_target_alias
                )
                comment_alias_matches += int(matched)
                exact_hits += int(matched)
                class_hits[step.gt_action] += int(matched)
            else:
                target_missing += 1
        else:
            type_only_matches += int(type_match)
            exact_hits += int(type_match)
            class_hits[step.gt_action] += int(type_match)
    class_rates = {
        action: class_hits[action] / support
        for action, support in sorted(class_support.items())
    }
    non_scroll_rates = [
        rate for action, rate in class_rates.items() if action not in SCROLL_ACTIONS
    ]
    return {
        "steps": len(steps),
        "exact_action_accuracy": round(exact_hits / len(steps), 4) if steps else 0.0,
        "macro_exact_action_accuracy": round(mean(list(class_rates.values())), 4)
        if class_rates
        else None,
        "non_scroll_macro_exact_action_accuracy": round(mean(non_scroll_rates), 4)
        if non_scroll_rates
        else None,
        "per_class": {
            action: {
                "support": support,
                "exact_hits": class_hits[action],
                "exact_accuracy": round(class_rates[action], 4),
            }
            for action, support in sorted(class_support.items())
        },
        "invalid_predictions": invalid_predictions,
        "target_coverage": {
            "click_id_compared": click_id_compared,
            "click_title_fallback_compared": click_title_fallback_compared,
            "comment_alias_compared": comment_alias_compared,
            "target_missing_or_unresolved": target_missing,
        },
        "target_match_source": {
            "click_id": click_id_matches,
            "click_title_fallback": click_title_fallback_matches,
            "comment_alias": comment_alias_matches,
            "action_type_only": type_only_matches,
        },
    }


def _mrr_figures(click_steps: Sequence[ReplayStep]) -> dict:
    ranked = [step for step in click_steps if step.ranked_candidates]
    ranks = [
        reciprocal_rank(step.ranked_candidates, str(step.gt_target_item_id))
        for step in ranked
    ]
    hits_at = {}
    for cutoff in (1, 3, 5):
        hits_at["hit@{}".format(cutoff)] = round(
            mean(
                [
                    1.0
                    if str(step.gt_target_item_id)
                    in [str(item) for item in step.ranked_candidates[:cutoff]]
                    else 0.0
                    for step in ranked
                ]
            ),
            4,
        )
    missing_target = sum(
        (
            1
            for step in click_steps
            if str(step.gt_target_item_id) not in [str(v) for v in step.candidates]
        )
    )
    degenerate = 0
    for step in ranked:
        values = list(step.candidate_scores.values())
        if len(values) > 1 and max(values) - min(values) < 1e-09:
            degenerate += 1
    without_candidates = sum((1 for step in click_steps if not step.candidates))
    return {
        "click_steps": len(click_steps),
        "ranked_steps": len(ranked),
        "unranked_steps": len(click_steps) - len(ranked),
        "mrr": round(mean(ranks), 4),
        "mrr_including_unranked": round(sum(ranks) / len(click_steps), 4)
        if click_steps
        else 0.0,
        "mean_candidates": round(
            mean([float(len(step.candidates)) for step in click_steps]), 2
        ),
        "target_missing_from_candidates": missing_target,
        "steps_without_candidates": without_candidates,
        "degenerate_rankings": degenerate,
        **hits_at,
    }


def exposure_conditioned_mrr(steps: Sequence[ReplayStep]) -> dict:
    click_steps = [
        step for step in steps if step.gt_action == "click" and step.gt_target_item_id
    ]
    ranked = [step for step in click_steps if step.ranked_candidates]
    scorers = sorted({step.scorer for step in ranked if step.scorer})
    by_scorer = (
        {
            scorer: _mrr_figures(
                [step for step in click_steps if step.scorer == scorer]
            )
            for scorer in scorers
        }
        if len(scorers) > 1
        else {}
    )
    return {**_mrr_figures(click_steps), "scorers": scorers, "by_scorer": by_scorer}


def evaluate_replay(trace: ReplayTrace) -> dict:
    return {
        "macro_action_f1": macro_action_f1(trace.steps),
        "exact_action_accuracy": exact_action_accuracy(trace.steps),
        "exposure_conditioned_item_mrr": exposure_conditioned_mrr(trace.steps),
    }
