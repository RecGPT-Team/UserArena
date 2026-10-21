"""Normalize logged decisions and audit excluded reference events."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence
from simulator.data import Session
from simulator.replay import feed_click_reaches_ipv
from .trace import UserProfile, RolloutStep, RolloutTrace, parse_price
from simulator.actions import layer_of


def iter_sessions(path, max_sessions=None, stats=None):
    from simulator.session import load_records, cross_session_visits

    counts = stats if stats is not None else {}
    seen = set()
    yielded = 0
    for raw in load_records(path):
        visits = cross_session_visits(raw) if "pv_ids" in raw else [raw]
        for visit in visits:
            session = Session.from_raw(visit)
            if not session.session_id or session.session_id in seen:
                raise ValueError("Reference session IDs must be nonempty and unique")
            seen.add(session.session_id)
            counts["sessions"] = counts.get("sessions", 0) + 1
            if any(session.structural_defects.values()):
                counts["dropped_sessions"] = counts.get("dropped_sessions", 0) + 1
                for reason, count in session.structural_defects.items():
                    if count:
                        counts[reason] = counts.get(reason, 0) + count
                continue
            yield session
            yielded += 1
            if max_sessions is not None and yielded >= max_sessions:
                return


def validate_cohort_fields(path, schemes=None, session_ids=None):
    """Detect history fields removed by an incompatible dataset export."""
    from evaluation.metrics.l3 import DEFAULT_SCHEMES

    if "category_breadth" not in (schemes or DEFAULT_SCHEMES):
        return
    for session in iter_sessions(path):
        if session_ids is not None and session.session_id not in session_ids:
            continue
        history = session.user_click_list
        if any("cate" not in item for item in history):
            raise ValueError(
                "L3 category_breadth requires user_click_list[].cate; "
                "re-export the dataset with history categories retained, or explicitly "
                "select different cohort schemes for a nonstandard evaluation."
            )


def profile_from_session(session: Session) -> UserProfile:
    counts = session.history_counts or {}
    clicks = int(counts.get("clicks", len(session.user_click_list)))
    buys = int(counts.get("buys", len(session.user_buy_list)))
    categories = [
        str(item.get("cate") or "")
        for item in session.user_click_list
        if item.get("cate")
    ]
    return UserProfile(
        visitor_id=str(session.visitor_id or ""),
        click_history=clicks,
        buy_history=buys,
        history_price_mean=None,
        history_categories=categories,
        attributes=dict(session.user_info or {}),
    )


def _item_facts(session: Session, item_id: Optional[str]) -> "tuple":
    if not item_id:
        return (None, "")
    item = session.get_item(item_id)
    if item is None:
        return (None, "")
    return (
        parse_price(getattr(item, "item_price", None)),
        str(getattr(item, "item_cate", "") or ""),
    )


def real_trace(session: Session) -> RolloutTrace:
    steps: List[RolloutStep] = []
    current_item_id: Optional[str] = None
    policy_trajectory = [
        step
        for step in session.gt_trajectory
        if step.action
    ]
    for position, step in enumerate(policy_trajectory, 1):
        layer = (
            "ipv"
            if step.screen_type == "ipv"
            else "feed"
            if step.screen_type == "feed"
            else layer_of(step.action)
        )
        if step.action == "click" and step.clicked_item_id:
            current_item_id = str(step.clicked_item_id)
        item_id = (
            str(step.clicked_item_id)
            if step.clicked_item_id
            else str(step.item_id)
            if step.item_id
            else None
        )
        if layer == "ipv" and (not item_id):
            item_id = current_item_id
        price, category = _item_facts(session, item_id)
        steps.append(
            RolloutStep(
                index=position,
                layer=layer,
                action=step.action,
                executed=True,
                target_item_id=item_id,
                target_title=(step.item_title or (session.get_item(step.item_id).item_title
                              if step.action != "end" and step.item_id and session.get_item(step.item_id) else "")),
                target_alias=(f"C{step.review_index + 1}" if step.review_index is not None else None),
                item_price=price,
                item_cate=category,
                visible_item_ids=list(step.visible_items),
            )
        )
    return RolloutTrace(
        session_id=session.session_id,
        source="real",
        rollout_index=0,
        visitor_id=str(session.visitor_id or ""),
        profile=profile_from_session(session),
        steps=steps,
        terminated=True,
        truncated=False,
        termination_reason=session.termination_reason or "",
        max_steps=len(steps),
        model="ground_truth",
    )


def real_traces(
    path: Path,
    max_sessions: Optional[int] = None,
    audit: Optional[Dict[str, dict]] = None,
    loader_stats: Optional[Dict[str, int]] = None,
) -> Iterator[RolloutTrace]:
    for session in iter_sessions(
        Path(path), max_sessions=max_sessions, stats=loader_stats
    ):
        if audit is not None:
            audit[session.session_id] = session_audit(session)
        yield real_trace(session)


def session_audit(session: Session) -> dict:
    raw = session.trajectory
    clicks_before = sum(
        (1 for step in raw if step.screen_type == "feed" and step.action == "click")
    )
    clicks_after = sum((1 for step in session.gt_trajectory if step.action == "click"))
    without_ipv, via_back_home = _clicks_without_ipv(raw)
    before = _funnel_flags(raw)
    after = _funnel_flags(session.gt_trajectory)
    return {
        "dropped": dict(session.dropped),
        "raw_steps": len(raw),
        "kept_steps": len(session.gt_trajectory),
        "clicks_before": clicks_before,
        "clicks_after": clicks_after,
        "funnel_before": before,
        "funnel_after": after,
        "clicks_without_ipv": without_ipv,
        "clicks_without_ipv_via_back_home": via_back_home,
        "has_ipv_step": any((step.screen_type == "ipv" for step in raw)),
    }


FUNNEL_EVENTS = ("click", "ipv", "cart", "buy")


def _funnel_flags(steps: Sequence) -> Dict[str, bool]:
    actions = {step.action for step in steps}
    return {
        "click": "click" in actions,
        "ipv": any(
            (step.screen_type == "ipv" or step.action == "back_home" for step in steps)
        ),
        "cart": "ipv_cart" in actions,
        "buy": "ipv_buy" in actions,
    }


def _clicks_without_ipv(raw_trajectory: Sequence) -> tuple:
    total = 0
    via_back_home = 0
    for index, step in enumerate(raw_trajectory):
        if step.screen_type != "feed" or step.action != "click":
            continue
        if feed_click_reaches_ipv(raw_trajectory, index):
            continue
        total += 1
        via_back_home += int(
            any(
                (
                    follower.action == "back_home"
                    for follower in _excursion_after(raw_trajectory, index)
                )
            )
        )
    return (total, via_back_home)


def _excursion_after(raw_trajectory: Sequence, index: int) -> list:
    excursion = []
    for follower in raw_trajectory[index + 1 :]:
        if follower.screen_type == "feed":
            if follower.action == "click":
                break
            continue
        excursion.append(follower)
    return excursion
