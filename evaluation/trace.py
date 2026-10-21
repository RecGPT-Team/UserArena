"""Canonical traces and episode features."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from simulator.actions import (
    ACTIONS, BROWSE_ACTIONS, INFO_ACQUISITION_ACTIONS, INFO_CHANNELS, info_channel, layer_of,
)


def parse_price(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    digits = []
    seen_dot = False
    for char in text:
        if char.isdigit():
            digits.append(char)
        elif char == "." and digits and (not seen_dot):
            seen_dot = True
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    try:
        return float("".join(digits).rstrip("."))
    except ValueError:
        return None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "Usage":
        payload = payload or {}
        return cls(
            input_tokens=int(payload.get("input_tokens") or 0),
            output_tokens=int(payload.get("output_tokens") or 0),
            llm_calls=int(payload.get("llm_calls") or 0),
            latency_ms=payload.get("latency_ms"),
        )


@dataclass
class UserProfile:
    visitor_id: str = ""
    click_history: int = 0
    buy_history: int = 0
    history_price_mean: Optional[float] = None
    history_categories: List[str] = field(default_factory=list)
    attributes: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "UserProfile":
        payload = payload or {}
        return cls(
            visitor_id=str(payload.get("visitor_id") or ""),
            click_history=int(payload.get("click_history") or 0),
            buy_history=int(payload.get("buy_history") or 0),
            history_price_mean=payload.get("history_price_mean"),
            history_categories=list(payload.get("history_categories") or []),
            attributes=dict(payload.get("attributes") or {}),
        )


@dataclass
class RolloutStep:
    index: int
    layer: str
    action: str
    executed: bool = True
    parse_ok: bool = True
    legal: bool = True
    target_resolved: bool = True
    reject_reason: str = ""
    target_item_id: Optional[str] = None
    target_title: Optional[str] = None
    target_alias: Optional[str] = None
    item_price: Optional[float] = None
    item_cate: str = ""
    available_actions: List[str] = field(default_factory=list)
    visible_item_ids: List[str] = field(default_factory=list)
    environment_state: Dict[str, object] = field(default_factory=dict)
    raw_output: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def executable(self) -> bool:
        return bool(self.action in ACTIONS and self.parse_ok and self.legal and self.target_resolved)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usage"] = self.usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "RolloutStep":
        action = str(payload.get("action") or "")
        return cls(
            index=int(payload.get("index") or 0),
            layer=str(payload.get("layer") or layer_of(action)),
            action=action,
            executed=bool(payload.get("executed", True)),
            parse_ok=bool(payload.get("parse_ok", True)),
            legal=bool(payload.get("legal", True)),
            target_resolved=bool(payload.get("target_resolved", True)),
            reject_reason=str(payload.get("reject_reason") or ""),
            target_item_id=payload.get("target_item_id"),
            target_title=payload.get("target_title"),
            target_alias=payload.get("target_alias"),
            item_price=payload.get("item_price"),
            item_cate=str(payload.get("item_cate") or ""),
            available_actions=list(payload.get("available_actions") or []),
            visible_item_ids=[str(v) for v in payload.get("visible_item_ids") or []],
            environment_state=dict(payload.get("environment_state") or {}),
            raw_output=str(payload.get("raw_output") or ""),
            usage=Usage.from_dict(payload.get("usage")),
        )


@dataclass
class RolloutTrace:
    session_id: str
    source: str = "sim"
    rollout_index: int = 0
    visitor_id: str = ""
    profile: UserProfile = field(default_factory=UserProfile)
    steps: List[RolloutStep] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""
    max_steps: Optional[int] = None
    model: str = ""
    error: str = ""

    @property
    def executed_steps(self) -> List[RolloutStep]:
        return [step for step in self.steps if step.executed and step.executable]

    @property
    def policy_steps(self) -> List[RolloutStep]:
        return [step for step in self.executed_steps if step.action in ACTIONS]

    @property
    def length(self) -> int:
        return len(self.executed_steps)

    def actions(self) -> List[str]:
        return [step.action for step in self.executed_steps]

    def to_dict(self) -> dict:
        return {
            "kind": "rollout",
            "session_id": self.session_id,
            "source": self.source,
            "rollout_index": self.rollout_index,
            "visitor_id": self.visitor_id,
            "profile": self.profile.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "max_steps": self.max_steps,
            "model": self.model,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RolloutTrace":
        return cls(
            session_id=str(payload.get("session_id") or ""),
            source=str(payload.get("source") or "sim"),
            rollout_index=int(payload.get("rollout_index") or 0),
            visitor_id=str(payload.get("visitor_id") or ""),
            profile=UserProfile.from_dict(payload.get("profile")),
            steps=[RolloutStep.from_dict(item) for item in payload.get("steps") or []],
            terminated=bool(payload.get("terminated")),
            truncated=bool(payload.get("truncated")),
            termination_reason=str(payload.get("termination_reason") or ""),
            max_steps=payload.get("max_steps"),
            model=str(payload.get("model") or ""),
            error=str(payload.get("error") or ""),
        )


@dataclass
class ReplayStep:
    session_id: str
    step_index: int
    layer: str
    gt_action: str
    pred_action: Optional[str] = None
    gt_target_item_id: Optional[str] = None
    gt_target_title: Optional[str] = None
    pred_target_item_id: Optional[str] = None
    pred_target_title: Optional[str] = None
    gt_target_alias: Optional[str] = None
    pred_target_alias: Optional[str] = None
    parse_ok: bool = True
    legal: bool = True
    target_resolved: bool = True
    reject_reason: str = ""
    available_actions: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    ranked_candidates: List[str] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    scorer: str = ""
    raw_output: str = ""
    observation_source_step_index: Optional[int] = None
    observation_provenance: str = ""
    prediction_present: bool = True
    usage: Usage = field(default_factory=Usage)

    @property
    def executable(self) -> bool:
        return bool(self.pred_action in ACTIONS and self.parse_ok and self.legal and self.target_resolved)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["usage"] = self.usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ReplayStep":
        gt_action = str(payload.get("gt_action") or "")
        if gt_action not in ACTIONS:
            raise ValueError("Unknown replay ground-truth action")
        return cls(
            session_id=str(payload.get("session_id") or ""),
            step_index=int(payload.get("step_index") or 0),
            layer=str(payload.get("layer") or layer_of(gt_action)),
            gt_action=gt_action,
            pred_action=payload.get("pred_action"),
            gt_target_item_id=payload.get("gt_target_item_id"),
            gt_target_title=payload.get("gt_target_title"),
            pred_target_item_id=payload.get("pred_target_item_id"),
            pred_target_title=payload.get("pred_target_title"),
            gt_target_alias=payload.get("gt_target_alias"),
            pred_target_alias=payload.get("pred_target_alias"),
            parse_ok=bool(payload.get("parse_ok", True)),
            legal=bool(payload.get("legal", True)),
            target_resolved=bool(payload.get("target_resolved", True)),
            reject_reason=str(payload.get("reject_reason") or ""),
            available_actions=list(payload.get("available_actions") or []),
            candidates=[str(v) for v in payload.get("candidates") or []],
            ranked_candidates=[str(v) for v in payload.get("ranked_candidates") or []],
            candidate_scores={
                str(k): float(v)
                for k, v in (payload.get("candidate_scores") or {}).items()
            },
            scorer=str(payload.get("scorer") or ""),
            raw_output=str(payload.get("raw_output") or ""),
            observation_source_step_index=payload.get("observation_source_step_index"),
            observation_provenance=str(payload.get("observation_provenance") or ""),
            prediction_present=bool(payload.get("prediction_present", True)),
            usage=Usage.from_dict(payload.get("usage")),
        )


@dataclass
class ReplayTrace:
    steps: List[ReplayStep] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": "replay",
            "model": self.model,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ReplayTrace":
        return cls(
            steps=[ReplayStep.from_dict(item) for item in payload.get("steps") or []],
            model=str(payload.get("model") or ""),
        )


def write_rollout_traces(path: Path, traces: Iterable[RolloutTrace]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_rollout_traces(path: Path) -> Iterator[RolloutTrace]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RolloutTrace.from_dict(json.loads(line))


def write_replay_trace(path: Path, trace: ReplayTrace) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for step in trace.steps:
            payload = step.to_dict()
            payload["model"] = trace.model
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(trace.steps)


def read_replay_trace(path: Path) -> ReplayTrace:
    steps: List[ReplayStep] = []
    model = ""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            model = model or str(payload.get("model") or "")
            steps.append(ReplayStep.from_dict(payload))
    return ReplayTrace(steps=steps, model=model)


def split_by_source(
    traces: Sequence[RolloutTrace],
) -> "tuple[List[RolloutTrace], List[RolloutTrace]]":
    real = [trace for trace in traces if trace.source == "real"]
    sim = [trace for trace in traces if trace.source != "real"]
    return (real, sim)


_DECISION_BY_ACTION = {
    "back_home": "return",
    "ipv_cart": "cart",
    "ipv_buy": "buy",
    "end": "end",
}
TERMINAL_OUTCOMES = ("purchase", "cart_only", "browse_only", "truncated")


@dataclass
class IpvVisit:
    item_id: Optional[str]
    start_index: int
    steps_to_decision: int = 0
    info_to_decision: int = 0
    browse_to_decision: int = 0
    channel_counts: Dict[str, int] = field(default_factory=dict)
    decision: str = "open"
    tail_steps: int = 0

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "start_index": self.start_index,
            "steps_to_decision": self.steps_to_decision,
            "info_to_decision": self.info_to_decision,
            "browse_to_decision": self.browse_to_decision,
            "channel_counts": dict(self.channel_counts),
            "decision": self.decision,
            "tail_steps": self.tail_steps,
        }


def ipv_visits(trace: RolloutTrace) -> List[IpvVisit]:
    visits: List[IpvVisit] = []
    current: Optional[IpvVisit] = None
    for step in trace.executed_steps:
        if step.layer != "ipv":
            current = None
            if step.action == "click":
                current = IpvVisit(item_id=step.target_item_id, start_index=step.index)
                visits.append(current)
            continue
        if current is None:
            current = IpvVisit(item_id=step.target_item_id, start_index=step.index)
            visits.append(current)
        if current.item_id is None and step.target_item_id:
            current.item_id = step.target_item_id
        decided = current.decision != "open"
        if decided:
            current.tail_steps += 1
        else:
            current.steps_to_decision += 1
            if step.action in INFO_ACQUISITION_ACTIONS:
                current.info_to_decision += 1
                channel = info_channel(step.action)
                if channel:
                    current.channel_counts[channel] = (
                        current.channel_counts.get(channel, 0) + 1
                    )
            elif step.action in BROWSE_ACTIONS:
                current.browse_to_decision += 1
            decision = _DECISION_BY_ACTION.get(step.action)
            if decision:
                current.decision = decision
        if step.action == "back_home":
            current = None
    return visits


def terminal_outcome(trace: RolloutTrace) -> str:
    actions = set(trace.actions())
    if "ipv_buy" in actions:
        return "purchase"
    if trace.truncated and (not trace.terminated):
        return "truncated"
    if "ipv_cart" in actions:
        return "cart_only"
    return "browse_only"


def session_features(trace: RolloutTrace) -> dict:
    policy_steps = trace.policy_steps
    length = len(policy_steps)
    actions = [step.action for step in policy_steps]
    visits = ipv_visits(trace)
    counts = Counter(actions)
    prices = [
        step.item_price for step in trace.executed_steps if step.item_price is not None
    ]
    categories = [step.item_cate for step in trace.executed_steps if step.item_cate]
    interacted = [
        step.target_item_id for step in trace.executed_steps if step.target_item_id
    ]
    ipv_steps = sum((1 for step in policy_steps if step.layer == "ipv"))
    decisions = Counter((visit.decision for visit in visits))
    channel_totals = {
        channel: sum((visit.channel_counts.get(channel, 0) for visit in visits))
        for channel in INFO_CHANNELS
    }
    return {
        "session_id": trace.session_id,
        "visitor_id": trace.visitor_id or trace.profile.visitor_id,
        "source": trace.source,
        "rollout_index": trace.rollout_index,
        "length": length,
        "action_counts": dict(counts),
        "ipv_step_share": ipv_steps / length if length else 0.0,
        "clicked": int("click" in counts),
        "entered_ipv": int(bool(visits)),
        "carted": int("ipv_cart" in counts),
        "bought": int("ipv_buy" in counts),
        "click_count": counts.get("click", 0),
        "ipv_visit_count": len(visits),
        "return_proportion": counts.get("back_home", 0) / len(visits)
        if visits
        else 0.0,
        "cart_proportion": decisions.get("cart", 0) / len(visits) if visits else 0.0,
        "buy_proportion": decisions.get("buy", 0) / len(visits) if visits else 0.0,
        "info_depths": [visit.info_to_decision for visit in visits],
        "decision_depths": [visit.steps_to_decision for visit in visits],
        "mean_info_depth": sum((visit.info_to_decision for visit in visits))
        / len(visits)
        if visits
        else 0.0,
        "channel_counts": channel_totals,
        "prices": prices,
        "categories": categories,
        "interacted_item_ids": list(dict.fromkeys(interacted)),
        "terminal_outcome": terminal_outcome(trace),
        "truncated": int(trace.truncated),
        "visits": [visit.to_dict() for visit in visits],
    }
