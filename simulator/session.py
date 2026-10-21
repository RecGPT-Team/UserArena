"""Shopping sessions and model-facing rollout traces."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import json
from pathlib import Path
from typing import List, Optional

from . import actions as reject
from .actions import ACTIONS, ActionPrediction, match_click_target
from .data import Session
from .environment import Observation, RolloutEnvironment
from .ipv_snapshot import visible_review_aliases
from .prompts import build_action_candidates, format_observation
from .images import ImageRenderer


def cross_session_visits(raw):
    """Yield visits in pv_ids order, with each visit's own catalog and geometry."""
    pages = {str(entry["pv_id"]): entry for entry in raw["feed_session_screenshots"]}
    steps = defaultdict(list)
    for step in raw["trajectory"]:
        steps[str(step["pv_id"])].append(step)
    for value in raw["pv_ids"]:
        pv = str(value)
        session = {key: deepcopy(value) for key, value in raw.items()
                   if key not in {"trajectory", "pv_ids", "num_sessions", "feed_session_screenshots"}}
        session.update(session_id=pv, pv_id=pv, trajectory=deepcopy(steps[pv]))
        page = pages[pv]
        session.update({key: deepcopy(page[key])
                        for key in ("feed_long_image", "feed_catalog")})
        yield session


def load_records(path):
    """Read one self-contained JSON object per line, with visit-local catalogs."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class ShoppingSession:
    """One free-rollout session; provide ActionPrediction objects to step()."""

    def __init__(self, raw, *, image_store=None, max_steps=200,
                 render_images=True, evaluation_mode=False):
        self.session = Session.from_raw(raw)
        self.renderer = ImageRenderer(image_store)
        self.renderer.bind_session(raw)
        self._render_images = render_images
        self.env = RolloutEnvironment(
            self.session, max_steps=max_steps,
            ipv_layout_provider=self.renderer.page_layout,
            evaluation_mode=evaluation_mode,
        )
        self.done = False

    def reset(self):
        self.done = False
        observation, _ = self.env.reset()
        return observation

    def step(self, action: ActionPrediction):
        if self.done:
            raise RuntimeError("The session has ended; reset before stepping again")
        result = self.env.step(action)
        self.done = result.terminated or result.truncated
        return result

    def observe(self):
        return self.env.observe()

    def render(self):
        return self.renderer.render(self.observe()) if self._render_images else None


class CrossSessionRollout:
    """Visit-local environments with a bounded history of accepted policy actions.

    Advancing is explicit. Logged future actions are never added to this history.
    """

    def __init__(self, raw, *, history_size=20, **session_options):
        if history_size < 1:
            raise ValueError("History size must be positive")
        self._visits = list(cross_session_visits(raw))
        self._options = session_options
        self.history = deque(maxlen=history_size)
        self.index = 0
        self.current = ShoppingSession(self._visits[0], **self._options)

    def step(self, action):
        result = self.current.step(action)
        if result.info.get("valid"):
            self.history.append({"session_id": self.current.session.session_id,
                                 "action": action.to_dict()})
        return result

    def advance(self):
        if not self.current.done:
            raise RuntimeError("Finish the current visit before advancing")
        if self.index + 1 == len(self._visits):
            return None
        self.index += 1
        self.current = ShoppingSession(self._visits[self.index], **self._options)
        return self.current.observe()


def _strict_action_object(text: str) -> Optional[dict]:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("rationale"), str):
        return None
    action = payload.get("action")
    if not isinstance(action, dict):
        return None
    if any(action.get(key) is not None and not isinstance(action[key], str)
           for key in ("target", "target_item_id")):
        return None
    action_type = action.get("type")
    return action if isinstance(action_type, str) and action_type.strip() else None


class ModelSession:
    def __init__(self, raw, *, image_store=None, max_steps=31):
        self.raw = raw
        self.max_steps = max_steps
        self.renderer = ImageRenderer(image_store)
        self._session = Session.from_raw(raw)
        self._rollout = None
        self._trace_steps = []
        self._trace_profile = {}
        self._trace_terminated = False
        self._trace_truncated = False
        self._trace_termination_reason = ""
        self._pending_step_usage = None

    def reset(self):
        from evaluation.reference import profile_from_session
        from dataclasses import asdict

        self._trace_steps = []
        self._trace_profile = asdict(profile_from_session(self._session))
        self._trace_terminated = self._trace_truncated = False
        self._trace_termination_reason = ""
        self._pending_step_usage = None
        self.renderer.bind_session(self.raw)
        self._rollout = RolloutEnvironment(
            self._session,
            max_steps=self.max_steps,
            ipv_layout_provider=self.renderer.page_layout,
            evaluation_mode=True,
        )
        observation, info = self._rollout.reset()
        return self._observe(observation), info

    def _observe(self, observation):
        image = self.renderer.render(observation)
        return {"image": [image] if image is not None else []}

    def _parse_student_action(self, action, observation):
        if action is None:
            return ActionPrediction("__invalid__")
        kind = action["type"]
        item_id, title = None, None
        if kind == "click":
            item_id, title = match_click_target(
                action.get("target"), action.get("target_item_id"), observation.visible_items
            )
        return ActionPrediction(
            kind, item_id, title, distance=action.get("distance"),
            target_alias=action.get("target") if kind == "ipv_click_comment" else None,
        )

    def trace(self):
        return {
            "kind": "rollout",
            "session_id": self._session.session_id,
            "source": "sim",
            "rollout_index": 0,
            "visitor_id": str(self._session.visitor_id or ""),
            "profile": deepcopy(self._trace_profile),
            "steps": deepcopy(self._trace_steps),
            "terminated": self._trace_terminated,
            "truncated": self._trace_truncated,
            "reason": self._trace_termination_reason,
            "termination_reason": self._trace_termination_reason,
            "max_steps": self.max_steps,
            "model": "",
            "error": "",
        }

    def _observation_text(self, observation: Observation) -> str:
        text = format_observation(observation)
        note = self.renderer.observation_note(observation)
        return text + ("\n" + note if note else "")

    def _action_candidates(self, observation: Observation) -> List[dict]:
        record = {
            "screen_type": observation.screen_type,
            "scroll_viewport_height": observation.scroll_viewport_height,
            "available_actions": list(observation.available_actions),
            "click_targets": list(
                dict.fromkeys(
                    (
                        item.item_title
                        for item in observation.visible_items
                        if item.item_title
                    )
                )
            ),
            "comment_aliases": list(visible_review_aliases(observation.snapshot)),
        }
        return build_action_candidates(record)

    def set_step_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        llm_calls: int = 0,
        latency_ms: Optional[float] = None,
    ) -> None:
        self._pending_step_usage = {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "llm_calls": int(llm_calls or 0),
            "latency_ms": latency_ms,
        }

    def set_episode_status(
        self, terminated: bool, truncated: bool, reason: str = ""
    ) -> None:
        self._trace_terminated = bool(terminated)
        self._trace_truncated = bool(truncated)
        if reason:
            self._trace_termination_reason = str(reason)
        if self._trace_steps:
            self._trace_steps[-1].update(
                {
                    "terminated": self._trace_terminated,
                    "truncated": self._trace_truncated,
                    "termination_reason": self._trace_termination_reason,
                }
            )

    def _record_trace_step(
        self,
        raw_output: str,
        strict_action: Optional[dict],
        prediction,
        observation: Observation,
        info: dict,
        terminated: bool,
        truncated: bool,
    ) -> None:
        requested_action = strict_action["type"] if strict_action is not None else None
        action = requested_action or prediction.action
        if strict_action is None:
            reason = 'parse_failure'
        elif requested_action not in ACTIONS:
            reason = 'unknown_action'
        elif requested_action not in observation.available_actions:
            reason = 'illegal_in_state'
        elif not info.get("valid", True):
            reason = reject.classify(str(info.get("reason") or ""))
        else:
            reason = ""
        parse_ok, legal, target_resolved = reject.gates(reason)
        target_item_id = prediction.target_item_id
        target_title = prediction.target_title
        target_alias = prediction.target_alias
        from evaluation.trace import parse_price

        current_item = observation.current_item
        if requested_action != "click" and current_item is not None:
            target_item_id = current_item.item_id
            target_title = current_item.item_title
        target_item_id = (
            info.get("clicked_item_id")
            or info.get("bought_item_id")
            or info.get("carted_item_id")
            or info.get("current_item_id")
            or target_item_id
        )
        item = self._session.get_item(target_item_id) if target_item_id else None
        step_reason = str(info.get("termination_reason") or "")
        if truncated and (not step_reason):
            step_reason = "max_steps"
        usage = self._pending_step_usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "latency_ms": None,
        }
        self._pending_step_usage = None
        self._trace_steps.append(
            {
                "index": len(self._trace_steps) + 1,
                "layer": observation.screen_type,
                "screen_type": observation.screen_type,
                "action": action,
                "executed": bool(not reason and info.get("valid", True)),
                "parse_ok": parse_ok,
                "legal": legal,
                "target_resolved": target_resolved,
                "reject_reason": reason,
                "target_item_id": str(target_item_id) if target_item_id else None,
                "target_title": target_title,
                "target_alias": target_alias,
                "item_price": parse_price(getattr(item, "item_price", None)),
                "item_cate": str(getattr(item, "item_cate", "") or ""),
                "available_actions": list(observation.available_actions),
                "visible_item_ids": [
                    str(item.item_id)
                    for item in observation.visible_items
                    if item.item_id
                ],
                "environment_state": observation.trace_state(),
                "raw_output": raw_output,
                "usage": usage,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": step_reason,
            }
        )
        self.set_episode_status(terminated, truncated, step_reason)

    def step(self, action: str):
        if self._rollout is None or self._trace_terminated or self._trace_truncated:
            raise RuntimeError("Reset the session before taking another action")
        raw_output = action if isinstance(action, str) else str(action or "")
        pre_observation = self._rollout.observe()
        strict_action = _strict_action_object(raw_output)
        prediction = self._parse_student_action(strict_action, pre_observation)
        result = self._rollout.step(prediction)
        observation = result.observation
        info = dict(result.info)
        terminated = result.terminated
        truncated = result.truncated
        self._record_trace_step(
            raw_output,
            strict_action,
            prediction,
            pre_observation,
            info,
            terminated,
            truncated,
        )
        obs = {"image": []} if terminated or truncated else self._observe(observation)
        return (obs, float(result.reward), terminated, truncated, info)
