"""Reconstruct logged pre-action states without exposing the next action to a policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from . import geometry
from .actions import ActionPrediction, IPV_SCROLL_ACTIONS
from .data import IpvState, Session, TrajectoryStep
from .environment import EnvStep, Observation, RolloutEnvironment
from .ipv_snapshot import visible_review_aliases

REPLAY_DISCARD_REASONS = (
    "action_unavailable",
    "target_unresolved",
    "missing_pre_action_ipv_state",
    "comment_target_not_previously_visible",
)

@dataclass(frozen=True)
class ReplayPreAction:
    """Pre-action observation for replay."""

    session_id: str
    step_index: int
    observation: Observation
    replay_only: bool
    scroll_source: Optional[str]
    observation_source_step_index: Optional[int]
    observation_provenance: str
    available_actions: Tuple[str, ...]
    visible_item_ids: Tuple[str, ...]
    review_aliases: Tuple[str, ...]
    opening_click_step_index: Optional[int]
    desynced: bool

@dataclass(frozen=True)
class ReplayCommit:
    """Ground-truth transition committed after a preceding policy decision."""

    pre_action: ReplayPreAction
    ground_truth: Optional[ActionPrediction]
    ground_truth_title: str
    result: Optional[EnvStep]
    post_observation: Observation
    accepted: bool
    replay_only: bool
    discard_reason: str
    scroll_distance: Optional[int]
    scroll_distance_source: Optional[str]
    review_match_method: Optional[str]
    gui_image: str
    frame_ordinal: Optional[int]
    logged_scroll_y: Optional[float]
    opening_click_step_index: Optional[int]
    retain_record_on_reject: bool

@dataclass(frozen=True)
class _PendingStep:
    pre_action: ReplayPreAction
    step: TrajectoryStep
    logged_feed_state: bool


def seat_feed_offset(env: RolloutEnvironment, step: TrajectoryStep) -> str:
    """Place a Feed replay observation at its logged offset when available."""
    if step.scroll_y is None:
        return "assumed"
    env.seat_feed_scroll(step.scroll_y)
    return "log"

def review_match_for(step: TrajectoryStep, page_snapshot) -> tuple:
    """Map the prepared zero-based target to a currently visible review alias."""
    if step.action != "ipv_click_comment" or page_snapshot is None:
        return None, None
    aliases = set(visible_review_aliases(page_snapshot))
    for review in page_snapshot.reviews:
        if review.alias in aliases and review.review_index == step.review_index:
            return review.alias, "review_index"
    return None, None

class LoggedReplayCursor:
    """Reconstruct logged pre-action states without exposing labels to a policy."""

    def __init__(
        self,
        session: Session,
        max_steps: int = 200,
        layout_manifest=None,
        ipv_layout_provider=None,
    ) -> None:
        self.session = session
        self.env = RolloutEnvironment(
            session,
            max_steps=max_steps,
            logged_replay=True,
            layout_manifest=layout_manifest,
            ipv_layout_provider=ipv_layout_provider,
            strict_click=False,
        )
        self._observation, _ = self.env.reset()
        self._trajectory = session.gt_trajectory
        self._index = 0
        self._pending: Optional[_PendingStep] = None
        self._finished = False
        self._desynced = False
        self._opening_click_step_index: Optional[int] = None
        self._discards = {reason: 0 for reason in REPLAY_DISCARD_REASONS}

    @property
    def discards(self) -> dict:
        return dict(self._discards)

    @property
    def finished(self) -> bool:
        return self._finished

    def next_pre_action(self) -> Optional[ReplayPreAction]:
        """Return the next safely reconstructed state, skipping unreplayable rows."""
        if self._pending is not None:
            raise RuntimeError("commit_ground_truth() is required before the next state")
        while not self._finished and self._index < len(self._trajectory):
            step = self._trajectory[self._index]
            scroll_source = self._seat_feed_state(step)
            if step.screen_type == "ipv" and step.action not in ("back_home", "end"):
                if not self._apply_pre_state(step):
                    self._discard("missing_pre_action_ipv_state")
                    self._index += 1
                    continue
            if not step.action:
                pre = self._pre_action(scroll_source, replay_only=True)
                self._pending = _PendingStep(pre, step, False)
                return pre
            logged_feed_state = (
                self._observation.screen_type == "feed"
                and step.screen_type == "feed"
            )
            review_alias, _ = review_match_for(step, self._observation.snapshot)
            if step.action == "ipv_click_comment" and review_alias is None:
                self._discard("comment_target_not_previously_visible")
                self._index += 1
                continue
            if step.action not in self._observation.available_actions and not logged_feed_state:
                self._discard("action_unavailable")
                self._desynced = True
                self._index += 1
                continue
            pre = self._pre_action(scroll_source, replay_only=False)
            self._pending = _PendingStep(pre, step, logged_feed_state)
            return pre
        self._finished = True
        return None

    def commit_ground_truth(self) -> ReplayCommit:
        """Advance through the pending logged row after its model output was saved."""
        if self._pending is None:
            raise RuntimeError("next_pre_action() must return a state before committing")
        pending = self._pending
        self._pending = None
        step = pending.step
        if pending.pre_action.replay_only:
            self._index += 1
            return ReplayCommit(
                pre_action=pending.pre_action,
                ground_truth=None,
                ground_truth_title=self.step_title(step),
                result=None,
                post_observation=self._observation,
                accepted=True,
                replay_only=True,
                discard_reason="",
                scroll_distance=None,
                scroll_distance_source=None,
                review_match_method=None,
                gui_image=step.gui_image,
                frame_ordinal=step.frame_ordinal,
                logged_scroll_y=step.scroll_y,
                opening_click_step_index=self._opening_click_step_index,
                retain_record_on_reject=False,
            )

        distance, distance_source = self._scroll_label(step)
        review_alias, review_match_method = review_match_for(
            step, pending.pre_action.observation.snapshot
        )
        ground_truth = ActionPrediction(
            step.action,
            target_item_id=step.clicked_item_id,
            target_title=self.step_title(step) or None,
            target_alias=review_alias,
            distance=self._ground_truth_distance(step, distance),
        )
        result = self.env.step(ground_truth)
        accepted = bool(result.info.get("valid", True))
        discard_reason = ""
        if not accepted:
            discard_reason = "target_unresolved"
            if not pending.logged_feed_state:
                self._discards[discard_reason] += 1
        self._observation = result.observation
        if accepted:
            if self._observation.screen_type == "feed":
                self._opening_click_step_index = None
            if result.terminated or result.truncated:
                self._finished = True
        self._index += 1
        return ReplayCommit(
            pre_action=pending.pre_action,
            ground_truth=ground_truth,
            ground_truth_title=self.step_title(step),
            result=result,
            post_observation=self._observation,
            accepted=accepted,
            replay_only=False,
            discard_reason=discard_reason,
            scroll_distance=distance,
            scroll_distance_source=distance_source,
            review_match_method=review_match_method,
            gui_image=step.gui_image,
            frame_ordinal=step.frame_ordinal,
            logged_scroll_y=step.scroll_y,
            opening_click_step_index=pending.pre_action.opening_click_step_index,
            retain_record_on_reject=pending.logged_feed_state,
        )

    def step_title(self, step: TrajectoryStep) -> str:
        if step.item_title:
            return step.item_title
        item = self.session.get_item(step.clicked_item_id or step.item_id)
        return item.item_title if item else ""

    def _seat_feed_state(self, step: TrajectoryStep) -> Optional[str]:
        if self._observation.screen_type != "feed":
            return None
        scroll_source = seat_feed_offset(self.env, step)
        if step.screen_type == "feed":
            self.env.seat_logged_feed_visible_items(step.visible_items)
        self._observation = self.env.observe()
        if step.screen_type == "feed" and step.action == "click":
            self._opening_click_step_index = self._index
        return scroll_source


    def _pre_action(self, scroll_source: Optional[str], replay_only: bool) -> ReplayPreAction:
        snapshot = self._observation.snapshot
        source_index = snapshot.source_step_index if snapshot is not None else None
        if snapshot is not None:
            provenance = "logged_current_pre_state"
        elif scroll_source == "log":
            provenance = "logged_feed_offset"
        elif scroll_source == "assumed":
            provenance = "modelled_feed_offset"
        else:
            provenance = "environment_state"
        return ReplayPreAction(
            session_id=self.session.session_id,
            step_index=self._index,
            observation=self._observation,
            replay_only=replay_only,
            scroll_source=scroll_source,
            observation_source_step_index=source_index,
            observation_provenance=provenance,
            available_actions=tuple(self._observation.available_actions),
            visible_item_ids=tuple(
                str(item.item_id) for item in self._observation.visible_items if item.item_id
            ),
            review_aliases=visible_review_aliases(snapshot),
            opening_click_step_index=(
                self._opening_click_step_index if snapshot is not None else None
            ),
            desynced=self._desynced,
        )

    def _apply_pre_state(self, step: TrajectoryStep) -> bool:
        if self._observation.screen_type != "ipv":
            return False

        state = IpvState.from_step(step, self._index,
                                  self.session.get_item(step.item_id or step.clicked_item_id))
        self._observation = self.env.apply_ipv_state(state)
        return True

    def _scroll_label(self, step: TrajectoryStep) -> tuple:
        if step.action in ("scroll_down", "scroll_up", "ipv_scroll_down", "ipv_scroll_up"):
            value = step.feed_scroll_distance if step.screen_type == "feed" else step.ipv_scroll_distance
            if isinstance(value, (int, float)) and math.isfinite(value) and abs(value) > 0:
                return int(round(abs(value))), "logged_current_action_pixels"
            return None, "distance_unknown"
        return None, None

    def _ground_truth_distance(self, step: TrajectoryStep, distance: Optional[int]) -> Optional[int]:
        if step.action in ("scroll_down", "scroll_up"):
            return distance if distance is not None else geometry.SCROLL_STEP
        if step.action in IPV_SCROLL_ACTIONS:
            viewport = self._observation.scroll_viewport_height
            return min(viewport, distance if distance is not None else viewport)
        return None

    def _discard(self, reason: str) -> None:
        self._discards[reason] += 1


def _feed_click_unpresentable(
    step: TrajectoryStep, session: "Session"
) -> bool:
    """Whether this logged Feed click had no on-screen card to click."""
    if step.screen_type != "feed" or step.action != "click":
        return False
    scroll_y = step.scroll_y
    if scroll_y is None:
        return False
    item = session.get_item(step.clicked_item_id)
    if item is None or not item.bbox:
        return True
    return not geometry.is_clickable(item.bbox, scroll_y)

def feed_click_reaches_ipv(
    trajectory: Sequence[TrajectoryStep], index: int
) -> bool:
    """Whether an IPV step follows ``index`` before the next Feed click."""
    for follower in trajectory[index + 1:]:
        if follower.screen_type == "ipv":
            return True
        if follower.screen_type == "feed" and follower.action == "click":
            return False
    return False

def structural_defects(trajectory: Sequence[TrajectoryStep]) -> Dict[str, int]:
    """Steps in a prepared trajectory that no state change can explain."""
    counts = {
        "orphan_ipv_step": 0,
        "ipv_product_mismatch": 0,
        "back_home_without_page": 0,
        "feed_without_return": 0,
    }
    on_ipv = False
    active_product = ""
    for step in trajectory:
        if step.action == "back_home":
            if not on_ipv:
                counts["back_home_without_page"] += 1
            on_ipv = False
            active_product = ""
            continue
        if step.screen_type == "ipv":
            if not on_ipv:
                counts["orphan_ipv_step"] += 1
            elif active_product and step.item_id and step.item_id != active_product:
                counts["ipv_product_mismatch"] += 1
            on_ipv = True
            continue
        if step.screen_type == "feed":
            if on_ipv:
                counts["feed_without_return"] += 1
            on_ipv = step.action == "click"
            active_product = (
                str(step.clicked_item_id or step.item_id or "") if on_ipv else ""
            )
    return counts


def _terminal_on_its_own_screen(
    step: TrajectoryStep, previous: Optional[TrajectoryStep]
) -> TrajectoryStep:
    """The log's terminal ``end`` step, moved onto the screen it happened on."""
    if previous is None:
        return replace(step, screen_type="feed")

    ends_on_ipv = previous.screen_type == "ipv" and previous.action != "back_home"
    return replace(
        step,
        screen_type="ipv" if ends_on_ipv else "feed",
        item_id=(previous.item_id or previous.clicked_item_id) if ends_on_ipv else None,
        visible_items=list(previous.visible_items),
    )

def prepare_trajectory(
    trajectory: Sequence[TrajectoryStep],
    session: "Session",
    discards: Dict[str, int],
) -> List[TrajectoryStep]:
    """Normalize logged scroll events and terminal actions for replay."""
    merged: List[TrajectoryStep] = []

    def discard(reason: str) -> None:
        discards[reason] = discards.get(reason, 0) + 1

    skipping: Optional[str] = None

    on_ipv = False
    for position, step in enumerate(trajectory):
        if skipping is not None:
            if step.action == "end":

                skipping = None
            elif step.screen_type != "feed":
                discard(skipping)
                continue
            elif step.action == "click":

                skipping = None
        if (
            on_ipv
            and step.screen_type == "feed"
            and step.action != "click"
            and feed_click_reaches_ipv(trajectory, position)
        ):

            discard("ipv_excursion_feed_noise")
            continue
        if _feed_click_unpresentable(step, session):
            discard("feed_click_unpresentable")
            skipping = "feed_click_unpresentable_excursion"
            continue
        if (
            step.screen_type == "feed"
            and step.action == "click"
            and not feed_click_reaches_ipv(trajectory, position)
        ):

            discard("feed_click_without_ipv")
            skipping = "feed_click_without_ipv_excursion"
            continue

        if step.action == "back_home" and step.screen_type == "transition":
            if not on_ipv:

                discard("redundant_back_home")
                continue

            step = replace(step, screen_type="ipv")
        if step.action == "end" and step.screen_type == "transition":
            step = _terminal_on_its_own_screen(step, merged[-1] if merged else None)
        merged.append(step)

        if step.action == "back_home":
            on_ipv = False
        elif step.screen_type == "ipv":
            on_ipv = True
        elif step.screen_type == "feed":
            on_ipv = step.action == "click"

    return merged
