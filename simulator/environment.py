"""Shopping environment with separate logged replay and free rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import geometry
from . import ipv_snapshot as snapshot
from .actions import ActionPrediction, FEED_SCROLL_ACTIONS, IPV_SCROLL_ACTIONS, IPV_ACTIONS, match_click_target
from .data import FeedItem, IpvState, Session


REWARDS = {
    "scroll_down": 0.0, "scroll_up": 0.0,
    "click": 0.0, "back_home": 0.0, "end": 0.0,
    "ipv_scroll_down": 0.0, "ipv_scroll_up": 0.0, "ipv_swipe_pic": 0.2,
    "ipv_click_param": 0.5,
    "ipv_enter_comment": 0.5, "ipv_click_comment": 0.5,
    "ipv_cart": 3.0, "ipv_buy": 5.0,
}

@dataclass
class Observation:
    screen_type: str
    step_number: int
    visible_items: List[FeedItem] = field(default_factory=list)
    current_item: Optional[FeedItem] = None
    available_actions: Tuple[str, ...] = ()
    scroll_position: int = 0
    max_scroll_y: int = 0
    scroll_viewport_height: int = 0

    snapshot: Optional[snapshot.PageSnapshot] = None
    previous_actions: Tuple[str, ...] = ()
    execution_feedback: str = ""

    def trace_state(self):
        return {
            "screen_type": self.screen_type,
            "scroll_y": int(self.scroll_position),
            "viewport_number": round(self.scroll_position / self.scroll_viewport_height + 1, 4)
            if self.scroll_viewport_height else 1.0,
        }

@dataclass
class EnvStep:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict

@dataclass(frozen=True)
class ActionValidation:

    valid: bool
    reason: str = ""
    resolved_item_id: Optional[str] = None
    resolved_title: Optional[str] = None
    resolved_alias: Optional[str] = None
    resolved_review_index: Optional[int] = None
    resolved_target: Optional[object] = field(default=None, repr=False, compare=False)

def _resolve_layouts(layout_manifest) -> Dict[str, geometry.PageLayout]:
    """Normalize the ``layout_manifest`` argument into a item_id -> layout map."""
    if layout_manifest is None:
        return {}
    if not isinstance(layout_manifest, dict):
        return geometry.load_layout_manifest(layout_manifest)
    resolved: Dict[str, geometry.PageLayout] = {}
    for item_id, entry in layout_manifest.items():
        if isinstance(entry, geometry.PageLayout):
            resolved[str(item_id)] = entry
        elif isinstance(entry, dict) and entry.get("total_height"):
            resolved[str(item_id)] = geometry.PageLayout.from_measurement(entry)
    return resolved

def _seat_captured_heights(
    session: Session, measured: Dict[str, geometry.PageLayout]
) -> tuple:
    """``(layouts, blockless)``: measured blocks re-seated on captured heights."""
    layouts = dict(measured)
    blockless = []
    for item_id, capture in session.ipv_captures.items():
        seated, adopted = geometry.with_measured_blocks(
            capture.page_layout(), measured.get(item_id)
        )
        layouts[item_id] = seated
        if not adopted:
            blockless.append(item_id)
    return layouts, blockless

class RolloutEnvironment:
    """Rule-driven rollout aligned with the catalog geometry."""

    def __init__(
        self,
        session: Session,
        max_steps: int = 20,
        logged_replay: bool = False,
        layout_manifest=None,
        strict_click: bool = True,
        count_invalid_actions: bool = True,
        ipv_layout_provider=None,
        evaluation_mode: bool = False,
    ) -> None:
        self.evaluation_mode = evaluation_mode
        self.count_invalid_actions = count_invalid_actions
        self._ipv_layout_provider = ipv_layout_provider
        self._resolved_pixel_layouts = set()
        self.logged_replay = logged_replay
        self._layouts, self._blockless_layouts = _seat_captured_heights(
            session, _resolve_layouts(layout_manifest)
        )
        self.session = session
        self.viewport_height = session.feed_geometry["viewport_height"]
        self.max_steps = max_steps
        self.strict_click = strict_click
        self._landing_states = {} if logged_replay else session.landing_states
        self._feed_items = session.feed_items
        self._max_scroll_y = session.feed_geometry["max_scroll_y"]
        self.reset()

    def _initial_scroll_position(self) -> int:
        """Feed offset the rollout opens at."""
        offset = self.session.initial_feed_scroll_y
        if offset is None:
            return 0
        return max(0, min(int(round(offset)), self._max_scroll_y))

    def reset(self) -> Tuple[Observation, dict]:
        self._screen_type = "feed"
        self._scroll_position = self._initial_scroll_position()
        self._current_item: Optional[FeedItem] = None
        self._ipv_state: Optional[IpvState] = None
        self._step_count = 0
        self._reset_ipv_state()
        self._carted_items: set = set()
        self._bought_items: set = set()
        self._history: List[str] = []
        self._logged_feed_visible_items: Optional[List[FeedItem]] = None
        return self.observe(), {
            "session_id": self.session.session_id,
            "total_feed_items": len(self._feed_items),
            "bbox_coverage": 1.0,
            "synthetic_rollout": True,
            "feed_layout": "bbox",
            "feed_viewport_height": self.viewport_height,
            "initial_scroll_y": self._scroll_position,
            "bbox_fallback": False,
            "ipv_logged_state_available": self.session.has_logged_ipv_states,
            "ipv_viewport_height": geometry.IPV_VIEWPORT_HEIGHT,

            "ipv_products_without_blocks": len(self._blockless_layouts),
        }

    @property
    def visible_items(self) -> List[FeedItem]:
        """Cards exposed enough for the observation to describe them."""
        return self._feed_page(geometry.is_visible)

    @property
    def clickable_items(self) -> List[FeedItem]:
        """Cards a tap could land on: anything with a pixel inside the viewport."""
        return self._feed_page(geometry.is_clickable)

    def _feed_page(self, predicate) -> List[FeedItem]:
        """Feed items passing ``predicate(bbox, scroll_y)`` at the current offset."""
        if self._screen_type != "feed":
            return []
        if self._logged_feed_visible_items is not None:

            return self._logged_feed_visible_items
        return [
            item for item in self._feed_items
            if predicate(item.bbox, self._scroll_position, self.viewport_height)
        ]

    def seat_logged_feed_visible_items(self, item_ids: list[str]) -> None:
        """Seat the current logged Feed frame for replay."""
        if not self.logged_replay:
            raise ValueError("logged Feed cards are available only during replay")
        self._logged_feed_visible_items = [
            self.session.get_item(item_id) for item_id in item_ids
        ]

    def available_actions(self) -> Tuple[str, ...]:
        if self._screen_type == "feed":

            actions = []
            if (self.logged_replay and not self.evaluation_mode) or self._scroll_position < self._max_scroll_y:
                actions.append("scroll_down")
            if self._scroll_position > 0:
                actions.append("scroll_up")

            if self.visible_items:
                actions.append("click")
            actions.append("end")
            return tuple(actions)
        return self._ipv_available_actions()

    def _ipv_available_actions(self) -> Tuple[str, ...]:
        if self._ipv_state is None:
            return tuple(action for action in IPV_ACTIONS
                         if action != "ipv_click_comment"
                         and (action != "ipv_scroll_up" or self._ipv_scroll_y > 0))

        layout = self.current_layout()
        has_review_target = bool(self.current_reviews) and (
            self._subpage == snapshot.SUBPAGE_COMMENT_LIST
            or (
                self._subpage == snapshot.SUBPAGE_MAIN
                and layout is not None
                and any(
                    block.kind == geometry.BLOCK_REVIEW
                    for block in layout.visible_blocks(self._ipv_scroll_y)
                )
            )
        )
        can_scroll_down = True

        can_scroll_up = self._ipv_scroll_y > 0
        if layout is not None and self._subpage == snapshot.SUBPAGE_MAIN:
            can_scroll_down = self._ipv_scroll_y < layout.max_scroll_y

        actions = []
        for action in IPV_ACTIONS:
            if action == "ipv_click_comment" and not has_review_target:
                continue
            if action == "ipv_scroll_down" and not can_scroll_down:
                continue
            if action == "ipv_scroll_up" and not can_scroll_up:
                continue
            actions.append(action)
        return tuple(actions)

    def apply_ipv_state(self, state: IpvState) -> Observation:
        if self._screen_type != "ipv" or self._current_item is None:
            raise ValueError("an IPV state can only be applied on the IPV screen")
        if state.item_id and state.item_id != self._current_item.item_id:
            raise ValueError(
                "IPV state product {} does not match current product {}".format(
                    state.item_id, self._current_item.item_id
                )
            )
        self._ipv_state = state
        self._subpage = state.subpage
        self._params_open = state.params_open
        if state.hero_index is not None:
            self._hero_index = state.hero_index
        return self.observe()

    @property
    def ipv_view_state(self) -> dict:
        return {
            "subpage": self._subpage,
            "params_open": bool(self._params_open),
            "hero_index": int(self._hero_index),
        }

    @property
    def current_reviews(self) -> Sequence[snapshot.ReviewEntry]:
        state = self._ipv_state
        if state is None:
            return ()
        return snapshot.build_reviews(state.reviews)

    def current_layout(self) -> Optional[geometry.PageLayout]:
        """Current product page layout."""

        item = self._current_item
        if item is None or not item.item_id:
            return None
        if self._ipv_layout_provider is not None and item.item_id not in self._resolved_pixel_layouts:
            pixel_layout = self._ipv_layout_provider(item.item_id)
            if pixel_layout is None:
                raise ValueError("ipv_scroll_geometry_missing: no readable long image")
            layout, adopted = geometry.with_measured_blocks(pixel_layout, self._layouts.get(item.item_id))
            self._layouts[item.item_id] = layout
            if not adopted and item.item_id not in self._blockless_layouts:
                self._blockless_layouts.append(item.item_id)
            self._resolved_pixel_layouts.add(item.item_id)
        return self._layouts.get(item.item_id)

    def observe(self) -> Observation:
        layout = self.current_layout() if self._screen_type == "ipv" else None
        return Observation(
            screen_type=self._screen_type,
            step_number=self._step_count,
            visible_items=self.visible_items,
            current_item=self._current_item,
            available_actions=self.available_actions(),
            scroll_position=(
                self._ipv_scroll_y if self._screen_type == "ipv"
                else self._scroll_position
            ),
            max_scroll_y=(
                (layout.max_scroll_y if layout else 0)
                if self._screen_type == "ipv" else self._max_scroll_y
            ),
            scroll_viewport_height=(
                (layout.viewport_height if layout
                 else geometry.IPV_VIEWPORT_HEIGHT)
                if self._screen_type == "ipv" else int(round(self.viewport_height))
            ),
            snapshot=self._build_snapshot() if self._screen_type == "ipv" else None,
            previous_actions=tuple(self._history[-10:]),
        )

    def _build_snapshot(self) -> Optional[snapshot.PageSnapshot]:
        state = self._ipv_state
        if state is None:
            return None
        layout = self.current_layout()
        viewport_height = layout.viewport_height if layout else geometry.IPV_VIEWPORT_HEIGHT
        total_height = layout.total_height if layout else 0
        reviews = self.current_reviews
        parts = {
            "price": state.price,
            "hero_index": self._hero_index,
            "parameters": state.parameters,
            "sku": state.sku_options,
            "reviews": reviews,
        }
        visible = []
        if layout is not None and self._subpage == snapshot.SUBPAGE_MAIN:
            for block in layout.visible_blocks(self._ipv_scroll_y):
                rendered = snapshot.render_visible_block(block, parts)
                if rendered is not None:
                    visible.append(rendered)
        current_item_id = self._current_item.item_id if self._current_item else ""
        selected_cta = state.selected_cta
        if current_item_id in self._carted_items:
            selected_cta = "cart"
        if current_item_id in self._bought_items:
            selected_cta = "buy"
        return snapshot.PageSnapshot(
            item_id=state.item_id,
            title=state.title,
            source_step_index=state.source_step_index,
            gui_image=state.gui_image,
            price=state.price,
            category=state.category,
            subpage=self._subpage,
            scroll_y=self._ipv_scroll_y,
            total_height=total_height,
            viewport_height=viewport_height,
            visible=tuple(visible),
            hero_index=self._hero_index,
            params_open=self._params_open,
            parameters=state.parameters,
            sku_open=state.sku_open,
            sku_options=state.sku_options,
            selected_cta=selected_cta,
            reviews=reviews,
            action_history=self._action_history(),
        )

    def _action_history(self) -> Tuple[Tuple[str, int], ...]:
        counts: Dict[str, int] = {}
        for action in self._ipv_actions_taken:
            if action in ("back_home", "end"):
                continue
            counts[action] = counts.get(action, 0) + 1
        return tuple(counts.items())

    def validate_action(self, action: ActionPrediction) -> ActionValidation:
        """Validate ``action`` in the current state without advancing or mutating it."""
        current_item_id = (
            str(self._current_item.item_id)
            if self._screen_type == "ipv" and self._current_item is not None
            else None
        )
        if action.action not in self.available_actions():
            layer = "Feed" if self._screen_type == "feed" else "IPV"
            return ActionValidation(
                False,
                "action is not available on {}".format(layer),
                resolved_item_id=current_item_id,
            )
        if not self.logged_replay or self.evaluation_mode:
            if action.action in FEED_SCROLL_ACTIONS + IPV_SCROLL_ACTIONS:
                layout = self.current_layout() if self._screen_type == "ipv" else None
                maximum = (geometry.MAX_SCROLL_DISTANCE if self._screen_type == "feed"
                           else layout.viewport_height if layout else geometry.IPV_VIEWPORT_HEIGHT)
                if type(action.distance) is not int or not geometry.MIN_SCROLL_DISTANCE <= action.distance <= maximum:
                    return ActionValidation(False, "cannot resolve scroll distance")
        if ((not self.logged_replay or self.evaluation_mode) and self._screen_type == "feed"
                and action.action in FEED_SCROLL_ACTIONS):
            distance = action.distance
            position = self._scroll_position
            target = (min(position + distance, self._max_scroll_y)
                      if action.action == "scroll_down" else max(0, position - distance))
            if target == position:
                return ActionValidation(False, "action is not available on Feed: scroll has no displacement")
        if self._screen_type == "feed" and action.action == "click":
            item = self._resolve_click(action)
            if item is None:
                return ActionValidation(False, "cannot resolve click target")
            return ActionValidation(
                True,
                resolved_item_id=str(item.item_id) if item.item_id else None,
                resolved_title=item.item_title or None,
                resolved_target=item,
            )
        if self._screen_type == "ipv" and action.action == "ipv_click_comment":
            review = self._resolve_review(action)
            if review is None:
                return ActionValidation(
                    False,
                    "cannot resolve review target",
                    resolved_item_id=current_item_id,
                )
            return ActionValidation(
                True,
                resolved_item_id=current_item_id,
                resolved_alias=review.alias,
                resolved_review_index=review.review_index,
                resolved_target=review,
            )
        return ActionValidation(True, resolved_item_id=current_item_id)

    def step(self, action: ActionPrediction) -> EnvStep:
        validation = self.validate_action(action)
        if self._screen_type == "feed":
            return self._step_feed(action, validation)
        return self._step_ipv(action, validation)

    def seat_feed_scroll(self, scroll_y: float) -> int:
        """Place the Feed viewport at a known offset; returns the offset actually set."""
        if self._screen_type != "feed":
            raise ValueError("seat_feed_scroll requires Feed")
        self._scroll_position = max(0, int(round(float(scroll_y))))
        self._max_scroll_y = max(self._max_scroll_y, self._scroll_position)
        return self._scroll_position

    def _step_feed(
        self, action: ActionPrediction, validation: ActionValidation
    ) -> EnvStep:
        info = {"screen_type": "feed", "action": action.action, "valid": True}
        terminated = False
        if not validation.valid:
            return self._invalid(action, validation.reason)
        if action.action == "scroll_down":
            distance = action.distance
            self._scroll_position = min(self._scroll_position + distance, self._max_scroll_y)
        elif action.action == "scroll_up":
            distance = action.distance
            self._scroll_position = max(0, self._scroll_position - distance)
        elif action.action == "click":
            item = validation.resolved_target

            on_screen_ids = {shown.item_id for shown in self.clickable_items}
            if item.item_id not in on_screen_ids and item.bbox:
                center = int(
                    item.bbox["y"]
                    + item.bbox.get("height", geometry.DEFAULT_CARD_HEIGHT) / 2
                )
                self._scroll_position = max(
                    0, min(int(center - self.viewport_height // 2), self._max_scroll_y)
                )
            self._screen_type = "ipv"
            self._current_item = item
            self._ipv_state = None
            self._reset_ipv_state()
            if not self.logged_replay:
                self.apply_ipv_state(self._landing_states[item.item_id])
            info["clicked_item_id"] = item.item_id
        elif action.action == "end":
            terminated = True
            info["termination_reason"] = "end"
        return self._valid_result(action, info, terminated)


    def _step_ipv(
        self, action: ActionPrediction, validation: ActionValidation
    ) -> EnvStep:
        info = {
            "screen_type": "ipv", "action": action.action, "valid": True,
            "current_item_id": self._current_item.item_id if self._current_item else None,
        }
        terminated = False
        if not validation.valid:
            return self._invalid(action, validation.reason)

        if action.action in ("ipv_enter_comment", "ipv_click_comment"):

            self._params_open = False
        elif action.action != "ipv_click_param":
            if self._subpage != snapshot.SUBPAGE_MAIN:
                self._leave_comment_subpage()
                info["implicit_return"] = True
            if self._params_open:
                self._params_open = False
                info["params_closed"] = True

        layout = self.current_layout()
        max_scroll_y = layout.max_scroll_y if layout else 0
        new_content = False

        if action.action == "ipv_scroll_down":
            delta = action.distance
            self._ipv_scroll_y = min(self._ipv_scroll_y + delta, max_scroll_y)
            info["distance"] = delta
        elif action.action == "ipv_scroll_up":
            delta = action.distance
            self._ipv_scroll_y = max(0, self._ipv_scroll_y - delta)
            info["distance"] = delta
        elif action.action == "ipv_swipe_pic":
            self._hero_index += 1
            new_content = self._hero_index not in self._seen_hero_indices
            self._seen_hero_indices.add(self._hero_index)
            info["hero_index"] = self._hero_index
        elif action.action == "ipv_click_param":
            if self._subpage != snapshot.SUBPAGE_MAIN:
                self._leave_comment_subpage()
                info["implicit_return"] = True
            new_content = not self._params_open
            self._params_open = True
            info["parameter_count"] = len(
                self._ipv_state.parameters if self._ipv_state else ()
            )
        elif action.action == "ipv_enter_comment":
            self._enter_comment_subpage()
            new_content = "ipv_enter_comment" not in self._ipv_actions_taken
            info["review_count"] = len(self.current_reviews)
        elif action.action == "ipv_click_comment":
            review = validation.resolved_target
            self._enter_comment_subpage()
            new_content = review.review_index not in self._seen_review_indices
            self._seen_review_indices.add(review.review_index)
            info["review_alias"] = review.alias
            info["review_index"] = review.review_index
        elif action.action == "ipv_cart":
            new_content = info["current_item_id"] not in self._carted_items
            self._carted_items.add(info["current_item_id"])
            info["carted_item_id"] = info["current_item_id"]
        elif action.action == "ipv_buy":
            new_content = info["current_item_id"] not in self._bought_items
            self._bought_items.add(info["current_item_id"])
            info["bought_item_id"] = info["current_item_id"]
        elif action.action == "back_home":
            self._screen_type = "feed"
            self._current_item = None
            self._ipv_state = None
        elif action.action == "end":
            terminated = True
            info["termination_reason"] = "end"

        self._ipv_actions_taken.append(action.action)
        info["new_content"] = new_content
        return self._valid_result(action, info, terminated, new_content=new_content)

    def _enter_comment_subpage(self) -> None:
        """Open the review list, remembering the main page's offset."""
        if self._subpage == snapshot.SUBPAGE_MAIN:
            self._main_scroll_y = self._ipv_scroll_y
        self._subpage = snapshot.SUBPAGE_COMMENT_LIST

    def _leave_comment_subpage(self) -> None:
        self._subpage = snapshot.SUBPAGE_MAIN
        self._ipv_scroll_y = self._main_scroll_y

    def _resolve_review(
        self, action: ActionPrediction
    ) -> Optional[snapshot.ReviewEntry]:
        """Resolve an explicitly requested visible review."""
        page_snapshot = self._build_snapshot()
        visible_aliases = set(snapshot.visible_review_aliases(page_snapshot))
        return next((review for review in self.current_reviews
                     if review.alias == action.target_alias and review.alias in visible_aliases), None)

    def _resolve_click(self, action: ActionPrediction) -> Optional[FeedItem]:

        candidates = self.clickable_items

        matched_item_id, _ = match_click_target(
            action.target_title, action.target_item_id, candidates
        )
        if matched_item_id:
            for item in candidates:
                if str(item.item_id) == str(matched_item_id):
                    return item
        if self.strict_click:

            return None

        if action.target_item_id:
            return self.session.get_item(action.target_item_id)
        if action.target_title:
            for item in self._feed_items:
                if item.item_title == action.target_title:
                    return item
        return None

    def _valid_result(
        self,
        action: ActionPrediction,
        info: dict,
        terminated: bool,
        new_content: bool = True,
    ) -> EnvStep:
        reward = REWARDS.get(action.action, 0.0)
        if info.get("screen_type") == "ipv" and not new_content:
            reward = 0.0
        self._step_count += 1
        self._history.append(action.action)
        truncated = self._step_count >= self.max_steps and not terminated
        info["step_count"] = self._step_count
        info["reward"] = reward
        return EnvStep(self.observe(), reward, terminated, truncated, info)

    def _invalid(self, action: ActionPrediction, reason: str) -> EnvStep:
        if self.count_invalid_actions:
            self._step_count += 1
        truncated = self._step_count >= self.max_steps
        info = {
            "screen_type": self._screen_type, "action": action.action,
            "valid": False, "reason": reason, "step_count": self._step_count,
        }
        return EnvStep(self.observe(), 0.0, False, truncated, info)

    def _reset_ipv_state(self) -> None:
        self._ipv_actions_taken = []
        self._ipv_scroll_y = 0
        self._main_scroll_y = 0
        self._hero_index = 0
        self._params_open = False
        self._subpage = snapshot.SUBPAGE_MAIN
        self._seen_review_indices: set = set()
        self._seen_hero_indices: set = {0}
