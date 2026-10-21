"""Action parsing, vocabulary and rejection categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


FEED_ACTIONS = (
    "scroll_down", "scroll_up", "click", "end",
)

IPV_ACTIONS = (
    "ipv_scroll_down", "ipv_scroll_up", "ipv_swipe_pic",
    "ipv_click_param", "ipv_enter_comment", "ipv_click_comment",
    "ipv_cart", "ipv_buy", "back_home", "end",
)
ACTIONS = tuple(dict.fromkeys(FEED_ACTIONS + IPV_ACTIONS))

FEED_SCROLL_ACTIONS = ("scroll_down", "scroll_up")
IPV_SCROLL_ACTIONS = ("ipv_scroll_down", "ipv_scroll_up")

ACTION_GROUPS: Dict[str, str] = {
    "scroll_down": "feed_navigation", "scroll_up": "feed_navigation",
    "click": "feed_click", "end": "termination", "back_home": "transition",
    "ipv_scroll_down": "ipv_browse", "ipv_scroll_up": "ipv_browse",
    "ipv_swipe_pic": "ipv_media",
    "ipv_click_param": "ipv_information",
    "ipv_enter_comment": "ipv_information", "ipv_click_comment": "ipv_information",
    "ipv_cart": "conversion", "ipv_buy": "conversion",
}

@dataclass(frozen=True)
class ActionPrediction:
    action: str
    target_item_id: Optional[str] = None
    target_title: Optional[str] = None

    distance: Optional[int] = None

    target_alias: Optional[str] = None

    def to_dict(self) -> dict:
        result = {"type": self.action}
        if self.action == "click":
            if self.target_title:
                result["target"] = self.target_title
            if self.target_item_id:
                result["target_item_id"] = self.target_item_id
        if self.target_alias:
            result["target"] = self.target_alias
        if self.distance is not None:
            result["distance"] = self.distance
        return result

def valid_actions(screen_type: str) -> Tuple[str, ...]:
    return FEED_ACTIONS if screen_type == "feed" else IPV_ACTIONS

def match_click_target(
    title: Optional[str], item_id: Optional[str], visible_items: Sequence[object]
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an exact item ID, or an exact full title when no ID is supplied."""
    for item in visible_items:
        matched = (item.item_id == item_id if item_id is not None
                   else bool(title) and item.item_title == title)
        if matched:
            return item.item_id, item.item_title
    return None, None

INFO_ACQUISITION_ACTIONS = frozenset(
    (
        "ipv_click_param",
        "ipv_enter_comment",
        "ipv_click_comment",
        "ipv_swipe_pic",
    )
)
INFO_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "reviews": ("ipv_enter_comment", "ipv_click_comment"),
    "spec": ("ipv_click_param",),
    "media": ("ipv_swipe_pic",),
}
BROWSE_ACTIONS = frozenset(("ipv_scroll_down", "ipv_scroll_up"))


def layer_of(action: str) -> str:
    if action in IPV_ACTIONS and action not in FEED_ACTIONS:
        return "ipv"
    if action in FEED_ACTIONS:
        return "feed"
    return "unknown"


def info_channel(action: str) -> str:
    for channel, members in INFO_CHANNELS.items():
        if action in members:
            return channel
    return ""


CANONICAL = (
    'parse_failure',
    'fallback_substituted',
    'unknown_action',
    'illegal_in_state',
    'target_unresolved',
)



def classify(reason: str) -> str:
    text = (reason or "").strip()
    if not text:
        return ""
    if text in CANONICAL:
        return text
    lowered = text.lower()
    for pattern, canonical in (("not available", "illegal_in_state"),
                               ("resolve", "target_unresolved"),
                               ("unknown action", "unknown_action")):
        if pattern in lowered:
            return canonical
    return text


def gates(reason: str) -> Tuple[bool, bool, bool]:
    canonical = classify(reason)
    if not canonical:
        return (True, True, True)
    if canonical in ('parse_failure', 'fallback_substituted'):
        return (False, False, False)
    if canonical in ('unknown_action', 'illegal_in_state'):
        return (True, False, False)
    return (True, True, False)
