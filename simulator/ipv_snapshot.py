"""Structured product-page observations and text rendering."""

from __future__ import annotations
from .data import top_category

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import geometry

SUBPAGE_MAIN = "main"
SUBPAGE_COMMENT_LIST = "comment_list"

ACTION_LABELS = {
    "ipv_scroll_down": "Item scroll down", "ipv_scroll_up": "Item scroll up",
    "ipv_swipe_pic": "Swipe image",
    "ipv_click_param": "Click specification",
    "ipv_enter_comment": "Open reviews", "ipv_click_comment": "Click review",
    "ipv_cart": "Add to cart", "ipv_buy": "Buy",
}

@dataclass(frozen=True)
class ReviewEntry:
    """One review, addressable by a short alias in the observation."""

    review_index: int
    content: str

    @property
    def alias(self) -> str:
        return review_alias(self.review_index)

@dataclass(frozen=True)
class VisibleBlock:
    """A content block currently inside the viewport, with its rendered text."""

    kind: str
    text: str
    ordinal: int = 0

@dataclass(frozen=True)
class PageSnapshot:
    """Policy-visible IPV state backed by one earlier logged step."""

    item_id: str
    title: str
    source_step_index: int
    gui_image: str
    price: object = None
    category: str = ""
    subpage: str = SUBPAGE_MAIN
    scroll_y: int = 0
    total_height: int = 0
    viewport_height: int = geometry.IPV_VIEWPORT_HEIGHT
    visible: Sequence[VisibleBlock] = field(default_factory=tuple)
    hero_index: Optional[int] = None
    params_open: bool = False
    parameters: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    sku_open: bool = False
    sku_options: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    selected_cta: str = ""
    reviews: Sequence[ReviewEntry] = field(default_factory=tuple)
    action_history: Sequence[Tuple[str, int]] = field(default_factory=tuple)

    @property
    def screen_position(self) -> float:
        return self.scroll_y / self.viewport_height + 1

    @property
    def screen_count(self) -> float:
        return max(1.0, self.total_height / self.viewport_height)

    @property
    def screens_below(self) -> float:
        return max(0.0, self.screen_count - self.screen_position)

def visible_review_aliases(page_snapshot: Optional[PageSnapshot]) -> Tuple[str, ...]:
    if page_snapshot is None:
        return ()
    if page_snapshot.subpage == SUBPAGE_COMMENT_LIST:
        return tuple(review.alias for review in page_snapshot.reviews)
    if page_snapshot.subpage != SUBPAGE_MAIN:
        return ()
    visible = {
        review.alias
        for review in page_snapshot.reviews
        if any(
            "{}: \"".format(review.alias) in block.text
            for block in page_snapshot.visible
        )
    }
    return tuple(
        review.alias for review in page_snapshot.reviews if review.alias in visible
    )

def _price_text(price: object) -> str:
    if price is None or price == "":
        return "Price unavailable"
    try:
        value = float(price)
    except (TypeError, ValueError):
        return "¥{}".format(price)
    return "¥{:g}".format(value)


def render_visible_block(
    block: geometry.Block,
    snapshot_parts: dict,
) -> Optional[VisibleBlock]:
    """Turn a laid-out block into its observation text."""
    kind = block.kind
    if kind == geometry.BLOCK_HERO:
        hero_index = snapshot_parts.get("hero_index")
        if hero_index is None:
            return VisibleBlock(kind, "Main product image")
        return VisibleBlock(kind, "Main image: {}".format(hero_index + 1))
    if kind == geometry.BLOCK_PRICE:
        price = snapshot_parts.get("price")
        return VisibleBlock(kind, "Price: {}".format(_price_text(price))) if price not in (None, "") else None
    if kind == geometry.BLOCK_SKU:
        sku = snapshot_parts.get("sku") or ()
        if not sku:
            return None
        summary = " ".join("{} {}".format(label, value) for label, value in sku)
        return VisibleBlock(kind, "Selected options: {}".format(summary))
    if kind == geometry.BLOCK_PARAMS:
        parameters = snapshot_parts.get("parameters") or ()
        if not parameters:
            return None
        preview = " / ".join(
            "{} {}".format(label, value) for label, value in parameters[:3]
        )
        return VisibleBlock(kind, "Specifications entry: {} ({} entries)".format(
            preview, len(parameters)))
    if kind == geometry.BLOCK_REVIEW:
        reviews = snapshot_parts.get("reviews") or ()
        if not reviews:
            return None
        first = reviews[0]
        return VisibleBlock(kind, "Review preview: {}: \"{}\"".format(first.alias, first.content))
    return None

def render_snapshot(snapshot: PageSnapshot, step_number: int) -> str:
    """Serialize a snapshot into the observation text shown to the policy."""
    if snapshot.subpage == SUBPAGE_COMMENT_LIST:
        return _render_comment_list(snapshot, step_number)
    return _render_main(snapshot, step_number)

def _header(snapshot: PageSnapshot, step_number: int, page_label: str) -> List[str]:
    lines = ["{} (step {})".format(page_label, step_number)]
    facts = ["Item: \"{}\"".format(snapshot.title)]
    if snapshot.price not in (None, ""):
        facts.append(_price_text(snapshot.price))
    if snapshot.category:
        facts.append("Category: {}".format(top_category(snapshot.category)))
    lines.append("  ".join(facts))
    return lines

def _render_main(snapshot: PageSnapshot, step_number: int) -> str:
    lines = _header(snapshot, step_number, "Product page")
    state_facts = []
    if snapshot.total_height:
        state_facts.append(
            "Page position: {:.1f} of {:.1f} viewports ({:.1f} below)".format(
                snapshot.screen_position, snapshot.screen_count, snapshot.screens_below
            )
        )
    hero_visible = any(
        block.kind == geometry.BLOCK_HERO for block in snapshot.visible
    )
    if hero_visible and snapshot.hero_index is not None:
        state_facts.append("Currently visible main image: {}".format(snapshot.hero_index + 1))
    if state_facts:
        lines.append("; ".join(state_facts))

    if snapshot.visible:
        lines.extend(["", "Visible in the current viewport:"])
        lines.extend("- {}".format(block.text) for block in snapshot.visible)

    if snapshot.params_open:
        lines.append("")
        if snapshot.parameters:
            lines.append("Specifications panel (open, {} entries):".format(len(snapshot.parameters)))
            lines.extend("- {}: {}".format(label, value) for label, value in snapshot.parameters)
        else:
            lines.append("Specifications panel: open (no structured specification text)")

    if snapshot.sku_open:
        lines.append("")
        if snapshot.sku_options:
            lines.append("Options panel (open):")
            lines.extend("- {}: {}".format(label, value) for label, value in snapshot.sku_options)
        else:
            lines.append("Options panel: open (no structured option text)")
    elif snapshot.sku_options:
        lines.append("Selected options: {}".format("; ".join(
            "{}: {}".format(label, value) for label, value in snapshot.sku_options
        )))

    if snapshot.selected_cta:
        lines.append("Last selected: {}".format(
            "Add to cart" if snapshot.selected_cta == "cart" else "Buy"
        ))
    if snapshot.selected_cta == "cart":
        lines.append("Fixed bottom action bar: Add to cart (added) / Buy")
    elif snapshot.selected_cta == "buy":
        lines.append("Fixed bottom action bar: Add to cart / Buy (purchased)")
    else:
        lines.append("Fixed bottom action bar: Add to cart / Buy")
    lines.extend(_history_lines(snapshot))
    return "\n".join(lines)

def _history_lines(snapshot: PageSnapshot) -> List[str]:
    """Compact record of what the agent already did to this product."""
    if not snapshot.action_history:
        return []
    parts = []
    for action, count in snapshot.action_history:
        label = ACTION_LABELS[action]
        parts.append(label if count == 1 else "{}×{}".format(label, count))
    return ["Actions already taken on this item: {}".format(", ".join(parts))]

def _render_comment_list(snapshot: PageSnapshot, step_number: int) -> str:
    lines = _header(snapshot, step_number, "Product review list")
    if snapshot.reviews:
        lines.append("Currently visible reviews:")
        lines.extend(
            "- {}: \"{}\"".format(review.alias, review.content)
            for review in snapshot.reviews
        )
    else:
        lines.append("Review list (no structured reviews)")
    lines.extend(_history_lines(snapshot))
    return "\n".join(lines)

def review_alias(index: int) -> str:
    """Short, copy-safe alias for a review (``C1``, ``C2``, ...)."""
    return "C{}".format(index + 1)

def build_reviews(items: Sequence[str]) -> Tuple[ReviewEntry, ...]:
    """Preserve prepared order, including distinct reviews with identical text."""
    return tuple(ReviewEntry(review_index=index, content=content)
                 for index, content in enumerate(items))
