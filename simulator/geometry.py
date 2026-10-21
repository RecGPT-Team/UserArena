"""Feed card geometry and product-page layouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SCREEN_WIDTH = 402
SCREEN_HEIGHT = 874

# Logged offsets; rendering rounds the white margins to 127/54 px.
CHROME_TOP = 126.734
CHROME_BOTTOM = 53.766

FEED_VIEWPORT_HEIGHT = SCREEN_HEIGHT - CHROME_TOP - CHROME_BOTTOM

SCROLL_STEP = 331

MIN_SCROLL_DISTANCE = 1
MAX_SCROLL_DISTANCE = int(round(FEED_VIEWPORT_HEIGHT))

EXPOSURE_VISIBLE = 0.5

DEFAULT_CARD_HEIGHT = 328.0

def visible_height(
    bbox: Optional[dict], scroll_y: float, viewport_height: float = FEED_VIEWPORT_HEIGHT
) -> float:
    """Height of the intersection between a card and the content viewport."""
    if not bbox:
        return 0.0
    top = float(bbox.get("y", 0.0))
    height = float(bbox.get("height", DEFAULT_CARD_HEIGHT))
    view_top = float(scroll_y)
    view_bottom = view_top + float(viewport_height)
    return max(0.0, min(top + height, view_bottom) - max(top, view_top))

def exposure_ratio(
    bbox: Optional[dict], scroll_y: float, viewport_height: float = FEED_VIEWPORT_HEIGHT
) -> float:
    """Fraction of the card inside the content viewport, the log's ``expo_ratio``."""
    if not bbox:
        return 0.0
    height = float(bbox.get("height", DEFAULT_CARD_HEIGHT)) or 1.0
    return visible_height(bbox, scroll_y, viewport_height) / height

def is_visible(
    bbox: Optional[dict], scroll_y: float, viewport_height: float = FEED_VIEWPORT_HEIGHT
) -> bool:
    """Whether the card is exposed enough to be described to the agent."""
    return exposure_ratio(bbox, scroll_y, viewport_height) >= EXPOSURE_VISIBLE

def is_clickable(
    bbox: Optional[dict], scroll_y: float, viewport_height: float = FEED_VIEWPORT_HEIGHT
) -> bool:
    """Whether any pixel of the card is on screen, so a tap could land on it."""
    return visible_height(bbox, scroll_y, viewport_height) > 0.0


IPV_VIEWPORT_HEIGHT = 837

BLOCK_HERO = "hero"
BLOCK_PRICE = "price"
BLOCK_SKU = "sku"
BLOCK_PARAMS = "params_entry"
BLOCK_REVIEW = "review"
BLOCK_DETAIL_HEADING = "detail_heading"
BLOCK_DETAIL_IMAGE = "detail_image"

@dataclass(frozen=True)
class Block:
    """One laid-out content block, as measured in the browser."""

    kind: str
    top: int
    height: int

    ordinal: int = 0

    @property
    def bottom(self) -> int:
        return self.top + self.height

@dataclass(frozen=True)
class PageLayout:
    """Absolute vertical layout of one product's IPV page."""

    blocks: Sequence[Block]
    total_height: int
    viewport_height: int = IPV_VIEWPORT_HEIGHT

    @property
    def max_scroll_y(self) -> int:
        return max(0, self.total_height - self.viewport_height)

    def visible_blocks(self, scroll_y: int) -> List[Block]:
        """Blocks intersecting the viewport at ``scroll_y``."""
        top = max(0, scroll_y)
        bottom = top + self.viewport_height
        visible = []
        for block in self.blocks:
            overlap = min(block.bottom, bottom) - max(block.top, top)
            if overlap <= 0:
                continue
            visible.append(block)
        return visible

    @classmethod
    def from_measurement(cls, entry: dict) -> "PageLayout":
        """Build a layout from one browser-measured manifest entry."""
        blocks = tuple(
            Block(
                kind=str(item["kind"]),
                top=int(item["top"]),
                height=int(item["height"]),
                ordinal=int(item.get("ordinal") or 0),
            )
            for item in entry.get("blocks") or ()
        )
        return cls(
            blocks=blocks,
            total_height=int(entry["total_height"]),
            viewport_height=int(entry.get("viewport_height") or IPV_VIEWPORT_HEIGHT),
        )

def load_layout_manifest(path) -> Dict[str, PageLayout]:
    """Load ``item_id -> PageLayout`` from a measured manifest file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(item_id): PageLayout.from_measurement(entry)
        for item_id, entry in raw.items()
        if entry.get("total_height")
    }

def with_measured_blocks(
    layout: "PageLayout", measured: Optional["PageLayout"],
    tolerance: int = 93,
) -> tuple:
    """``(layout, adopted)``: block positions added to a captured page's height."""
    if measured is None or not measured.blocks:
        return layout, False
    if measured.total_height <= 0:
        return layout, False
    if abs(measured.total_height - layout.total_height) > tolerance:
        return layout, False
    return (
        PageLayout(
            blocks=measured.blocks,
            total_height=layout.total_height,
            viewport_height=layout.viewport_height,
        ),
        True,
    )
