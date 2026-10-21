"""Prepared session records and catalog-backed page states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import geometry
from .images import Capture, ImageRef, image_ref


def top_category(value):
    return str(value or "").split("-")[0]


def _parameter_pairs(raw_params) -> tuple[tuple[str, str], ...]:
    return tuple((entry["label"].strip(), entry["value"].strip()) for entry in (raw_params or ()))

@dataclass
class FeedItem:
    item_id: str
    item_title: str
    item_price: object
    item_cate: str
    bbox: dict
    item_reviews: tuple = field(default_factory=tuple)
    parameters: tuple = field(default_factory=tuple)


    @classmethod
    def from_raw(cls, raw: dict) -> "FeedItem":
        return cls(
            item_id=str(raw.get("item_id") or ""),
            item_title=str(raw.get("item_title") or ""),
            item_price=raw.get("item_price", 0),
            item_cate=str(raw.get("item_cate") or ""),
            bbox=dict(raw["bbox"]),
            item_reviews=tuple(review.strip() for review in raw.get("item_reviews") or ()),
            parameters=_parameter_pairs(raw.get("item_params")),
        )

def _delimited_pairs(value: object) -> tuple:
    pairs = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, _, item = part.partition(":")
        label, item = label.strip(), item.strip()
        if label and item:
            pairs.append((label, item))
    return tuple(pairs)

@dataclass(frozen=True)
class IpvState:
    """Product-page state reconstructed from a logged observation."""

    source_step_index: int
    item_id: str
    title: str
    price: object = None
    category: str = ""
    gui_image: ImageRef = ""
    ui_state: tuple = field(default_factory=tuple)
    parameters: tuple = field(default_factory=tuple)
    sku_options: tuple = field(default_factory=tuple)
    reviews: tuple = field(default_factory=tuple)

    @classmethod
    def from_step(
        cls,
        step: "TrajectoryStep",
        source_step_index: int,
        feed_item: FeedItem,
    ) -> "IpvState":
        return cls(
            source_step_index=source_step_index,
            item_id=feed_item.item_id,
            title=feed_item.item_title,
            price=feed_item.item_price,
            category=feed_item.item_cate,
            gui_image=step.gui_image,
            ui_state=tuple(sorted(step.ipv_step_ui_state.items())),
            parameters=feed_item.parameters,
            sku_options=_delimited_pairs(step.property_values),
            reviews=feed_item.item_reviews,
        )

    def ui(self, key: str, default=None):
        return dict(self.ui_state).get(key, default)

    @property
    def subpage(self) -> str:
        return "comment_list" if self.ui("reviewExpanded") is True else "main"

    @property
    def params_open(self) -> bool:
        return self.ui("paramsOpen") is True

    @property
    def sku_open(self) -> bool:
        return self.ui("skuOpen") is True

    @property
    def hero_index(self) -> Optional[int]:
        return self.ui("heroIndex")

    @property
    def selected_cta(self) -> str:
        return self.ui("selectedCta", "")

@dataclass
class TrajectoryStep:
    screen_type: str
    action: str
    clicked_item_id: Optional[str] = None
    item_id: Optional[str] = None
    item_title: str = ""
    visible_items: List[str] = field(default_factory=list)

    review_index: Optional[int] = None
    gui_image: ImageRef = ""
    ipv_step_ui_state: Dict[str, object] = field(default_factory=dict)
    property_values: str = ""


    feed_scroll_distance: Optional[float] = None
    feed_scroll_top: Optional[float] = None

    ipv_scroll_distance: Optional[float] = None

    @classmethod
    def from_raw(cls, raw: dict) -> "TrajectoryStep":
        return cls(
            screen_type=raw["step_type"],
            action=raw.get("action") or "",
            clicked_item_id=raw.get("clicked_item_id"),
            item_id=raw.get("item_id"),
            item_title=str(raw.get("item_title") or ""),
            visible_items=list(raw.get("visible_items") or []),
            review_index=raw.get("review_index"),
            gui_image=image_ref(raw.get("gui_image")),
            ipv_step_ui_state=dict(raw.get("ipv_step_ui_state") or {}),
            property_values=str(raw.get("property_values") or ""),
            feed_scroll_distance=raw.get("scroll_delta_px"),
            feed_scroll_top=raw.get("scroll_top"),
            ipv_scroll_distance=raw.get("scroll_distance"),
        )

    @property
    def scroll_y(self) -> Optional[float]:
        """This step's action-before Feed offset, or None when unknowable."""
        return float(self.feed_scroll_top) if self.feed_scroll_top is not None else None

    @property
    def frame_ordinal(self) -> Optional[int]:
        """Index in ``render_step_NNNN.png``, parsed rather than re-counted."""
        if not isinstance(self.gui_image, str) or not self.gui_image:
            return None
        stem = self.gui_image.rsplit("/", 1)[-1].split(".", 1)[0]
        _, _, digits = stem.rpartition("_")
        return int(digits) if digits.isdigit() else None

def feed_catalog_items(raw: dict) -> List[dict]:
    """Read catalog items with the fixed Feed card width."""
    return [{**item, "bbox": {"width": 196.5, **item["bbox"]}}
            for item in raw["feed_catalog"]]


def feed_page_geometry(raw: dict) -> dict:
    """Frozen pixel geometry of the same 402px, scale-1 Feed document."""
    image = raw["feed_long_image"]
    content_height = (int(geometry.SCREEN_HEIGHT) - round(geometry.CHROME_TOP)
                      - round(geometry.CHROME_BOTTOM))
    return {"viewport_height": content_height,
            "max_scroll_y": max(0, image["height"] - content_height)}


def _extract_feed_items(catalog: List[dict]) -> List[FeedItem]:
    items = [FeedItem.from_raw(raw) for raw in catalog]
    items.sort(key=lambda item: (item.bbox["y"], item.bbox["x"]))
    return items

@dataclass
class Session:
    session_id: str
    visitor_id: str
    user_info: Dict[str, object]
    user_click_list: List[dict]
    user_buy_list: List[dict]
    trajectory: List[TrajectoryStep]
    termination_reason: str = ""
    feed_items: List[FeedItem] = field(default_factory=list)
    gt_trajectory: List[TrajectoryStep] = field(default_factory=list)

    history_counts: Dict[str, int] = field(default_factory=dict)

    dropped: Dict[str, int] = field(default_factory=dict)
    feed_catalog: List[dict] = field(default_factory=list, repr=False)
    feed_geometry: dict = field(default_factory=dict)
    _item_index: Dict[str, FeedItem] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from .replay import prepare_trajectory

        self.feed_items = _extract_feed_items(self.feed_catalog)
        self._item_index = {item.item_id: item for item in self.feed_items}
        self._capture_index = None
        self.gt_trajectory = prepare_trajectory(self.trajectory, self, self.dropped)

    @classmethod
    def from_raw(cls, raw: dict) -> "Session":
        trajectory = [TrajectoryStep.from_raw(step) for step in raw["trajectory"]]
        raw_clicks = raw.get("user_click_list") or []
        raw_buys = raw.get("user_buy_list") or []
        supplied_counts = raw.get("history_counts", {})
        history_counts = {name: supplied_counts.get(name, len(items))
                          for name, items in (("clicks", raw_clicks), ("buys", raw_buys))}
        catalog = feed_catalog_items(raw)
        return cls(
            session_id=str(raw.get("session_id") or raw.get("pv_id") or ""),
            visitor_id=str(raw.get("visitor_id") or ""),
            user_info=dict(raw.get("user_info") or {}),
            user_click_list=recent_history(raw_clicks, 20),
            user_buy_list=recent_history(raw_buys, 20),
            trajectory=trajectory,
            termination_reason=str(raw.get("termination_reason") or ""),
            feed_catalog=catalog,
            feed_geometry=feed_page_geometry(raw),
            history_counts=history_counts,
        )

    @property
    def landing_states(self):
        states = {}
        for raw in self.feed_catalog:
            item = self._item_index[str(raw["item_id"])]
            states[item.item_id] = IpvState(
                source_step_index=-1, item_id=item.item_id,
                title=item.item_title, price=item.item_price, category=item.item_cate,
                gui_image=image_ref(raw["ipv_long_image"]),
                parameters=item.parameters, reviews=item.item_reviews,
            )
        return states

    def get_item(self, item_id: Optional[str]) -> Optional[FeedItem]:
        return self._item_index.get(str(item_id)) if item_id else None

    @property
    def ipv_captures(self) -> Dict[str, "Capture"]:
        """``item_id -> Capture`` for every detail page this session visited."""
        if self._capture_index is None:
            self._capture_index = {
                str(item["item_id"]): Capture.from_meta(item["ipv_long_image"], item["ipv_capture_meta"])
                for item in self.feed_catalog if item.get("ipv_capture_meta")
            }
        return self._capture_index


    @property
    def initial_feed_scroll_y(self) -> Optional[float]:
        """Scroll offset the session's first Feed step was taken from."""
        for step in self.trajectory:
            if step.screen_type != "feed":
                continue
            offset = step.scroll_y
            if offset is not None:
                return offset
        return None


    @property
    def has_logged_ipv_states(self) -> bool:
        return any(step.screen_type == "ipv" for step in self.trajectory)


    @property
    def structural_defects(self) -> Dict[str, int]:
        """``reason -> count`` for prepared steps no state change can explain."""
        from .replay import structural_defects

        return structural_defects(self.gt_trajectory)


def recent_history(items: Sequence[dict], limit: int = 20) -> List[dict]:
    """Return the latest items in chronological order (oldest to newest)."""
    chronological = sorted(items, key=lambda item: item.get("timestamp", 0))
    return chronological[-limit:]
