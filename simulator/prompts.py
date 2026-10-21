"""English prompt blocks aligned with Table 4 of the UserArena paper."""

from __future__ import annotations
from .data import top_category
import json
from typing import List
from . import geometry
from .environment import Observation
from .ipv_snapshot import render_snapshot


SYSTEM_INSTRUCTION = (
    "Role-play a real shopper, infer the next action from the supplied context, and return "
    "a concise first-person rationale followed by one structured action."
)
CURRENT_SCREENSHOT_NOTE = "The final screenshot is the current page."
SCREENSHOT_CONTEXT = (
    "You receive the current page screenshot. History is provided as text; some steps "
    "may also include their pre-action screenshots. " + CURRENT_SCREENSHOT_NOTE
)
TEXT_CONTEXT = "The current observation contains a textual page description, visible elements, and page state."
VISUAL_CONTEXT = (
    "The current observation contains a screenshot and an aligned textual page description, "
    "including visible elements and page state. Use both to interpret the current page."
)


def _action_space_block() -> str:
    """Action-space contract with pixel-valued distances on both screens."""
    return """# Action space
Actions are JSON objects. The current observation specifies which feed or product-page actions are executable.

## Feed
- Feed scroll down: {{"type": "scroll_down", "distance": <{low}-{high}>}}
- Feed scroll up: {{"type": "scroll_up", "distance": <{low}-{high}>}}
- Click item: {{"type": "click", "target": "item title"}}
- End session: {{"type": "end"}}

Feed scroll distance is an integer from {low} to {high} pixels.

## Product page
- Item scroll down: {{"type": "ipv_scroll_down", "distance": <{ipv_low}-{ipv_high}>}}
- Item scroll up: {{"type": "ipv_scroll_up", "distance": <{ipv_low}-{ipv_high}>}}
- Swipe image: {{"type": "ipv_swipe_pic"}}
- Click specification: {{"type": "ipv_click_param"}}
- Open reviews: {{"type": "ipv_enter_comment"}}
- Click review: {{"type": "ipv_click_comment", "target": "C1"}}
- Add to cart: {{"type": "ipv_cart"}}
- Buy: {{"type": "ipv_buy"}}
- Return to feed: {{"type": "back_home"}}
- End session: {{"type": "end"}}

Product-page scroll distance is an integer from {ipv_low} to {ipv_high} pixels, at most one viewport per action.""".format(
        low=geometry.MIN_SCROLL_DISTANCE,
        high=geometry.MAX_SCROLL_DISTANCE,
        ipv_low=geometry.MIN_SCROLL_DISTANCE,
        ipv_high=geometry.IPV_VIEWPORT_HEIGHT,
    )


OUTPUT_FORMAT_BLOCK = """# Response format
Return one JSON object containing a concise first-person rationale followed by one structured action:
{
  "rationale": "<one-sentence first-person rationale>",
  "action": { "type": "<action type>", ... }
}
The action must match one executable candidate. Output only the JSON object, with no extra text."""


def build_system_prompt(has_current_image: bool) -> str:
    """System prompt, worded to match the current image actually supplied."""
    context = "# Current environment\n" + (VISUAL_CONTEXT if has_current_image else TEXT_CONTEXT)
    screenshot_note = "\n# Screenshots\n" + SCREENSHOT_CONTEXT + "\n" if has_current_image else ""
    return "{instruction}\n\n{action_space}\n\n{context}\n{screenshot}\n{output_format}".format(
        instruction=SYSTEM_INSTRUCTION,
        action_space=_action_space_block(),
        context=context,
        screenshot=screenshot_note,
        output_format=OUTPUT_FORMAT_BLOCK,
    )


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text) * 1.6)


def build_action_candidates(record: dict) -> List[dict]:
    screen_type = record["screen_type"]
    viewport = record.get("scroll_viewport_height")
    if screen_type == "feed":
        maximum = geometry.MAX_SCROLL_DISTANCE
    else:
        maximum = int(viewport or geometry.IPV_VIEWPORT_HEIGHT)
    distance = "integer pixels from {} to {}".format(
        geometry.MIN_SCROLL_DISTANCE,
        maximum,
    )
    candidates = []
    for action in record["available_actions"]:
        if action == "click":
            candidates.extend(
                (
                    {"type": action, "target": title}
                    for title in record.get("click_targets") or ()
                )
            )
        elif action == "ipv_click_comment":
            candidates.extend(
                (
                    {"type": action, "target": alias}
                    for alias in record.get("comment_aliases") or ()
                )
            )
        elif action in ("scroll_down", "scroll_up", "ipv_scroll_down", "ipv_scroll_up"):
            candidates.append({"type": action, "distance": distance})
        else:
            candidates.append({"type": action})
    return candidates


PROFILE_ITEMS = 20


def history_titles(items):
    """Read ordered titles from histories prepared before release."""
    titles = [str(item.get("item_title") or item.get("title") or "").strip()
              for item in items or []]
    return [title for title in titles if title]


def profile_block(raw):
    info = raw.get("user_info") or {}
    fields = [
        f"{label}{info[key]}"
        for key, label in (("age", "Age: "), ("gender", "Gender: "), ("city", "City: "))
        if info.get(key)
    ]
    lines = ["# User context", "## User profile", ", ".join(fields) or "Unknown profile"]
    for key, label in (
        ("user_buy_list", "Recent purchases"),
        ("user_click_list", "Recent clicks"),
    ):
        titles = history_titles(raw.get(key))
        lines.append(
            f"## {label} (up to {PROFILE_ITEMS} recent titles; consecutive duplicates collapsed)\n"
            + ("\n".join(("- " + title for title in titles)) or "None")
        )
    return "\n".join(lines)


def system_prompt():
    return build_system_prompt(True)


def build_prompt(
    profile,
    history,
    observation,
    candidates,
    current_image,
    *,
    cutoff_len=32768,
    response="",
    tokenizer=None,
    history_actions=None,
    history_images=0,
    allow_missing_images=False,
    system_instruction=None,
    describe_missing_image=True,
):
    def text(value):
        return str(value).replace("<image>", "&lt;image&gt;")

    history = list(history)
    if history_actions is not None:
        history = history[-history_actions:] if history_actions else []
    for entry in history[-history_images:] if history_images else []:
        if not allow_missing_images and (not entry.get("image_key")):
            raise ValueError(f"Missing screenshot at step {entry['ordinal']}")
    if not allow_missing_images and (not current_image):
        raise ValueError("Current screenshot is missing")
    current_view = (
        "<image>\n"
        if current_image
        else "(No current screenshot is available; use the current textual observation.)\n"
        if describe_missing_image
        else ""
    )
    tail = (
        "# Current observation\n"
        + current_view
        + text(observation)
        + "\n# Executable actions\n"
        + text(json.dumps(candidates, ensure_ascii=False, indent=2))
    )
    tail += "\nChoose one of the JSON candidates above. For scrolling, replace distance with an integer in the permitted range.\nPredict your next action and provide its rationale."
    reserve = max(512, count_tokens(response, tokenizer))
    instructions = (
        system_instruction if system_instruction is not None else build_system_prompt(bool(current_image))
    )
    if not current_image:
        instructions = instructions.replace(
            "You receive the current page screenshot.", "No current screenshot is available."
        ).replace(
            CURRENT_SCREENSHOT_NOTE, "All attached screenshots belong to the labeled historical steps, not the current page."
        ).replace(VISUAL_CONTEXT, TEXT_CONTEXT)
    retained = list(history)
    while True:
        available = [i for i, entry in enumerate(retained) if entry.get("image_key")]
        image_indices = set(available[-history_images:] if history_images else [])
        entries = [
            "## Step {ordinal}\n### Observation\n{image}{observation}\n### Rationale\n{rationale}\n### Action\n{action}".format(
                **{k: text(entry[k]) for k in ("ordinal", "observation", "rationale")},
                image="<image>\n" if i in image_indices else "",
                action=text(json.dumps(entry["action_json"], ensure_ascii=False)),
            )
            for i, entry in enumerate(retained)
        ]
        profile_section = text(profile) + "\n" if profile else ""
        prompt = (
            instructions
            + "\n\n"
            + profile_section
            + "# Interaction history\n"
            + ("\n".join(entries) or "No prior actions.")
            + "\n"
            + tail
        )
        images = [
            entry["image_key"] for i, entry in enumerate(retained) if i in image_indices
        ]
        if current_image:
            images.append(current_image)
        tokens = (
            count_tokens(prompt, tokenizer) + len(images) * 1280 + reserve
        )
        if tokens <= cutoff_len:
            audit = {
                "history_actions": len(retained),
                "history_images": len(image_indices),
                "history_dropped_for_budget": len(history) - len(retained),
                "estimated_tokens": tokens,
            }
            if allow_missing_images:
                audit.update(
                    current_image_available=bool(current_image),
                    history_images_missing=sum(
                        (not entry.get("image_key") for entry in retained)
                    ),
                )
            return (prompt, images, audit)
        if not retained:
            raise ValueError("Current context exceeds cutoff_len without history")
        retained.pop(0)


CROSS_SESSION_INTRO = (
    SYSTEM_INSTRUCTION + "\nThis trajectory contains multiple visits at different times, each called a session. "
    "You receive your user context, up to 20 recent actions across visits, and the current observation. "
    "An end action finishes one visit; subsequent actions belong to the next visit."
)


def cross_session_instruction(has_current_image=True):
    base = system_prompt() if has_current_image else build_system_prompt(False)
    return base.replace(SYSTEM_INSTRUCTION, CROSS_SESSION_INTRO, 1)


def teacher_prompt(student_prompt: str, intent_summaries) -> str:
    """Append intent summaries to the teacher prompt."""
    summaries = [s.strip().replace("<image>", "&lt;image&gt;") for s in intent_summaries]
    return student_prompt + "\n\n# Trajectory-level intent (frozen teacher only; training only)\n" + "\n".join("- " + s for s in summaries)


def format_observation(observation: Observation) -> str:
    if observation.screen_type == "feed":
        lines = [
            "Feed (step {}, scroll position {}/{}). Currently visible items:".format(
                observation.step_number,
                observation.scroll_position,
                observation.max_scroll_y,
            )
        ]
        for index, item in enumerate(observation.visible_items, 1):
            lines.append(
                '{}. "{}" ¥{} ({})'.format(
                    index,
                    item.item_title,
                    item.item_price,
                    top_category(item.item_cate),
                )
            )
        return "\n".join(lines)
    if observation.snapshot is not None:
        return render_snapshot(observation.snapshot, observation.step_number)
    item = observation.current_item
    return 'Product page (step {}):\nItem: "{}"'.format(
        observation.step_number, item.item_title if item else ""
    )
