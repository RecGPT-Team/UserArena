"""Autoregressive model rollouts with bounded history and per-attempt traces."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

from simulator.session import ModelSession
from .prompts import (
    build_prompt,
    profile_block,
    system_prompt,
    cross_session_instruction,
)
from .session import load_records, cross_session_visits


@dataclass
class RolloutHistory:
    """Only observations and executed actions already seen by the model."""

    actions: deque
    images: dict = field(default_factory=dict)
    next_ordinal: int = 1


def strip_thinking_prefix(text):
    """Remove a complete leading <think>...</think> block."""
    leading = text.lstrip()
    if not leading.startswith("<think>"):
        return text
    body, closing, rest = leading[len("<think>") :].partition("</think>")
    if not closing or "<think>" in body:
        return text
    return rest.lstrip()


def rejected_action_feedback(info, current, action):
    valid = info.get("valid", True)
    if valid:
        return ""
    kind = action.get("type", "") if isinstance(action, dict) else ""
    at_top = getattr(current, "scroll_position", None) == 0
    layer = getattr(current, "screen_type", "")
    prefix = f"The previous action {kind} was not executed: " if kind else "The previous action was not executed: "
    if at_top and (layer, kind) in (("feed", "scroll_up"), ("ipv", "ipv_scroll_up")):
        return (
            prefix
            + "the page is already at the top. Its position is unchanged. Do not repeat scrolling up; choose an executable action."
        )
    position = getattr(current, "scroll_position", None)
    maximum = getattr(current, "max_scroll_y", None)
    if (
        layer == "feed"
        and kind == "scroll_down"
        and position is not None
        and maximum is not None
        and position >= maximum
    ):
        return (
            prefix
            + "the registered feed scroll boundary has been reached. The page position is unchanged. Choose an executable action."
        )
    allowed = getattr(current, "available_actions", ())
    if kind and allowed and kind not in allowed:
        return (
            prefix
            + "it is not in the current executable-action list. The page state is unchanged. Choose an executable action."
        )
    return (
        prefix
        + "the output failed action or target validation. The page state is unchanged. Check the format and target, then choose an executable action."
    )


def _capture(folder, index, prompt, keys, images, observation):
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for slot, key in enumerate(keys):
        name = f"{index:03d}-{slot:02d}.png"
        images[key].save(folder / name)
        names.append(name)
    (folder / f"{index:03d}-input.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "images": names,
                "before": observation.trace_state(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def rollout_session(
    raw,
    env,
    generate,
    *,
    cutoff_len=32768,
    tokenizer=None,
    history_actions=10,
    history_images=3,
    history_state=None,
    system_instruction=None,
    max_try=1,
    retry_feedback=True,
    capture_dir=None,
):
    """generate(prompt, PIL_images) returns (text, input_tokens, output_tokens)."""
    state = (
        history_state
        if history_state is not None
        else RolloutHistory(deque(maxlen=history_actions))
    )
    history, images = state.actions, state.images
    instruction = system_instruction or system_prompt()
    profile = profile_block(raw)
    generations = []
    error = pending_feedback = ""
    try:
        obs, _ = env.reset()
        env._rollout.count_invalid_actions = False
        executed_steps = failed_tries = 0
        for ordinal in range(1, env.max_steps * max_try + 1):
            current = env._rollout.observe()
            key = str(state.next_ordinal)
            if not obs["image"]:
                raise ValueError("current_screenshot_missing")
            images[key] = obs["image"][0]
            history_observation = observation = env._observation_text(current)
            if pending_feedback:
                observation += "\n# Previous action feedback\n" + pending_feedback
            prompt, keys, context_audit = build_prompt(
                profile,
                history,
                observation,
                env._action_candidates(current),
                key,
                cutoff_len=cutoff_len,
                tokenizer=tokenizer,
                history_actions=history_actions,
                history_images=history_images,
                system_instruction=instruction,
            )
            if capture_dir is not None:
                _capture(Path(capture_dir), ordinal, prompt, keys, images, current)
            start = time.monotonic()
            raw_text, input_tokens, output_tokens = generate(
                prompt, [images[k] for k in keys]
            )
            text = strip_thinking_prefix(raw_text)
            generations.append(
                {
                    "index": ordinal,
                    "raw_output": raw_text,
                    "action_output": text,
                    "thinking_prefix_removed": text != raw_text,
                    "context": context_audit,
                }
            )
            env.set_step_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_calls=1,
                latency_ms=(time.monotonic() - start) * 1000,
            )
            try:
                label = json.loads(text)
                if not isinstance(label, dict):
                    label = {}
            except ValueError:
                label = {}
            action_step, attempt = executed_steps + 1, failed_tries + 1
            obs, _, terminated, truncated, info = env.step(text)
            feedback = rejected_action_feedback(info, current, label.get("action"))
            pending_feedback = feedback if retry_feedback else ""
            valid = info.get("valid", True)
            if valid:
                executed_steps += 1
                failed_tries = 0
            else:
                failed_tries += 1
            if failed_tries >= max_try:
                error = "invalid_action_retry_exhausted"
                terminated, truncated = False, True
                env.set_episode_status(False, True, error)
            generations[-1].update(
                execution_feedback=feedback,
                action_step=action_step,
                attempt_in_state=attempt,
                executed=bool(valid),
            )
            if capture_dir is not None:
                (Path(capture_dir) / f"{ordinal:03d}-output.json").write_text(
                    json.dumps(
                        {
                            **generations[-1],
                            "after": env._rollout.observe().trace_state(),
                            "terminated": terminated,
                            "truncated": truncated,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if valid:
                history.append({
                    "ordinal": state.next_ordinal,
                    "observation": history_observation,
                    "rationale": label.get("rationale", ""),
                    "action_json": label["action"],
                    "image_key": key,
                })
                state.next_ordinal += 1
            retained = list(history)[-history_images:] if history_images else []
            images = {
                entry["image_key"]: images[entry["image_key"]] for entry in retained
            }
            state.images = images
            if terminated or truncated or executed_steps >= env.max_steps:
                break
    except Exception as exc:
        # Sanitize errors for public traces.
        error = type(exc).__name__
        if str(exc) == "current_screenshot_missing":
            error += ": " + str(exc)
        env.set_episode_status(False, True, "evaluation_error")
    state.images = {
        entry["image_key"]: images[entry["image_key"]]
        for entry in (list(history)[-history_images:] if history_images else [])
        if entry.get("image_key") in images
    }
    trace = env.trace()
    trace.update(
        generation_outputs=generations,
        retry_policy={
            "feedback_to_model": retry_feedback,
            "max_try": max_try,
            "invalid_consumes_action_step": False,
        },
    )
    if error:
        trace["error"] = error
    return trace


def inspect_dataset(path, *, cross_session=False, cohort_schemes=None):
    """Check identities and required metadata before loading a model."""
    from .data import Session
    from evaluation.reference import validate_cohort_fields

    seen = set()
    users = max_decisions = 0
    for raw in load_records(path):
        if ("pv_ids" in raw) != cross_session:
            raise ValueError("Dataset structure and --cross-session disagree")
        visits = cross_session_visits(raw) if cross_session else [raw]
        users += 1
        for visit in visits:
            session = Session.from_raw(visit)
            if not session.session_id or session.session_id in seen:
                raise ValueError("Session IDs must be nonempty and unique")
            seen.add(session.session_id)
            decisions = sum(bool(step.action) for step in session.trajectory)
            max_decisions = max(max_decisions, decisions)
    if not seen:
        raise ValueError("Dataset has no sessions")
    validate_cohort_fields(path, cohort_schemes)
    return {
        "users": users,
        "sessions": len(seen),
        "max_reference_decisions": max_decisions,
    }


def run_dataset(
    path,
    output_dir,
    generate,
    *,
    image_store,
    tokenizer=None,
    model_label="OPTD",
    cross_session=False,
    max_actions=None,
    history_actions=None,
    history_images=None,
    cutoff_len=32768,
    max_try=1,
    retry_feedback=True,
    capture_cases=False,
    min_cohort=30,
    cohort_schemes=None,
    provenance=None,
):
    """Persist each episode immediately, then score the complete population."""
    from evaluation.report import run_rollout_evaluation
    from evaluation.report import write_report

    model_label = model_label.strip()
    if max_actions is None:
        max_actions = 42 if cross_session else 31
    if history_actions is None:
        history_actions = 20 if cross_session else 10
    if history_images is None:
        history_images = 5 if cross_session else 3
    audit = inspect_dataset(
        path,
        cross_session=cross_session,
        cohort_schemes=cohort_schemes,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    instruction = cross_session_instruction() if cross_session else None
    settings = {
        "model": model_label,
        "cross_session": cross_session,
        "max_actions": max_actions,
        "history_actions": history_actions,
        "history_images": history_images,
        "cutoff_len": cutoff_len,
        "max_try": max_try,
        "retry_feedback": retry_feedback,
        "dataset": audit,
    }
    if provenance:
        settings["inference"] = provenance
    write_report(destination / "run.json", settings)
    traces = []
    endings = Counter()
    with (destination / "simulated.jsonl").open("x", encoding="utf-8") as output:
        for number, raw in enumerate(load_records(path), 1):
            visits = cross_session_visits(raw) if cross_session else [raw]
            state = (
                RolloutHistory(deque(maxlen=history_actions)) if cross_session else None
            )
            for visit_index, visit in enumerate(visits):
                env = ModelSession(
                    visit, image_store=image_store, max_steps=max_actions
                )
                trace = rollout_session(
                    visit,
                    env,
                    generate,
                    cutoff_len=cutoff_len,
                    tokenizer=tokenizer,
                    history_actions=history_actions,
                    history_images=history_images,
                    history_state=state,
                    system_instruction=instruction,
                    max_try=max_try,
                    retry_feedback=retry_feedback,
                    capture_dir=destination
                    / "cases"
                    / f"{number:05d}-{visit_index:03d}"
                    if capture_cases
                    else None,
                )
                trace["model"] = model_label
                output.write(json.dumps(trace, ensure_ascii=False) + "\n")
                output.flush()
                traces.append(trace)
                endings[trace["termination_reason"] or "unspecified"] += 1
                print(
                    f"Completed {len(traces)}/{audit['sessions']}: "
                    f"{len(trace['steps'])} attempts, {trace['termination_reason'] or 'ended'}"
                    + (f" ({trace['error']})" if trace["error"] else ""),
                    flush=True,
                )
    payload = run_rollout_evaluation(
        traces,
        Path(path),
        destination,
        model_label,
        min_cohort=min_cohort,
        cohort_schemes=cohort_schemes,
        provenance=settings,
    )
    write_report(destination / "endings.json", dict(endings))
    return payload
