# UserArena

Code for *UserArena: Benchmarking Interactive User Simulation for Long-Horizon
Shopping Agents*: environment, model rollout, evaluation, and On-Policy Trajectory
Distillation (OPTD). License: [MIT](LICENSE).

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Data and environment

The held-out evaluation set contains 535 sessions in [data/sessions.jsonl](data/sessions.jsonl).

| Field | Description |
| --- | --- |
| `session_id`, `visitor_id` | Session and visitor IDs |
| `user_info`, `user_click_list`, `user_buy_list` | Profile and histories |
| `history_counts` | Pre-session counts for L3 cohorts |
| `feed_long_image` | Feed long image: `url` and pixel `height` |
| `feed_catalog` | Product array: titles, prices, categories, bboxes, parameters, reviews and images |
| `trajectory` | Pre-action states, actions, scroll offsets/distances and UI state |

Catalog `item_id` values (`item1`–`item8957`) are shared across sessions.
`trajectory[].gui_image` references a per-step screenshot, separate from `feed_long_image.url`.
`trajectory[].visible_items` is an ordered array of catalog IDs, e.g. `["item1", "item2"]`.
Catalog `item_params` retains `label`/`value` objects; `item_reviews` contains strings,
with zero-based `review_index` targets. Rows without `action` are observations.
Empty `item_params`, `item_reviews` and `user_buy_list` arrays are omitted.
Cross-session records follow `pv_ids`, with per-visit data in `feed_session_screenshots[]`.

```python
from pathlib import Path
from simulator import ActionPrediction, ImageStore, ShoppingSession, load_records

root = Path("data")
session = ShoppingSession(next(load_records(root / "sessions.jsonl")),
                          image_store=ImageStore(root))
observation = session.reset()
image = session.render()
result = session.step(ActionPrediction("end"))
```

## Rollout and scoring

```bash
python -m evaluation run \
  --model-path /path/to/checkpoint --model-label OPTD \
  --data data/sessions.jsonl --output-dir outputs/rollout --history-images 0

python -m evaluation rollout \
  --data data/sessions.jsonl --traces outputs/rollout/simulated.jsonl \
  --output-dir outputs/rescored --model-label OPTD

python -m evaluation replay --traces outputs/replay.jsonl \
  --output-dir outputs/replay-scores
```

Use new/empty output directories. Main outputs are `simulated.jsonl`, `traces.jsonl`,
`report.json` and `headline.json`; `replay` scores existing traces.
Responses use `{"rationale": "...", "action": {...}}`. Rejected responses remain in
traces without consuming the action budget; `--max-try` limits consecutive failures.
See `python -m evaluation run --help`.


| Subcommand | Table 2 metrics |
| --- | --- |
| `replay` | RVA, macro F1, Acc (non-scroll macro exact accuracy) |
| `rollout` | OVA, DM-JSD, TE, PF, ΔCTR, ΔACR, T-JSD |

## OPTD

`optd/objective.py` implements the trust-region teacher, generalized JSD and paired
grounding. Initialize separate student and frozen teacher models from the same SFT checkpoint.

```python
import torch
from optd import action_class_weights
from optd.adapter import OptdAdapter, TorchBackend, TrainingState

student = TorchBackend(student_model, prepare=prepare)
teacher = TorchBackend(teacher_model, prepare=prepare)
adapter = OptdAdapter(student, teacher)
optimizer = torch.optim.Adam(student_model.parameters(), lr=2e-8)
states = [TrainingState(session_id, context, privileged_context, gold_token_ids)]
metrics = adapter.step(states, optimizer,
                       gold_weights=action_class_weights(gt_actions, action_frequencies))
```

The caller supplies `prepare(context, response)` returning `PreparedInput(inputs, response_start)`,
preserving response IDs including EOS and rebuilding image/mask/position inputs.
One `step()` updates one effective batch. The example uses optional GT action-class
weights (mean 1, cap 2); omit `gold_weights` for the unweighted GT term in Eq. 10.
KD is state-equal and the GT coefficient is 0.1.

`optd/framework.py` provides `roll_loss(LossConfig())` and `verl_loss(LossConfig())`.
Both need full-vocabulary logits. Compute `paired_row_weights(state_count, gold_weights=weights)`
before splitting the effective update; the engine owns backward and synchronization.
See docstrings for batch fields and `LossConfig.backward_scale`.
