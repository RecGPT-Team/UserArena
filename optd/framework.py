"""ROLL/verl loss hooks; the engine owns sampling, backward and updates.

Batch fields: optd_target_ids and optd_response_mask are [rows, response];
optd_response_start holds offsets including left padding. optd_kind and
optd_weight are [rows], with sampled=0 / GT=1 and update-wide weights.
optd_teacher_base_logits and optd_teacher_intent_logits are frozen
[rows, response, vocabulary] logits, omitted for GT-only batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .objective import generalized_jsd_loss, supervised_action_loss, trust_region_teacher


@dataclass(frozen=True)
class LossConfig:
    beta: float = 0.5
    gold_coefficient: float = 0.1
    trust_region_kl: float = 0.02
    # Undo the engine's DP/microbatch averaging (1 for summed gradients).
    backward_scale: float = 1.0


def paired_row_weights(state_count: int, gold_weights=None):
    """State weights with GT class multipliers, fixed before splitting the update."""
    weights = torch.full((2 * state_count,), 1 / state_count, dtype=torch.float64)
    if gold_weights is not None:
        weights[state_count:] *= gold_weights.to(weights)
    return {
        "optd_kind": torch.tensor([0] * state_count + [1] * state_count, dtype=torch.long),
        "optd_weight": weights,
    }


def align_response_logits(logits, response_start, response_mask):
    """Gather logits at P-1; offsets include left padding, responses pad right."""
    rows, length, _ = logits.shape
    offsets = torch.arange(response_mask.shape[1], device=logits.device)[None, :]
    positions = response_start[:, None] + offsets
    row = torch.arange(rows, device=logits.device)[:, None]
    return logits[row, (positions - 1).clamp_max(length - 1)]


def worker_loss(logits: torch.Tensor, batch: Mapping, config: LossConfig):
    """Sum globally weighted sampled (kind=0) and GT (kind=1) row losses."""
    targets, mask = batch["optd_target_ids"], batch["optd_response_mask"]
    aligned = align_response_logits(logits, batch["optd_response_start"], mask)
    kinds, weights = batch["optd_kind"], batch["optd_weight"]
    # Differentiable zero independent of padded logits.
    kd = aligned.reshape(-1)[:0].sum()
    gt = kd
    kl_sum, token_count = 0.0, 0
    for i in range(len(logits)):
        size = int(mask[i].sum())
        row, active = aligned[i:i+1, :size], mask[i:i+1, :size]
        if int(kinds[i]) == 0:
            base = batch["optd_teacher_base_logits"][i:i+1, :size]
            privileged = batch["optd_teacher_intent_logits"][i:i+1, :size]
            with torch.no_grad():
                guided, _, kl = trust_region_teacher(privileged, base, kl_budget=config.trust_region_kl)
            kd = kd + weights[i] * generalized_jsd_loss(row, guided, active, beta=config.beta)
            kl_sum += float(kl[active].sum())
            token_count += int(active.sum())
        else:
            gt = gt + weights[i] * supervised_action_loss(row, targets[i:i+1, :size], active)
    total = kd + config.gold_coefficient * gt
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite OPTD loss")
    # Partial sums over this microbatch.
    metrics = {"optd/loss_sum": float(total.detach()), "optd/distillation_sum": float(kd.detach()),
               "optd/grounding_sum": float(gt.detach()), "optd/guidance_kl_sum": kl_sum,
               "optd/guidance_tokens": token_count}
    return total * config.backward_scale, metrics


def roll_loss(config: LossConfig):
    """ROLL TrainStrategy loss_func(data: DataProto, output_tensor)."""
    def loss_func(data, output_tensor):
        return worker_loss(output_tensor, data.batch, config)
    return loss_func


def verl_loss(config: LossConfig):
    """verl TrainingWorker.set_loss_fn hook; forward_step must return logits."""
    def loss_func(model_output, data, dp_group=None):
        loss, metrics = worker_loss(model_output["logits"], data, config)
        from verl.utils.metric import AggregationType, Metric
        return loss, Metric.from_dict(metrics, aggregation=AggregationType.SUM)
    return loss_func
