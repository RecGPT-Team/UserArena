"""OPTD trust-region teacher, generalized JSD and paired grounding losses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OptdLoss:
    total: torch.Tensor
    distillation: torch.Tensor
    supervised: torch.Tensor
    guidance_kl_mean: torch.Tensor
    guidance_scale_mean: torch.Tensor


def action_class_weights(actions, frequencies, power=0.5, cap=2.0):
    """Inverse-frequency GT weights with update mean 1 and a final cap."""
    raw = torch.tensor([frequencies[action] ** (-power) for action in actions], dtype=torch.float64)
    lower, upper = 0.0, 1.0 / raw.min().item()
    for _ in range(64):
        middle = (lower + upper) / 2
        if (middle * raw).clamp_max(cap).mean() < 1:
            lower = middle
        else:
            upper = middle
    return (upper * raw).clamp_max(cap)


def _action_mean(token_loss: torch.Tensor, mask: torch.Tensor, weights=None) -> torch.Tensor:
    counts = mask.sum(-1)
    per_row = token_loss.masked_fill(~mask, 0).sum(-1) / counts.clamp_min(1)
    if weights is not None:
        per_row = per_row * weights.to(per_row)
    return per_row.sum() / (counts > 0).sum().clamp_min(1)


def trust_region_teacher(
    privileged_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    kl_budget: float = 0.02,
    maximum_scale: float = 1.0,
    search_steps: int = 8,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constrain the privileged teacher to a KL ball around the base teacher.

    For every token, choose the largest ``scale`` in ``[0, maximum_scale]`` for
    ``softmax(z_base + scale * (z_privileged - z_base))`` whose
    ``KL(guided || base)`` does not exceed ``kl_budget``.

    Returns normalized guided log probabilities, selected scales, and measured
    KL values. Inputs may be ``[rows, tokens, vocabulary]`` or
    ``[tokens, vocabulary]``.
    """

    dtype = torch.float64 if torch.float64 in (privileged_logits.dtype, base_logits.dtype) else torch.float32
    privileged = F.log_softmax(privileged_logits.detach().to(dtype) / temperature, dim=-1)
    base = F.log_softmax(base_logits.detach().to(dtype) / temperature, dim=-1)
    residual = privileged - base
    scale_shape = (*privileged.shape[:-1], 1)

    def interpolate(scale: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(base + scale * residual, dim=-1)

    upper = privileged.new_full(scale_shape, maximum_scale)
    full = privileged if maximum_scale == 1 else interpolate(upper)
    full_kl = (full.exp() * (full - base)).sum(-1).clamp_min(0)
    constrained = full_kl > kl_budget
    lower = torch.zeros_like(upper)
    for _ in range(search_steps):
        middle = (lower + upper) * 0.5
        candidate = interpolate(middle)
        feasible = ((candidate.exp() * (candidate - base)).sum(-1) <= kl_budget).unsqueeze(-1)
        lower = torch.where(feasible, middle, lower)
        upper = torch.where(feasible, upper, middle)
    scale = torch.where(constrained, lower.squeeze(-1), torch.full_like(full_kl, maximum_scale))
    guided = torch.where(constrained.unsqueeze(-1), interpolate(scale.unsqueeze(-1)), full)
    kl = (guided.exp() * (guided - base)).sum(-1).clamp_min(0)
    return guided, scale.detach(), kl.detach()


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Action-mean generalized JSD on response tokens.

    Each state first averages over its response tokens. For ``0 < beta < 1`` this is
    the paper's generalized JSD; ``beta=0.5`` is standard JSD. The endpoints
    select KL(teacher || student) at zero and KL(student || teacher) at one.
    """
    mask = response_mask.bool()
    dtype = torch.float64 if torch.float64 in (student_logits.dtype, teacher_log_probs.dtype) else torch.float32
    student = F.log_softmax(student_logits.to(dtype), dim=-1)
    teacher = teacher_log_probs.detach().to(dtype)
    if beta == 0:
        divergence = F.kl_div(student, teacher, reduction="none", log_target=True)
    elif beta == 1:
        divergence = F.kl_div(teacher, student, reduction="none", log_target=True)
    else:
        mixture = torch.logaddexp(student + math.log1p(-beta), teacher + math.log(beta))
        divergence = beta * F.kl_div(mixture, teacher, reduction="none", log_target=True)
        divergence += (1 - beta) * F.kl_div(mixture, student, reduction="none", log_target=True)
    return _action_mean(divergence.sum(-1), mask)


def supervised_action_loss(
    student_logits: torch.Tensor,
    gold_token_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    *,
    gold_weights=None,
) -> torch.Tensor:
    """Action-mean next-token cross entropy for paired logged responses.

    Logits must already be shifted to predict the corresponding token IDs.
    Masked token IDs may be padding values such as ``-100``.
    gold_weights supplies class multipliers normalized over the effective update.
    """
    mask = gold_mask.bool()
    safe_ids = gold_token_ids.long().masked_fill(~mask, 0)
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_probs = F.log_softmax(student_logits.to(dtype), dim=-1)
    chosen = log_probs.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
    return _action_mean(-chosen, mask, gold_weights)


def optd_loss(
    student_logits: torch.Tensor,
    privileged_teacher_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gold_student_logits: torch.Tensor,
    gold_token_ids: torch.Tensor,
    gold_mask: torch.Tensor,
    gold_weights=None,
    gold_coefficient: float = 0.1,
    beta: float = 0.5,
    trust_region_kl: float = 0.02,
) -> OptdLoss:
    """Combine state-mean distillation and grounding on paired states (Eqs. 7–10)."""
    guided, scales, guidance_kl = trust_region_teacher(
        privileged_teacher_logits, base_teacher_logits, kl_budget=trust_region_kl
    )
    kd = generalized_jsd_loss(student_logits, guided, response_mask, beta=beta)
    gt = supervised_action_loss(gold_student_logits, gold_token_ids, gold_mask, gold_weights=gold_weights)
    total = kd + gold_coefficient * gt
    active_kl = guidance_kl[response_mask.bool()]
    active_scale = scales[response_mask.bool()]
    zero = total.detach() * 0
    return OptdLoss(
        total=total,
        distillation=kd.detach(),
        supervised=gt.detach(),
        guidance_kl_mean=active_kl.mean() if active_kl.numel() else zero,
        guidance_scale_mean=active_scale.mean() if active_scale.numel() else zero,
    )
