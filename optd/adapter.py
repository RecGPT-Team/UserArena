"""Single-process OPTD updates with model-specific input preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .objective import generalized_jsd_loss, supervised_action_loss, trust_region_teacher


@dataclass(frozen=True)
class PreparedInput:
    inputs: Mapping[str, Any]
    response_start: int


@dataclass(frozen=True)
class TrainingState:
    session_id: str
    context: Any
    privileged_context: Any
    gold_token_ids: torch.Tensor


@dataclass(frozen=True)
class UpdateResult:
    loss: float
    distillation: float
    supervised: float
    guidance_kl_mean: float
    sampled_token_ids: tuple[tuple[int, ...], ...]
    states: int


def prepare_tokens(context: torch.Tensor, response: torch.Tensor | None) -> PreparedInput:
    """Text-token preparation; multimodal callers supply their own callback."""
    ids = context if response is None else torch.cat((context, response.to(context.device)))
    return PreparedInput({"input_ids": ids.unsqueeze(0),
                          "attention_mask": torch.ones_like(ids).unsqueeze(0)}, len(context))


class TorchBackend:
    """Causal model backend. prepare must append exact IDs and rebuild masks/positions."""

    def __init__(self, model: torch.nn.Module, *,
                 prepare: Callable = prepare_tokens, generation_kwargs=None):
        self.model = model
        self.prepare = prepare
        self.generation_kwargs = dict(generation_kwargs or {
            "max_new_tokens": 512, "do_sample": True, "temperature": 0.8})

    def _inputs(self, context, response):
        prepared = self.prepare(context, response)
        inputs = dict(prepared.inputs)
        start = prepared.response_start
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        return inputs, start

    def generate(self, context):
        inputs, start = self._inputs(context, None)
        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(**inputs, **self.generation_kwargs)
        finally:
            self.model.train(training)
        response = output[0, start:].detach()
        return response

    def score(self, context, response):
        inputs, start = self._inputs(context, response)
        logits = self.model(**inputs).logits
        return logits[:, start - 1:start - 1 + len(response), :]


class OptdAdapter:
    """One local optimizer update, averaging both losses over the same paired states."""

    def __init__(self, student: TorchBackend, teacher: TorchBackend, *,
                 gold_coefficient=0.1, beta=0.5, trust_region_kl=0.02):
        self.student, self.teacher = student, teacher
        self.gold_coefficient = gold_coefficient
        self.beta, self.trust_region_kl = beta, trust_region_kl
        teacher.model.requires_grad_(False)
        teacher.model.eval()

    def step(self, states: Sequence[TrainingState], optimizer: torch.optim.Optimizer, *, gold_weights=None) -> UpdateResult:
        self.student.model.train()
        self.teacher.model.eval()
        optimizer.zero_grad(set_to_none=True)
        kd_total = gt_total = kl_sum = 0.0
        tokens = 0
        sampled = []
        try:
            for i, state in enumerate(states):
                response = self.student.generate(state.context)
                sampled.append(tuple(response.cpu().tolist()))
                with torch.no_grad():
                    base = self.teacher.score(state.context, response)
                    privileged = self.teacher.score(state.privileged_context, response)
                    guided, _, kl = trust_region_teacher(privileged, base, kl_budget=self.trust_region_kl)
                student = self.student.score(state.context, response)
                mask = torch.ones(student.shape[:2], dtype=torch.bool, device=student.device)
                kd = generalized_jsd_loss(student, guided.to(student.device), mask, beta=self.beta)
                kd = kd / len(states)
                kd.backward()
                kd_total += float(kd.detach())
                kl_sum += float(kl.sum())
                tokens += kl.numel()
                del student, guided, base, privileged, kd
                gold = self.student.score(state.context, state.gold_token_ids)
                gold_ids = state.gold_token_ids.to(gold.device).unsqueeze(0)
                gold_mask = torch.ones_like(gold_ids, dtype=torch.bool)
                gt = supervised_action_loss(gold, gold_ids, gold_mask) / len(states)
                if gold_weights is not None:
                    gt = gt * gold_weights[i].to(gt)
                (self.gold_coefficient * gt).backward()
                gt_total += float(gt.detach())
                del gold, gt
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.student.model.parameters()):
                raise FloatingPointError("Nonfinite student gradients; optimizer step was not applied")
            optimizer.step()
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        return UpdateResult(kd_total + self.gold_coefficient * gt_total, kd_total, gt_total,
                            kl_sum / tokens, tuple(sampled), len(states))
