"""Sampling utilities for GRPO rollouts."""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn.functional as F
from torch import Tensor


def _sequence_logps(
    model: torch.nn.Module,
    sequences: Tensor,
    prompt_len: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute summed log-probabilities over generated tokens only."""

    if attention_mask is None:
        attention_mask = torch.ones_like(sequences)
    outputs = model(
        input_ids=sequences[:, :-1],
        attention_mask=attention_mask[:, :-1],
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    target = sequences[:, 1:].unsqueeze(-1)
    token_logps = log_probs.gather(-1, target).squeeze(-1)
    generated_mask = attention_mask[:, 1:][:, prompt_len - 1 :].to(token_logps.dtype)
    generated = token_logps[:, prompt_len - 1 :] * generated_mask
    return generated.sum(dim=-1)


def score_rollout(
    sequences: Tensor,
    prompt_len: int,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Score sampled sequences under policy and reference models."""

    policy_logps = _sequence_logps(model, sequences, prompt_len, attention_mask)
    with torch.no_grad():
        ref_logps = _sequence_logps(ref_model, sequences, prompt_len, attention_mask)
    return policy_logps, ref_logps


@torch.no_grad()
def rollout(
    prompt_ids: Tensor,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: Any,
    cfg: Any,
) -> tuple[List[str], Tensor, Tensor]:
    """Sample a rollout group and compute policy/reference log-prob sums."""

    group = cfg.group_size
    repeated = prompt_ids.repeat(group, 1)
    prompt_len = repeated.shape[1]
    outputs = model.generate(
        input_ids=repeated,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    attention_mask = outputs.ne(tokenizer.pad_token_id or tokenizer.eos_token_id).long()
    policy_logps, ref_logps = score_rollout(
        outputs,
        prompt_len,
        model,
        ref_model,
        attention_mask=attention_mask,
    )
    return texts, policy_logps, ref_logps
