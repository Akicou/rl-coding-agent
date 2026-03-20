"""Loss computation for Group Relative Policy Optimization."""

from __future__ import annotations

import torch
from torch import Tensor


def grpo_loss(
    p_lp: Tensor, r_lp: Tensor, rewards: Tensor, cfg: object
) -> tuple[Tensor, dict[str, float]]:
    """Compute the clipped GRPO policy loss and KL penalty."""

    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    old_lp = p_lp.detach()
    ratio = torch.exp(p_lp - old_lp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
    pg_loss = -torch.mean(torch.minimum(unclipped, clipped))

    log_ratio = r_lp - p_lp
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    loss = pg_loss + cfg.kl_coef * kl.mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl": float(kl.mean().detach().cpu()),
        "reward_mean": float(rewards.mean().detach().cpu()),
        "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
    }
