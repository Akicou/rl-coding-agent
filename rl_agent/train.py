"""Infinite GRPO training loop for the coding agent."""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from rl_agent.config import RLConfig
from rl_agent.generator import ProblemGenerator
from rl_agent.grpo import grpo_loss
from rl_agent.languages import LANGUAGE_REGISTRY
from rl_agent.models import load_models
from rl_agent.reward import compute_reward
from rl_agent.rollout import rollout, score_rollout


def _build_prompt(problem: Any, lang_key: str) -> str:
    """Create the instruction prompt for one language-specific rollout."""

    profile = LANGUAGE_REGISTRY[lang_key]
    constraints = "\n".join(f"- {item}" for item in problem.constraints)
    examples = "\n\n".join(
        f"Input:\n{example.input}\nOutput:\n{example.output}"
        for example in problem.examples
    )
    return (
        f"You are solving a programming problem in {profile.name}.\n"
        f"Return only a fenced ```{profile.fence}``` code block.\n"
        f"{profile.io_hint}\n\n"
        f"Title: {problem.title}\n\n"
        f"Description:\n{problem.description}\n\n"
        f"Input Format:\n{problem.input_format}\n\n"
        f"Output Format:\n{problem.output_format}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Examples:\n{examples}\n"
    )


def train(cfg: RLConfig) -> None:
    """Run the infinite GRPO training loop."""

    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer, policy, ref_model = load_models(cfg)
    device = next(policy.parameters()).device
    generator = ProblemGenerator(cfg.oai_base_url, cfg.oai_api_key, cfg.oai_model)
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=cfg.lr
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    step = 0
    lang_counter: Counter[str] = Counter()
    progress = tqdm(
        total=None if cfg.max_steps == -1 else cfg.max_steps,
        desc="training",
        unit="step",
    )

    while cfg.max_steps == -1 or step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        batch_rewards: list[float] = []
        batch_kls: list[float] = []
        batch_pgs: list[float] = []

        for _ in range(cfg.grad_accum):
            for _ in range(cfg.batch_size):
                lang_key = cfg.sample_language()
                lang_counter[lang_key] += 1
                topic = random.choice(list(generator.topics))
                problem = generator.generate(
                    cfg.gen_difficulty, cfg.n_test_cases, topic=topic
                )
                if problem is None:
                    continue

                prompt = _build_prompt(problem, lang_key)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                texts, _, _ = rollout(prompt_ids, policy, ref_model, tokenizer, cfg)
                generated_batch = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                generated_ids = generated_batch.input_ids.to(device)
                generated_mask = generated_batch.attention_mask.to(device)
                sequences = torch.cat(
                    [prompt_ids.repeat(len(texts), 1), generated_ids], dim=1
                )
                attention_mask = torch.cat(
                    [torch.ones_like(prompt_ids).repeat(len(texts), 1), generated_mask],
                    dim=1,
                )
                policy_lp, ref_lp = score_rollout(
                    sequences,
                    prompt_ids.shape[1],
                    policy,
                    ref_model,
                    attention_mask=attention_mask,
                )
                rewards_list = [
                    compute_reward(text, problem, lang_key, cfg)[0] for text in texts
                ]
                rewards = torch.tensor(
                    rewards_list, device=policy_lp.device, dtype=policy_lp.dtype
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    loss, stats = grpo_loss(policy_lp, ref_lp, rewards, cfg)
                    scaled_loss = loss / float(cfg.grad_accum * cfg.batch_size)
                scaler.scale(scaled_loss).backward()
                batch_rewards.append(stats["reward_mean"])
                batch_kls.append(stats["kl"])
                batch_pgs.append(stats["pg_loss"])

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        progress.update(1)

        if step % cfg.log_every == 0:
            print(
                f"[train] step={step} reward={sum(batch_rewards) / max(len(batch_rewards), 1):.4f} "
                f"kl={sum(batch_kls) / max(len(batch_kls), 1):.4f} "
                f"pg_loss={sum(batch_pgs) / max(len(batch_pgs), 1):.4f} "
                f"langs={dict(lang_counter)}"
            )

        if step % cfg.save_every == 0:
            checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"[train] saved checkpoint to {checkpoint_dir}")

    progress.close()
