"""Infinite GRPO training loop for the coding agent."""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from rl_agent.config import RLConfig
from rl_agent.generator import AsyncProblemGenerator
from rl_agent.grpo import grpo_loss
from rl_agent.languages import LANGUAGE_REGISTRY
from rl_agent.models import load_models
from rl_agent.reward import compute_reward
from rl_agent.rollout import rollout, score_rollout


def _compute_reward_task(args):
    """Helper for process pool mapping."""
    text, problem, lang_key, cfg = args
    return compute_reward(text, problem, lang_key, cfg)[0]


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

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer, policy, ref_model = load_models(cfg)
    device = next(policy.parameters()).device
    generator = AsyncProblemGenerator(
        cfg.oai_base_url,
        cfg.oai_api_key,
        cfg.oai_model,
        cfg.gen_difficulty,
        cfg.n_test_cases,
        queue_size=cfg.prefetch_queue_size,
    )
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=cfg.lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    step = 0
    lang_counter: Counter[str] = Counter()
    micro_batches_per_step = cfg.grad_accum * cfg.batch_size
    progress = tqdm(
        total=None if cfg.max_steps == -1 else cfg.max_steps,
        desc="training",
        unit="step",
        dynamic_ncols=True,
    )

    try:
        with logging_redirect_tqdm(), ProcessPoolExecutor(
            max_workers=cfg.reward_workers
        ) as executor:
            while cfg.max_steps == -1 or step < cfg.max_steps:
                optimizer.zero_grad(set_to_none=True)
                batch_rewards: list[float] = []
                batch_kls: list[float] = []
                batch_pgs: list[float] = []
                completed_micro_batches = 0

                for _ in range(cfg.grad_accum):
                    for _ in range(cfg.batch_size):
                        lang_key = cfg.sample_language()
                        lang_counter[lang_key] += 1
                        progress.set_postfix_str(
                            f"step={step + 1} batch={completed_micro_batches}/{micro_batches_per_step} stage=generate lang={lang_key}"
                        )
                        problem = generator.get()

                        prompt = _build_prompt(problem, lang_key)
                        prompt_batch = tokenizer(prompt, return_tensors="pt")
                        prompt_ids = prompt_batch.input_ids.to(device)
                        prompt_attention_mask = prompt_batch.attention_mask.to(device)
                        progress.set_postfix_str(
                            f"step={step + 1} batch={completed_micro_batches}/{micro_batches_per_step} stage=rollout lang={lang_key} plen={prompt_ids.shape[1]}"
                        )
                        texts, _, _ = rollout(
                            prompt_ids,
                            prompt_attention_mask,
                            policy,
                            ref_model,
                            tokenizer,
                            cfg,
                        )
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
                            [
                                torch.ones_like(prompt_ids).repeat(len(texts), 1),
                                generated_mask,
                            ],
                            dim=1,
                        )
                        progress.set_postfix_str(
                            f"step={step + 1} batch={completed_micro_batches}/{micro_batches_per_step} stage=score lang={lang_key}"
                        )
                        policy_lp, ref_lp = score_rollout(
                            sequences,
                            prompt_ids.shape[1],
                            policy,
                            ref_model,
                            attention_mask=attention_mask,
                        )
                        progress.set_postfix_str(
                            f"step={step + 1} batch={completed_micro_batches}/{micro_batches_per_step} stage=reward lang={lang_key}"
                        )

                        # Parallel reward computation
                        reward_args = [(text, problem, lang_key, cfg) for text in texts]
                        rewards_list = list(
                            executor.map(_compute_reward_task, reward_args)
                        )

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
                        progress.set_postfix_str(
                            f"step={step + 1} batch={completed_micro_batches}/{micro_batches_per_step} stage=backward lang={lang_key}"
                        )
                        scaler.scale(scaled_loss).backward()
                        batch_rewards.append(stats["reward_mean"])
                        batch_kls.append(stats["kl"])
                        batch_pgs.append(stats["pg_loss"])
                        completed_micro_batches += 1
                        progress.update(1 / micro_batches_per_step)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1

                if step % cfg.log_every == 0:
                    progress.write(
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
                    progress.write(f"[train] saved checkpoint to {checkpoint_dir}")
    finally:
        progress.close()

