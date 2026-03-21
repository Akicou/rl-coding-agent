"""Model loading helpers for policy and reference models."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _trainable_stats(model: torch.nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def load_models(cfg: Any) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
    """Load tokenizer, policy model, and frozen reference model."""

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )

    policy = prepare_model_for_kbit_training(policy)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    policy = get_peft_model(policy, lora_config)

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    trainable, total = _trainable_stats(policy)
    pct = 100.0 * trainable / max(total, 1)
    print(f"[models] trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return tokenizer, policy, ref_model
