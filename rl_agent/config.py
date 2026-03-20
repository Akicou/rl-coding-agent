"""Configuration objects for the RL coding agent."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field


@dataclass
class RLConfig:
    """Top-level runtime configuration for training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    load_in_4bit: bool = True

    # Generator
    oai_base_url: str = "http://localhost:11434/v1"
    oai_api_key: str = "ollama"
    oai_model: str = "qwen2.5:14b"
    gen_difficulty: str = "medium"
    n_test_cases: int = 5

    # Language sampling
    active_languages: list[str] = field(
        default_factory=lambda: ["python", "golang", "nodejs", "csharp", "cpp", "rust"]
    )
    language_weights: dict[str, float] | None = None

    # GRPO
    group_size: int = 8
    lr: float = 5e-7
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95

    # Loop
    batch_size: int = 2
    grad_accum: int = 4
    max_steps: int = -1
    save_every: int = 200
    log_every: int = 10
    output_dir: str = "./rl_coding_agent"

    # Reward
    w_pass: float = 1.0
    w_compile: float = 0.3
    w_format: float = 0.1
    exec_timeout: int = 10

    def sample_language(self) -> str:
        """Sample a language key using optional weights."""

        weights = None
        if self.language_weights is not None:
            weights = [
                self.language_weights.get(key, 0.0) for key in self.active_languages
            ]
        return random.choices(self.active_languages, weights=weights, k=1)[0]

    @classmethod
    def from_env(cls) -> "RLConfig":
        """Build a config instance from environment variables."""

        defaults = cls()
        return cls(
            model_name=os.getenv("HF_MODEL", defaults.model_name),
            oai_base_url=os.getenv("OAI_BASE_URL", defaults.oai_base_url),
            oai_api_key=os.getenv("OAI_API_KEY", defaults.oai_api_key),
            oai_model=os.getenv("OAI_MODEL", defaults.oai_model),
        )
