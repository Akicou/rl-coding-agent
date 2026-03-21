"""CUDA smoke test for local or remote causal language models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.config import RLConfig
from rl_agent.models import load_models


def main() -> None:
    """Load a model through the repo path and run one short generation."""

    parser = argparse.ArgumentParser(
        description="Run a CUDA smoke test using the repository model loader.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local snapshot path.",
    )
    parser.add_argument(
        "--prompt",
        default="def add(a, b):",
        help="Prompt used for the short generation check.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading for debugging.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the current environment.")

    model_path = str(Path(args.model)) if Path(args.model).exists() else args.model
    cfg = RLConfig(model_name=model_path, load_in_4bit=not args.no_4bit)
    tokenizer, policy, _ = load_models(cfg)
    device = next(policy.parameters()).device
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = policy.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    payload = {
        "cuda": torch.cuda.is_available(),
        "device": str(device),
        "model": model_path,
        "decoded": tokenizer.decode(generated[0], skip_special_tokens=True),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
