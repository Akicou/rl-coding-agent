"""Utility for sampling coding problems from the generator endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl_agent.generator import ProblemGenerator


def main() -> None:
    """Generate problems and print them as formatted JSON."""

    parser = argparse.ArgumentParser(
        description="Generate coding problems from an OpenAI-compatible endpoint.",
        epilog="Usage: python scripts/generate_problems.py --n 5 --difficulty hard --save",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of problems to generate."
    )
    parser.add_argument(
        "--difficulty", default="medium", help="Difficulty level to request."
    )
    parser.add_argument("--topic", default=None, help="Optional topic override.")
    parser.add_argument(
        "--lang", default=None, help="Optional language tag for your own tracking."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save output to problems.jsonl."
    )
    args = parser.parse_args()

    generator = ProblemGenerator(
        base_url=os.getenv("OAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OAI_API_KEY", "ollama"),
        model=os.getenv("OAI_MODEL", "qwen2.5:14b"),
    )

    sink = Path("problems.jsonl") if args.save else None
    if sink is not None:
        sink.write_text("", encoding="utf-8")

    for idx in range(args.n):
        problem = generator.generate(args.difficulty, 5, topic=args.topic)
        if problem is None:
            print(json.dumps({"index": idx, "error": "generation_failed"}, indent=2))
            continue
        payload = problem.model_dump()
        if args.lang:
            payload["lang"] = args.lang
        print(json.dumps(payload, indent=2))
        if sink is not None:
            with sink.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
