"""Reward computation using sandbox execution and output comparison."""

from __future__ import annotations

import re
from typing import Any

from rl_agent.languages import LANGUAGE_REGISTRY


def _extract_code(output_text: str, lang_key: str) -> tuple[str, bool]:
    """Extract code from the model response using language-specific fences first."""

    fence = LANGUAGE_REGISTRY[lang_key].fence
    exact = re.search(
        rf"```{re.escape(fence)}\s*(.*?)\s*```", output_text, re.DOTALL | re.IGNORECASE
    )
    if exact:
        return exact.group(1).strip(), True
    any_fence = re.search(r"```[a-zA-Z0-9_+-]*\s*(.*?)\s*```", output_text, re.DOTALL)
    if any_fence:
        return any_fence.group(1).strip(), False
    return output_text.strip(), False


def compute_reward(
    output_text: str, problem: Any, lang_key: str, cfg: Any
) -> tuple[float, dict[str, Any]]:
    """Compute format, compile, and pass-rate rewards for one candidate."""

    code, exact_fence = _extract_code(output_text, lang_key)
    executor = LANGUAGE_REGISTRY[lang_key].executor
    format_reward = 1.0 if exact_fence else 0.0

    first_case = problem.test_cases[0]
    compile_result = executor.execute(
        code, stdin=first_case.input, timeout=cfg.exec_timeout
    )
    compile_reward = 1.0 if compile_result.success else 0.0

    passed = 0
    case_results: list[dict[str, Any]] = []
    if compile_result.success:
        for case in problem.test_cases:
            result = executor.execute(code, stdin=case.input, timeout=cfg.exec_timeout)
            ok = result.success and result.stdout.strip() == case.output.strip()
            passed += int(ok)
            case_results.append(
                {"stdout": result.stdout, "stderr": result.stderr, "ok": ok}
            )
    pass_rate = passed / max(len(problem.test_cases), 1)

    total = (
        cfg.w_format * format_reward
        + cfg.w_compile * compile_reward
        + cfg.w_pass * pass_rate
    )
    return total, {
        "lang": lang_key,
        "format_reward": format_reward,
        "compile_reward": compile_reward,
        "pass_rate": pass_rate,
        "passed": passed,
        "n_cases": len(problem.test_cases),
        "compile_stdout": compile_result.stdout,
        "compile_stderr": compile_result.stderr,
        "cases": case_results,
    }
