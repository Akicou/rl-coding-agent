"""Tests for problem generation strategies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rl_agent.generator import CodingProblem, ProblemGenerator


def _fixture_problem() -> dict:
    return {
        "title": "Sum Two Numbers",
        "description": "Read two integers and print their sum.",
        "input_format": "Two integers separated by space.",
        "output_format": "One integer followed by a newline.",
        "constraints": ["-1000 <= a, b <= 1000"],
        "examples": [{"input": "1 2\n", "output": "3\n"}],
        "test_cases": [
            {"input": "1 2\n", "output": "3\n"},
            {"input": "-5 10\n", "output": "5\n"},
        ],
    }


@patch("rl_agent.generator.OpenAI")
def test_generate_structured_parse_happy_path(mock_openai: MagicMock) -> None:
    parsed = CodingProblem.model_validate(_fixture_problem())
    client = MagicMock()
    client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=parsed))]
    )
    mock_openai.return_value = client
    generator = ProblemGenerator("http://localhost:11434/v1", "ollama", "qwen2.5:14b")

    result = generator.generate("medium", 5, topic="strings")

    assert result is not None
    assert result.title == parsed.title


@patch("rl_agent.generator.OpenAI")
def test_generate_json_fallback_when_parse_fails(mock_openai: MagicMock) -> None:
    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = RuntimeError("parse failed")
    payload = _fixture_problem()
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=f"```json\n{CodingProblem.model_validate(payload).model_dump_json()}\n```"
                )
            )
        ]
    )
    mock_openai.return_value = client
    generator = ProblemGenerator("http://localhost:11434/v1", "ollama", "qwen2.5:14b")

    result = generator.generate("medium", 5, topic="strings")

    assert result is not None
    assert result.title == payload["title"]


@patch("rl_agent.generator.OpenAI")
def test_generate_returns_none_when_both_strategies_fail(
    mock_openai: MagicMock,
) -> None:
    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = RuntimeError("parse failed")
    client.chat.completions.create.side_effect = RuntimeError("json failed")
    mock_openai.return_value = client
    generator = ProblemGenerator("http://localhost:11434/v1", "ollama", "qwen2.5:14b")

    result = generator.generate("medium", 5, topic="strings")

    assert result is None


def test_coding_problem_validates_from_json_fixture() -> None:
    fixture = CodingProblem.model_validate(_fixture_problem()).model_dump_json()
    problem = CodingProblem.model_validate_json(fixture)

    assert problem.title == "Sum Two Numbers"
    assert len(problem.test_cases) == 2
