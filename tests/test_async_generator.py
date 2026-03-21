"""Tests for AsyncProblemGenerator."""

import asyncio
import time
from unittest.mock import MagicMock, patch
from rl_agent.generator import AsyncProblemGenerator, CodingProblem

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

@patch("rl_agent.generator.AsyncOpenAI")
def test_async_generator_prefetch(mock_async_openai: MagicMock) -> None:
    # Setup mock
    parsed = CodingProblem.model_validate(_fixture_problem())
    
    mock_client = MagicMock()
    # Mock for structured parse
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    
    # Async mock for parse
    async def mock_parse(*args, **kwargs):
        return mock_response
    
    mock_client.beta.chat.completions.parse = mock_parse
    mock_async_openai.return_value = mock_client
    
    # Initialize generator
    generator = AsyncProblemGenerator(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:14b",
        difficulty="medium",
        n_tests=5,
        queue_size=2
    )
    
    # Wait a bit for prefetch
    timeout = 5
    start = time.time()
    while generator.queue.empty() and time.time() - start < timeout:
        time.sleep(0.1)
    
    assert not generator.queue.empty(), "Queue should have been prefetched"
    
    problem = generator.get()
    assert problem.title == "Sum Two Numbers"
    
    generator.stop()

@patch("rl_agent.generator.AsyncOpenAI")
def test_async_generator_fallback(mock_async_openai: MagicMock) -> None:
    # Setup mock to fail parse and succeed with json
    payload = _fixture_problem()
    mock_client = MagicMock()
    
    async def mock_parse_fail(*args, **kwargs):
        raise RuntimeError("parse failed")
        
    async def mock_json_success(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=f"```json\n{CodingProblem.model_validate(payload).model_dump_json()}\n```"
                )
            )
        ]
        return mock_resp
        
    mock_client.beta.chat.completions.parse = mock_parse_fail
    mock_client.chat.completions.create = mock_json_success
    mock_async_openai.return_value = mock_client
    
    generator = AsyncProblemGenerator(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:14b",
        difficulty="medium",
        n_tests=5,
        queue_size=1
    )
    
    # Wait for prefetch
    timeout = 5
    start = time.time()
    while generator.queue.empty() and time.time() - start < timeout:
        time.sleep(0.1)
        
    assert not generator.queue.empty()
    problem = generator.get()
    assert problem.title == payload["title"]
    
    generator.stop()
