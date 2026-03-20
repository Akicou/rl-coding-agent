# 🧠 rl-coding-agent

*Infinite RL loop that trains an open-source LLM into a SOTA coding agent — self-generating problems, multi-language sandboxed execution, zero human labels.*

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

Train `Qwen/Qwen2.5-Coder-7B-Instruct` inside an infinite reinforcement learning loop powered by GRPO, AI-generated coding problems, and real code execution across six runtimes.

## Features

- 🚀 GRPO training loop with no critic or value network
- 🤖 Self-generated problems via any OpenAI-compatible endpoint: Ollama, vLLM, LM Studio, or OpenAI
- 🧱 Pydantic structured output first, JSON-mode fallback second
- 🌍 Six execution targets: Python 🐍, Go 🐹, Node.js 🟩, C# 🔵, C++ ⚙️, Rust 🦀
- ⏱️ Per-language sandboxed subprocess execution with strict timeouts
- 📦 Automatic dependency tracking and installation per language runtime
- 🧠 QLoRA 4-bit NF4 training that fits on a single 24 GB GPU
- 💾 Checkpoint saving and resume-friendly output layout
- 🛠️ Easily extensible language layer: add a new runtime in two steps

## How It Works

```text
Generator LLM
    │  generates structured CodingProblem (title, description, test cases)
    ▼
Policy Model (Qwen2.5-Coder)
    │  samples G=8 completions per problem per language
    ▼
Sandbox Executor (per language)
    │  runs code, compares stdout to expected
    ▼
Reward Signal
    │  format + compile + pass-rate
    ▼
GRPO Loss
    │  normalise rewards → advantages → policy gradient + KL penalty
    ▼
AdamW Update  ──►  repeat forever
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/rl-coding-agent
cd rl-coding-agent
bash system_deps.sh          # install language runtimes
pip install -r requirements.txt
cp .env.example .env         # fill in your endpoint
python scripts/train.py
```

## Configuration

`RLConfig` controls model loading, generation, rewards, and loop behavior.

| Field | Default | Description |
|------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace policy and reference model ID |
| `load_in_4bit` | `True` | Enable QLoRA-friendly 4-bit loading |
| `oai_base_url` | `http://localhost:11434/v1` | OpenAI-compatible generator endpoint |
| `oai_api_key` | `ollama` | API key or placeholder token |
| `oai_model` | `qwen2.5:14b` | Generator model name at the endpoint |
| `gen_difficulty` | `medium` | Problem difficulty: easy, medium, hard, competitive |
| `n_test_cases` | `5` | Number of test cases per generated problem |
| `active_languages` | `python,golang,nodejs,csharp,cpp,rust` | Runtime pool sampled during training |
| `language_weights` | `None` | Optional weighted sampling by language key |
| `group_size` | `8` | Number of sampled completions per rollout group |
| `lr` | `5e-7` | AdamW learning rate |
| `kl_coef` | `0.04` | KL regularization strength against reference policy |
| `clip_eps` | `0.2` | PPO-style clipping epsilon |
| `max_new_tokens` | `1024` | Maximum generated response length |
| `temperature` | `0.8` | Sampling temperature |
| `top_p` | `0.95` | Nucleus sampling threshold |
| `batch_size` | `2` | Problems processed per gradient accumulation slice |
| `grad_accum` | `4` | Number of accumulation slices per optimizer step |
| `max_steps` | `-1` | `-1` means run forever |
| `save_every` | `200` | Save checkpoint frequency in optimizer steps |
| `log_every` | `10` | Logging cadence |
| `output_dir` | `./rl_coding_agent` | Checkpoint and artifact directory |
| `w_pass` | `1.0` | Pass-rate reward weight |
| `w_compile` | `0.3` | Compile/execution reward weight |
| `w_format` | `0.1` | Code-format reward weight |
| `exec_timeout` | `10` | Per-test subprocess timeout in seconds |

## Language Support

| Key | Runtime | Dep tracking | Notes |
|-----|---------|-------------|-------|
| `python` | Python 3.11 | import -> pip | Auto-installs missing packages |
| `golang` | Go 1.22 | import paths -> go get | Fresh `go.mod` per run |
| `nodejs` | Node.js 20 | require/import -> npm | Fresh `package.json` per run |
| `csharp` | .NET 8 | using namespace -> NuGet | Fresh `.csproj` per run |
| `cpp` | GCC g++ C++17 | None (stdlib) | Compile + run |
| `rust` | Rust stable | use/extern -> Cargo.toml | Full cargo build |

## Adding a New Language

1. Subclass `LanguageExecutor` inside `rl_agent/languages/` and implement `extract_deps()` plus `execute()`.
2. Register the executor in `rl_agent/languages/__init__.py` by adding a new `LanguageProfile` entry to `LANGUAGE_REGISTRY`.

## Generator Endpoints

```python
# Ollama (default)
oai_base_url = "http://localhost:11434/v1"
oai_api_key = "ollama"

# vLLM
oai_base_url = "http://localhost:8000/v1"
oai_api_key = "none"

# LM Studio
oai_base_url = "http://localhost:1234/v1"
oai_api_key = "lm-studio"

# OpenAI
oai_base_url = "https://api.openai.com/v1"
oai_api_key = "sk-..."
```

## Hardware

- Minimum: 1x 24 GB VRAM GPU (RTX 3090/4090, RX 7900 XTX) for the 7B model with QLoRA
- AMD ROCm: swap the torch index URL for a ROCm wheel and use `bitsandbytes-rocm`
- Multi-GPU: keep `device_map="auto"` enabled, which is already the default

## License

MIT

## Contributing

PRs are welcome. For large changes, please open an issue first so the direction can be discussed before implementation.
