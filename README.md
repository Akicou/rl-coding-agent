# RL Coding Agent

**Self-improving LLMs through Group Relative Policy Optimization (GRPO) and multi-language execution sandboxes.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements an autonomous training loop that evolves `Qwen2.5-Coder` (or any compatible model) into a superior coding agent. It leverages **GRPO**—the same reinforcement learning logic behind DeepSeek-R1—to optimize model performance using real-world execution feedback across six programming languages, requiring **zero human labels**.

---

## 🚀 Highlights

- **GRPO Training:** Efficient reinforcement learning without the overhead of a separate critic or value network.
- **Autonomous Problem Generation:** Uses any OpenAI-compatible API (Ollama, vLLM, OpenRouter) to generate unique coding challenges on the fly.
- **Multi-Language Sandbox:** Integrated execution for **Python, Go, Node.js, C#, C++, and Rust** with strict timeouts and dependency management.
- **Hardware Efficient:** Optimized for consumer hardware; fits 7B models in 4-bit (NF4) on a single 24GB GPU (RTX 3090/4090).
- **Windows & Linux Ready:** Cross-platform support for all language runtimes and execution environments.

## 🛠️ The Feedback Loop

1. **Synthesize:** A "Teacher" LLM generates a structured coding problem with unit tests.
2. **Rollout:** The "Student" model generates multiple completion candidates (the "Group").
3. **Execute:** Every candidate is compiled and run against the unit tests in a secure sandbox.
4. **Reward:** Candidates are scored based on formatting, compilation success, and pass rate.
5. **Optimize:** GRPO computes advantages within the group to update the student policy.

## 🚦 Getting Started

### 1. Prerequisites
Ensure you have the necessary language runtimes (Go, Node, .NET, G++, Rust) installed.
```bash
# On Linux
bash system_deps.sh

# On Windows
# Ensure 'go', 'node', 'dotnet', 'g++', and 'cargo' are in your PATH.
```

### 2. Setup
```bash
git clone https://github.com/Akicou/rl-coding-agent
cd rl-coding-agent
pip install -r requirements.txt
cp .env.example .env  # Configure your OAI_BASE_URL and API keys
```

### 3. Training
```bash
# Run a smoke test to verify model loading and generation
python scripts/smoke_test.py

# Start the infinite training loop
python scripts/train.py
```

## ⚙️ Configuration (`RLConfig`)

Key parameters for tuning the training loop:

| Category | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **Model** | `model_name` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Target policy & reference model |
| **Generation** | `group_size` | `4` | Completions per rollout group |
| | `max_new_tokens` | `256` | Max generation length |
| **RL** | `kl_coef` | `0.04` | Regularization vs. reference policy |
| | `clip_eps` | `0.2` | PPO-style clipping epsilon |
| **Reward** | `w_pass` | `1.0` | Weight for test pass rate |
| | `w_compile` | `0.3` | Weight for compilation success |
| **Loop** | `batch_size` | `2` | Problems per micro-batch |
| | `grad_accum` | `4` | Gradients accumulated per step |

## 🧪 Language Runtimes

| Language | Engine | Sandbox Detail |
| :--- | :--- | :--- |
| **Python** | 3.11+ | Auto-installs missing packages via pip |
| **Go** | 1.22+ | Isolated `go.mod` environment |
| **Node.js** | 20+ | Dynamic `package.json` with npm support |
| **C#** | .NET 8 | Ephemeral `.csproj` with NuGet resolution |
| **C++** | G++ 17 | Direct compilation and execution |
| **Rust** | Stable | Full Cargo project isolation |

## 🤝 Contributing

We welcome technical contributions. To add a new language runtime:
1.  Subclass `LanguageExecutor` in `rl_agent/languages/`.
2.  Implement `extract_deps()` and `execute()`.
3.  Register it in the `LANGUAGE_REGISTRY`.

## 📄 License

MIT © 2026 Akicou
