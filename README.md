# autoresearch (latency edition)

![teaser](progress.png)

An experiment in giving an AI agent a small but real inference-latency optimization loop and letting it run overnight. It modifies the inference code, measures latency and correctness against a fixed reference, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a faster inference setup.

The target is **`Qwen/Qwen3.5-0.8B`**, a small pre-trained LLM. The goal is to minimize end-to-end greedy-generation latency on a fixed prompt set without changing the outputs. As with the pretraining version of this repo, the key idea is that you're not driving the Python files by hand — you're programming the `program.md` markdown file that provides context to the AI agents and sets up your autonomous research org.

## How it works

Three files matter:

- **`prepare.py`** — fixed setup and runtime utilities: downloads the target model, builds the fixed prompt set, captures reference greedy outputs from the stock model, and defines the latency + correctness harness. Not modified.
- **`optimize.py`** — the single file the agent edits. Contains model loading and the `generate` callable. Everything is fair game: attention implementation, KV cache strategy, `torch.compile`, CUDA graphs, dtype, quantization (bitsandbytes / GPTQ / AWQ / fp8), weight surgery (pruning, layer-skipping, speculative decoding). **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

The metric is **`latency_ms`** — mean of per-prompt median wall-clock times for greedy generation of 128 new tokens across a fixed 16-prompt set. Lower is better. A run is only valid if **`correctness_ok`** is `true`, meaning the generated token ids match the reference outputs within a per-prompt stable prefix (computed at setup time to tolerate floating-point non-determinism).

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download model and build reference outputs (one-time)
uv run prepare.py

# 4. Manually run a single inference experiment
uv run optimize.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — model download, reference capture, latency + correctness harness (do not modify)
optimize.py     — model loading and generate callable (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `optimize.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed workload.** A fixed 16-prompt set with a fixed `max_new_tokens=128`, greedy decoding, batch size 1. This makes experiments directly comparable regardless of what the agent changes (quantization, attention impl, KV cache, etc). The downside is that numbers are not comparable across hardware.
- **Correctness floor.** The measurement harness records reference outputs from the stock model at setup time and compares every experiment's outputs to that reference. A run only counts if `correctness_ok` is `true` — a latency-only metric is trivially "won" by returning garbage.
- **Stable-prefix tolerance.** The reference is captured twice and only the matching prefix is required to match. This handles fp non-determinism between attention backends without making the correctness check meaningless.
- **Self-contained.** Minimal dependencies at baseline — `transformers`, `accelerate`, `huggingface_hub`, plus standard scientific Python. The agent is free to add more (`bitsandbytes`, `auto-gptq`, `autoawq`, `vllm`, etc.) as experiments require.

## Platform support

This code currently requires a single NVIDIA GPU. Adapting to CPU, MPS, or other platforms is possible but would bloat the code; feel free to fork.

## License

MIT
