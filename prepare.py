"""
One-time setup and fixed runtime utilities for latency autoresearch.

Downloads the target model, builds the fixed prompt set, and captures a
reference of greedy outputs from the unmodified model. Also provides the
fixed measurement harness (`measure_latency`, `check_correctness`) that
`optimize.py` imports.

Usage:
    uv run prepare.py             # download model + build reference

Artifacts land in ~/.cache/autoresearch-latency/.
"""

import os
import sys
import time
import pickle
import argparse
import statistics

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-0.8B"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-latency")
PROMPTS_PATH = os.path.join(CACHE_DIR, "prompts.pt")
REFERENCE_PATH = os.path.join(CACHE_DIR, "reference.pt")

MAX_NEW_TOKENS = 128     # tokens generated per prompt, fixed so runs are comparable
WARMUP_RUNS = 2          # per-prompt warmup runs (discarded)
MEASURE_RUNS = 5         # per-prompt measured runs (used for median)
SEED = 0

# Fixed prompt set. Varied lengths so agent optimizations don't overfit to a
# single prefill size. These are checked into code (not downloaded) to keep
# the benchmark 100% reproducible.
PROMPTS = [
    "Hello.",
    "What is 2 + 2?",
    "Write a single-sentence summary of photosynthesis.",
    "List three prime numbers.",
    "Translate to French: The cat sat on the mat.",
    "Explain what a binary search tree is in one paragraph.",
    "Give me a one-line Python function that returns the factorial of n.",
    "Describe the plot of Hamlet in two sentences.",
    "What are the differences between TCP and UDP? Keep it brief.",
    "Compose a haiku about autumn leaves.",
    "Why is the sky blue? Answer in three sentences, aimed at a ten-year-old.",
    "Name five programming languages and one strength of each.",
    "In a concise paragraph, explain the concept of entropy in information theory.",
    "A farmer has 17 sheep, all but 9 die. How many are left? Show your reasoning, then give the final answer.",
    "Write a short professional email declining a meeting invitation scheduled for next Tuesday.",
    "Compare and contrast supervised and unsupervised learning. Respond in about 150 words, with clear structure.",
]
N_PROMPTS = len(PROMPTS)

# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

def download_model():
    """Idempotent: pulls Qwen3.5-0.8B weights + tokenizer into the HF cache."""
    from huggingface_hub import snapshot_download
    print(f"Model: ensuring {MODEL_ID} is downloaded...")
    snapshot_download(repo_id=MODEL_ID)
    print(f"Model: {MODEL_ID} ready.")


def build_prompt_set():
    """Tokenize the fixed prompt list and persist. Re-run is a no-op."""
    if os.path.exists(PROMPTS_PATH):
        print(f"Prompts: already built at {PROMPTS_PATH}")
        return
    from transformers import AutoTokenizer
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenized = []
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt").input_ids[0].tolist()
        tokenized.append(ids)
    with open(PROMPTS_PATH, "wb") as f:
        pickle.dump(tokenized, f)
    lens = [len(t) for t in tokenized]
    print(f"Prompts: tokenized {len(tokenized)} prompts, "
          f"len min/med/max = {min(lens)}/{statistics.median(lens):.0f}/{max(lens)}, "
          f"saved to {PROMPTS_PATH}")


def _greedy_generate_reference(model, tokenizer, prompt_ids_list):
    """Run the stock model greedily on each prompt; return list of generated-id lists."""
    outputs = []
    for prompt_ids in prompt_ids_list:
        input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Keep only the generated continuation (strip the prompt prefix)
        gen = out[0, input_ids.shape[1]:].tolist()
        outputs.append(gen)
    return outputs


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_reference():
    """
    Generate reference outputs twice from the stock model and store them along
    with a per-prompt `stable_prefix_len` — the longest prefix that matches
    across two independent runs. This tolerates fp non-determinism without
    inflating the correctness window.
    """
    if os.path.exists(REFERENCE_PATH):
        print(f"Reference: already built at {REFERENCE_PATH}")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(PROMPTS_PATH, "rb") as f:
        prompt_ids_list = pickle.load(f)

    print(f"Reference: loading stock {MODEL_ID} in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    torch.manual_seed(SEED)

    print("Reference: generating pass 1...")
    out1 = _greedy_generate_reference(model, tokenizer, prompt_ids_list)
    print("Reference: generating pass 2...")
    out2 = _greedy_generate_reference(model, tokenizer, prompt_ids_list)

    stable_len = [_common_prefix_len(a, b) for a, b in zip(out1, out2)]
    for i, (a, b, k) in enumerate(zip(out1, out2, stable_len)):
        flag = "" if k == MAX_NEW_TOKENS else f"  (NOTE: only {k}/{MAX_NEW_TOKENS} tokens stable across runs)"
        print(f"  prompt {i:02d}: gen_len={len(a)}/{len(b)}  stable={k}{flag}")

    payload = {"reference": out1, "stable_prefix_len": stable_len}
    with open(REFERENCE_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Reference: saved to {REFERENCE_PATH}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Runtime utilities (imported by optimize.py)
# ---------------------------------------------------------------------------

def load_prompts():
    """Return list[list[int]] of tokenized prompt ids."""
    with open(PROMPTS_PATH, "rb") as f:
        return pickle.load(f)


def load_reference():
    """Return (list[list[int]] reference outputs, list[int] stable_prefix_len)."""
    with open(REFERENCE_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["reference"], payload["stable_prefix_len"]


# ---------------------------------------------------------------------------
# Fixed measurement harness (DO NOT CHANGE — ground-truth metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_correctness(generate_fn):
    """
    Compare greedy outputs of `generate_fn` against the stored reference.
    Uses the per-prompt stable prefix length (computed at reference time) so
    floating-point non-determinism alone does not fail a run.

    Returns (ok: bool, info: str).
    """
    prompts = load_prompts()
    reference, stable_len = load_reference()
    for i, (prompt_ids, ref, k) in enumerate(zip(prompts, reference, stable_len)):
        if k == 0:
            continue  # nothing reliably comparable for this prompt
        input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        out = generate_fn(input_ids, MAX_NEW_TOKENS)
        gen = out[0, input_ids.shape[1]:].tolist()
        if len(gen) < k:
            return False, f"prompt {i}: generated only {len(gen)} tokens, need {k}"
        for j in range(k):
            if gen[j] != ref[j]:
                return False, (
                    f"prompt {i} token {j}: got {gen[j]}, ref {ref[j]} "
                    f"(stable prefix len for this prompt was {k})"
                )
    return True, "ok"


@torch.no_grad()
def measure_latency(generate_fn):
    """
    Measure wall-clock latency of `generate_fn(prompt_ids, max_new_tokens)`
    across the fixed prompt set. Returns mean(per-prompt median) in ms.

    Protocol per prompt:
      - WARMUP_RUNS warmup calls (discarded)
      - MEASURE_RUNS measured calls
      - median of the measured runs is the per-prompt latency
    Final metric: mean of per-prompt medians.
    """
    prompts = load_prompts()
    per_prompt_medians = []
    for i, prompt_ids in enumerate(prompts):
        input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        # Warmup (includes first-call compilation / cudnn autotune costs)
        for _ in range(WARMUP_RUNS):
            _ = generate_fn(input_ids, MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        measurements_ms = []
        for _ in range(MEASURE_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = generate_fn(input_ids, MAX_NEW_TOKENS)
            end.record()
            torch.cuda.synchronize()
            measurements_ms.append(start.elapsed_time(end))
        median_ms = statistics.median(measurements_ms)
        per_prompt_medians.append(median_ms)
        print(f"  prompt {i:02d}: median={median_ms:.2f}ms "
              f"(runs: {', '.join(f'{m:.1f}' for m in measurements_ms)})")
    return statistics.mean(per_prompt_medians)


def print_summary(latency_ms, correctness_ok, peak_vram_mb, total_seconds, info=""):
    """
    Print the final summary block. The grep contract for the agent loop is:
        grep "^latency_ms:\\|^correctness_ok:\\|^peak_vram_mb:" run.log
    """
    import math as _math
    print("---")
    if latency_ms is None or (isinstance(latency_ms, float) and _math.isnan(latency_ms)):
        print("latency_ms:       nan")
    else:
        print(f"latency_ms:       {latency_ms:.3f}")
    print(f"correctness_ok:   {str(bool(correctness_ok)).lower()}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"max_new_tokens:   {MAX_NEW_TOKENS}")
    print(f"n_prompts:        {N_PROMPTS}")
    print(f"warmup_runs:      {WARMUP_RUNS}")
    print(f"measure_runs:     {MEASURE_RUNS}")
    if info:
        print(f"info:             {info}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model + reference for latency autoresearch")
    parser.add_argument("--force-reference", action="store_true",
                        help="Rebuild reference even if cached.")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    os.makedirs(CACHE_DIR, exist_ok=True)

    if args.force_reference and os.path.exists(REFERENCE_PATH):
        os.remove(REFERENCE_PATH)

    download_model()
    print()
    build_prompt_set()
    print()
    build_reference()
    print()
    print("Done! Ready to run experiments with `uv run optimize.py`.")
