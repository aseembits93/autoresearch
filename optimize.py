"""
Baseline inference script for latency autoresearch. Single GPU, single file.

The agent edits this file to reduce `latency_ms` while keeping
`correctness_ok` true. The measurement harness and reference outputs live in
`prepare.py` and `~/.cache/autoresearch-latency/` respectively — those are
off-limits.

Usage: uv run optimize.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import (
    MODEL_ID,
    SEED,
    check_correctness,
    measure_latency,
    print_summary,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Model + generate function (fair game to modify)
# ---------------------------------------------------------------------------

def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model


def build_generate_fn(model, tokenizer):
    """
    Return a callable `generate(input_ids, max_new_tokens) -> full_output_ids`.
    The harness strips the prompt prefix itself, so return the concatenated
    [prompt, generation] tensor (HF's default).
    """
    @torch.no_grad()
    def generate(input_ids, max_new_tokens):
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return generate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = build_model()
generate_fn = build_generate_fn(model, tokenizer)

print("Checking correctness against reference...")
ok, info = check_correctness(generate_fn)
print(f"Correctness: {'OK' if ok else 'MISMATCH'} ({info})")

if ok:
    print("Measuring latency...")
    latency_ms = measure_latency(generate_fn)
else:
    latency_ms = float("nan")

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
total_seconds = time.time() - t_start

print_summary(
    latency_ms=latency_ms,
    correctness_ok=ok,
    peak_vram_mb=peak_vram_mb,
    total_seconds=total_seconds,
    info=info if not ok else "",
)
