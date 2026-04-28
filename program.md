# autoresearch (latency edition)

This is an experiment to have the LLM do its own research — on inference-latency optimization.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, model/prompt/reference setup, latency + correctness harness. Do not modify.
   - `optimize.py` — the file you modify. Loads the target model and defines the `generate` callable.
4. **Verify setup**: Check that `~/.cache/autoresearch-latency/` contains `prompts.pt` and `reference.pt`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script is gone — there is no training; the target model (`Qwen/Qwen3.5-0.8B`) is a fixed pre-trained checkpoint and you are optimizing its inference latency. You launch it simply as: `uv run optimize.py`.

**What you CAN do:**
- Modify `optimize.py` — this is the only source file you edit. Inference code, attention implementation, KV cache, `torch.compile`, CUDA graphs, dtype, quantization (bnb / GPTQ / AWQ / fp8), weight surgery (pruning, layer-skipping, speculative decoding draft model) — everything that changes how generation happens is fair game.
- Add dependencies. Edit `pyproject.toml` and run `uv sync`. Popular options to consider: `bitsandbytes`, `auto-gptq`, `autoawq`, `vllm`, `sglang`, `flash-attn`.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed latency measurement harness, the correctness check, the prompt set, and the target model id.
- Modify the cached reference outputs at `~/.cache/autoresearch-latency/reference.pt`. The correctness check relies on these.
- Change `MAX_NEW_TOKENS`, the prompt set, or the latency measurement protocol.
- Bypass `check_correctness` (e.g. by returning early from the `generate` callable, or by forcing `correctness_ok=True`).

**The goal is simple: get the lowest `latency_ms` while `correctness_ok` stays `true`.** Everything is fair game: swap attention implementations, quantize, compile, introduce a speculative-decoding draft, rewrite the decode loop to reuse KV caches across runs, whatever works. The only constraint is that outputs still match the reference within the per-prompt stable prefix (which the harness handles for you).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful latency gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small latency improvement that adds ugly complexity is not worth it. Removing code and getting equal or better latency is a great outcome — a simplification win. A 0.1ms improvement that adds 40 lines of hacky plumbing? Probably not worth it. A 0.1ms improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run `uv run optimize.py` unchanged.

## Output format

Once the script finishes it prints a summary like this:

```
---
latency_ms:       823.412
correctness_ok:   true
peak_vram_mb:     2158.4
total_seconds:    94.3
max_new_tokens:   128
n_prompts:        16
warmup_runs:      2
measure_runs:     5
```

Note that absolute numbers depend on your GPU — they are not comparable to runs on other hardware. You can extract the key metrics from the log file:

```
grep "^latency_ms:\|^correctness_ok:\|^peak_vram_mb:" run.log
```

If `correctness_ok` is `false`, `latency_ms` will be `nan` — the run is treated as a crash (status `crash` in the tsv, discard, revert).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	latency_ms	correctness	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `latency_ms` achieved (e.g. 823.412) — use 0.000 for crashes and for correctness failures
3. correctness: `true` or `false` — use `false` for crashes
4. peak memory in GB, round to .1f (e.g. 2.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	latency_ms	correctness	memory_gb	status	description
a1b2c3d	823.412	true	2.2	keep	baseline
b2c3d4e	742.180	true	2.2	keep	torch.compile the model
c3d4e5f	850.100	true	2.3	discard	switch to float32 (slower as expected)
d4e5f6g	0.000	false	2.2	crash	aggressive int4 quant broke correctness
e5f6g7h	0.000	false	0.0	crash	tried to import nonexistent package
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `optimize.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run optimize.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^latency_ms:\|^correctness_ok:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `correctness_ok` is `true` AND `latency_ms` is strictly lower than the running best, you "advance" the branch, keeping the git commit
9. Otherwise (equal or worse latency, or correctness failed, or crash), you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take a few minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or a correctness failure from an aggressive lossy change), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import, forgot to pass `device_map`), fix it and re-run. If the idea itself is fundamentally broken (e.g. int2 quantization destroys outputs), just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read documentation for the libraries referenced, re-read the in-scope files for new angles, try combining previous near-misses, try more radical approaches (a different attention backend, a different quantization scheme, a speculative draft, static KV cache + CUDA graphs). The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~2–3 minutes then you can run approx 20–30/hour, for a total of a couple hundred over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
