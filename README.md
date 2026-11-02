# tokenbench

**Pick the right open-source LLM for a workload — with numbers.**

`tokenbench` measures latency, throughput, quality and cost of LLMs served behind any
OpenAI-compatible API — [Nebius Token Factory](https://tokenfactory.nebius.com), a self-hosted
vLLM / SGLang server, or anything else that speaks `/v1/chat/completions` — and turns the
results into a report a customer could make a decision from.

It answers the questions that come up in almost every LLM proof-of-concept:

| Question | Experiment | Command |
|---|---|---|
| How fast is it for one user, and how does it hold up under load? | Concurrency sweep: TTFT, TPOT, E2E p50/p99, aggregate tokens/s | `tokenbench loadtest` |
| Is the small model *good enough* for my task? | Code-scored evals: math reasoning, JSON extraction, classification | `tokenbench eval` |
| Does reusing a long system prompt make it faster? | Prefix-cache A/B: shared vs cache-busted prefix | `tokenbench cache` |
| Can I save money by sending easy requests to a small model? | Rule-based router + offline replay on eval results | `tokenbench route-eval`, `tokenbench router` |
| What do KV cache, batching and precision do *inside* the model? | Local PyTorch/Transformers benchmark | `python local/hf_inference_bench.py` |

## Quick start

```bash
git clone https://github.com/jo1-yo/tokenbench && cd tokenbench
make install            # uv venv + editable install
make test               # runs everything against a built-in mock server, no key needed
```

Against Nebius Token Factory:

```bash
export NEBIUS_API_KEY=...                            # console → API keys
.venv/bin/tokenbench models -c configs/nebius.yaml   # which model ids are live
# edit configs/nebius.yaml: models to compare, prices ($/1M tokens), router small/large
make bench                                           # → runs/nebius/REPORT.md + charts
```

Offline (no key, no cost):

```bash
make mock     # terminal 1: fake OpenAI-compatible server
make demo     # terminal 2: full pipeline → runs/mock-demo/REPORT.md
```

## What gets measured, and why

**Latency is two different numbers.** A streaming response has a *prefill* phase (the model reads
the whole prompt and produces the first token) and a *decode* phase (one token at a time).

- **TTFT** — time to first token. What makes a chatbot feel responsive. Grows with prompt length
  and queueing.
- **TPOT** — time per output token after the first. What makes long answers feel slow. Grows with
  model size and with how many requests the server batches together.
- **E2E** ≈ TTFT + TPOT × (output tokens − 1).

`client.py` measures all three from the stream itself and reads token counts from the
`usage` chunk (`stream_options={"include_usage": true}`).

**Throughput vs latency is a trade-off, not a number.** The load test is closed-loop: at
concurrency *C* there are always *C* requests in flight. As *C* grows, the server batches more
requests per decode step, so *aggregate* tokens/s goes up while each individual request gets
slower. The third chart in the report (E2E vs aggregate tokens/s) shows where that curve bends —
the knee is the operating point you'd size a deployment around. Percentiles (p50/p99) are
reported because tail latency, not the average, is what users complain about.

**Quality is task-specific.** Public leaderboards don't tell you whether an 8B model can extract
your fields. The evals are small, hand-written and scored by code (exact numeric match,
per-field JSON match, label match), so a run is cheap and reproducible:

| task | n | scoring |
|---|---|---|
| math | 20 | multi-step word problems; final `Answer: <number>` must match |
| extraction | 12 | free text → JSON with 5 fields, scored per field |
| classification | 15 | support ticket → one of 5 intents |

**Prefix caching.** Engines like vLLM split the prompt into blocks and keep the KV cache of
blocks they've seen, keyed by a hash of the block *and everything before it*. So a long system
prompt shared by every request only needs to be prefilled once. `tokenbench cache` sends the
same ~2.8k-token system prompt repeatedly, interleaved with a variant that has a random id at the
very start (same length, but no prefix can ever match), and compares TTFT. If the provider
reports `usage.prompt_tokens_details.cached_tokens`, that is recorded too. Practical takeaway
for customers: put the static part of the prompt first and the variable part last.

**Routing.** `router.py` is a transparent rule-based router: structured tasks (classification,
extraction) go to the small model; prompts with signals of multi-step reasoning, math or code go
to the large one. Two ways to use it:

- `tokenbench route-eval runs/<run>` replays an existing eval run under three policies
  (always-small, always-large, routed) and reports accuracy, latency and cost of each — without
  spending extra tokens.
- `tokenbench router` serves it as an OpenAI-compatible proxy (FastAPI). Any OpenAI SDK client can
  point its `base_url` at it; the chosen model and the reason come back in `x-routed-model` /
  `x-route-reason` headers, and `x-tier: large` forces a tier. The `Dockerfile` packages it.

**Inside the model (local).** `local/hf_inference_bench.py` loads a small model
(Qwen2.5-0.5B-Instruct by default) with PyTorch + Transformers on CUDA, Apple Silicon or CPU and
measures, with everything else held fixed:

- generation with the KV cache on vs off,
- aggregate tokens/s at batch size 1, 4 and 8 (left-padded batched generation),
- half precision vs float32: weight memory and speed.

```bash
uv pip install -e ".[local]"
.venv/bin/python local/hf_inference_bench.py
```

## Project layout

```
tokenbench/
  client.py       one timed streaming request (TTFT / TPOT / E2E / tokens / cached tokens)
  metrics.py      percentiles and run summaries
  loadtest.py     closed-loop concurrency sweep
  evals.py        tasks, prompts, code-based scorers, runner
  cache_exp.py    prefix-cache A/B experiment
  router.py       choose_tier(), OpenAI-compatible routing proxy, offline route replay
  report.py       charts (matplotlib) + REPORT.md
  mock_server.py  fake OpenAI-compatible server with a realistic latency shape (tests/CI)
  cli.py          `tokenbench ...`
configs/          nebius.yaml, vllm-local.yaml, mock.yaml
data/             load-test prompts and eval sets (jsonl)
local/            PyTorch/Transformers in-process benchmark
tests/            unit tests + full pipeline against the mock server
```

## The mock server

`tokenbench mock` is not a model — it returns filler text, so eval accuracy against it is 0%. It
imitates the *latency shape* of a real server so the whole pipeline can be tested offline and in
CI: prefill cost proportional to prompt length, block-level prefix caching, and decode steps
that slow down as more requests are batched together.

## Limitations

- Closed-loop load only; real traffic arrives open-loop (Poisson), which produces worse tails at
  the same average load.
- Serverless endpoints are shared: results depend on time of day and other tenants. Run more than
  once before drawing conclusions.
- Eval sets are small (47 items) — good for spotting large gaps between models, not for
  separating models a few points apart.
- The router is rules, not a learned classifier. A natural next step is to train a small
  classifier on which items the small model actually gets wrong.

## License

MIT
