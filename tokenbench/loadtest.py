"""Concurrency sweep: how do latency and throughput change as more users hit the endpoint?

This is a *closed-loop* load test: at concurrency C there are always exactly C requests in
flight; as soon as one finishes the next one starts. Real traffic is open-loop (arrivals don't
wait for you), but closed-loop sweeps are the standard way to draw the latency/throughput curve.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .client import RequestResult, stream_chat
from .config import Config, ModelSpec
from .metrics import summarize

DATA = Path(__file__).resolve().parent.parent / "data"


def load_prompts(path: Path = DATA / "prompts.jsonl") -> list[str]:
    return [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]


async def run_level(client, spec: ModelSpec, prompts: list[str], concurrency: int, num_requests: int, max_tokens: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> RequestResult:
        async with sem:
            msgs = [{"role": "user", "content": prompts[i % len(prompts)]}]
            # temperature > 0 so repeated prompts don't come back identical from a response cache
            return await stream_chat(client, spec.id, msgs, max_tokens=max_tokens, temperature=0.7)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(num_requests)))
    wall = time.perf_counter() - t0
    summary = summarize(list(results), wall)
    summary.update(model=spec.label, concurrency=concurrency)
    in_tok, out_tok = summary["input_tokens"], summary["output_tokens"]
    cost = spec.cost_usd(in_tok, out_tok)
    ok = summary["requests"] - summary["errors"]
    summary["usd_per_1k_requests"] = cost / ok * 1000 if cost is not None and ok else None
    return summary, list(results)


async def sweep(cfg: Config, client, levels: list[int], requests_per_level: int, max_tokens: int, out_dir: Path) -> list[dict]:
    prompts = load_prompts()
    summaries, raw = [], []
    for spec in cfg.models:
        # Warm-up: the first request can pay for connection setup or a cold start.
        await stream_chat(client, spec.id, [{"role": "user", "content": "Say hi."}], max_tokens=8)
        for c in levels:
            n = max(requests_per_level, c * 2)
            summary, results = await run_level(client, spec, prompts, c, n, max_tokens)
            summaries.append(summary)
            raw.extend({"model": spec.label, "concurrency": c, **r.to_dict()} for r in results)
            print(
                f"  {spec.label:<20} c={c:<3} ttft_p50={_fmt(summary['ttft_p50'])}s "
                f"tpot_p50={_fmt(summary['tpot_p50'], 4)}s  {summary['output_tok_per_s']:.0f} tok/s  "
                f"errors={summary['errors']}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loadtest.json").write_text(json.dumps({"summaries": summaries, "requests": raw}, indent=2))
    return summaries


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"
