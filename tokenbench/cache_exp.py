"""Prefix-caching experiment: does reusing the same long system prompt make TTFT faster?

Most inference engines (vLLM, SGLang, TensorRT-LLM) keep the KV cache of recent prompt prefixes.
If many requests share the same long system prompt, the server can skip recomputing it (prefill),
so time-to-first-token should drop. We test this from the outside:

  shared  - every request uses the exact same ~2k-token system prompt (cache-friendly)
  busted  - same prompt, but a random id is put at the very START, so no prefix ever matches

The two conditions are interleaved so drift in server load affects both equally.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from statistics import median

from .client import stream_chat
from .config import Config

TOPICS = [
    "returns", "warranty", "shipping times", "international orders", "payment methods",
    "account security", "data privacy", "refund timelines", "gift cards", "bulk discounts",
    "repairs", "student pricing", "subscription changes", "accessibility", "loyalty points",
]

QUESTIONS = [
    "How long do I have to return an item?",
    "Do you ship to Canada?",
    "Can I pay with a gift card and a credit card together?",
    "What happens to my loyalty points if I cancel?",
    "Is there a discount for buying 50 units?",
    "How do I enable two-factor authentication?",
    "How long does a refund take to show up?",
    "Can I get a repair after the warranty ends?",
    "Do students get a discount?",
    "How is my personal data stored?",
]


def build_handbook(sections: int = 45) -> str:
    """Deterministic ~2k-token 'company handbook' used as a long shared system prompt."""
    lines = ["You are a support assistant for Acme Robotics. Answer in one sentence using the policy handbook below.", ""]
    for i in range(sections):
        topic = TOPICS[i % len(TOPICS)]
        lines.append(
            f"Section {i + 1} ({topic}): Requests about {topic} are handled by tier {i % 3 + 1} support. "
            f"Customers must provide their order number, and agents respond within {i % 5 + 1} business days. "
            f"Exceptions require approval from a team lead and must be logged in the {topic} tracker."
        )
    return "\n".join(lines)


async def run_cache_experiment(cfg: Config, client, out_dir: Path, rounds: int = 1) -> list[dict]:
    handbook = build_handbook()
    rows, raw = [], []
    for spec in cfg.models:
        # Prime the cache for the shared prefix.
        await stream_chat(client, spec.id, [{"role": "system", "content": handbook}, {"role": "user", "content": "Hi"}], max_tokens=4)
        ttft = {"shared": [], "busted": []}
        cached = {"shared": [], "busted": []}
        for _ in range(rounds):
            for q in QUESTIONS:
                for cond in ("shared", "busted"):
                    system = handbook if cond == "shared" else f"[session {uuid.uuid4()}]\n{handbook}"
                    r = await stream_chat(
                        client, spec.id, [{"role": "system", "content": system}, {"role": "user", "content": q}], max_tokens=16
                    )
                    raw.append({"model": spec.label, "condition": cond, **r.to_dict()})
                    if r.ok and r.ttft_s is not None:
                        ttft[cond].append(r.ttft_s)
                    if r.cached_tokens is not None:
                        cached[cond].append(r.cached_tokens)
        s, b = median(ttft["shared"]) if ttft["shared"] else None, median(ttft["busted"]) if ttft["busted"] else None
        row = {
            "model": spec.label,
            "prompt_tokens": raw[-1]["input_tokens"] if raw else None,
            "ttft_shared_p50": s,
            "ttft_busted_p50": b,
            "speedup": (b / s) if s and b else None,
            # None means the provider doesn't report cached tokens; then TTFT is the only evidence.
            "cached_tokens_shared_mean": sum(cached["shared"]) / len(cached["shared"]) if cached["shared"] else None,
            "cached_tokens_busted_mean": sum(cached["busted"]) / len(cached["busted"]) if cached["busted"] else None,
        }
        rows.append(row)
        speed = "n/a" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        print(f"  {spec.label:<20} ttft shared={_f(s)}s busted={_f(b)}s  speedup={speed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache.json").write_text(json.dumps({"summary": rows, "requests": raw}, indent=2))
    return rows


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"
