"""Aggregate per-request results into the numbers that go in a benchmark table."""

from __future__ import annotations

from statistics import mean

from .client import RequestResult


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[RequestResult], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    e2e = [r.e2e_s for r in ok]
    out_tokens = sum(r.output_tokens for r in ok)
    in_tokens = sum(r.input_tokens for r in ok)
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall_s,
        "rps": len(ok) / wall_s if wall_s > 0 else 0.0,
        # Aggregate throughput: what the endpoint delivers to all clients together.
        "output_tok_per_s": out_tokens / wall_s if wall_s > 0 else 0.0,
        # Per-request decode speed: what a single user feels while text streams in.
        "per_request_tok_per_s": mean(1 / t for t in tpot) if tpot else None,
        "ttft_p50": percentile(ttft, 50),
        "ttft_p90": percentile(ttft, 90),
        "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50),
        "tpot_p99": percentile(tpot, 99),
        "e2e_p50": percentile(e2e, 50),
        "e2e_p99": percentile(e2e, 99),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
    }
