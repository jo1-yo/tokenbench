"""Turn a run directory (loadtest.json, eval.json, cache.json, route_eval.json) into charts + REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(run_dir: Path, name: str):
    p = run_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, fmt="{:.3f}"):
    return "–" if v is None else fmt.format(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _loadtest_charts(summaries: list[dict], charts: Path) -> list[str]:
    models = list(dict.fromkeys(s["model"] for s in summaries))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in models:
        rows = sorted((s for s in summaries if s["model"] == m), key=lambda s: s["concurrency"])
        c = [s["concurrency"] for s in rows]
        axes[0].plot(c, [s["ttft_p50"] for s in rows], marker="o", label=m)
        axes[1].plot(c, [s["output_tok_per_s"] for s in rows], marker="o", label=m)
        axes[2].plot([s["output_tok_per_s"] for s in rows], [s["e2e_p50"] for s in rows], marker="o", label=m)
        for s in rows:
            axes[2].annotate(f"c={s['concurrency']}", (s["output_tok_per_s"], s["e2e_p50"]), fontsize=7, alpha=0.7)
    axes[0].set(title="TTFT p50 vs concurrency", xlabel="concurrent requests", ylabel="seconds", xscale="log")
    axes[1].set(title="Aggregate output throughput", xlabel="concurrent requests", ylabel="tokens / s", xscale="log")
    axes[2].set(title="Latency–throughput trade-off", xlabel="aggregate tokens / s", ylabel="E2E p50 (s)")
    levels = sorted({s["concurrency"] for s in summaries})
    for ax in axes[:2]:
        ax.set_xticks(levels, [str(c) for c in levels])
        ax.minorticks_off()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts / "loadtest.png", dpi=140)
    plt.close(fig)
    return ["loadtest.png"]


def _quality_chart(summary: list[dict], charts: Path) -> list[str]:
    rows = [r for r in summary if r["task"] == "all"]
    use_cost = all(r["total_cost_usd"] is not None for r in rows)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in rows:
        x = r["total_cost_usd"] if use_cost else r["mean_e2e_s"]
        ax.scatter(x, r["accuracy"], s=60)
        ax.annotate(r["model"], (x, r["accuracy"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set(
        title="Quality vs " + ("cost" if use_cost else "latency"),
        xlabel="total eval cost (USD)" if use_cost else "mean latency (s)",
        ylabel="accuracy (all tasks)",
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts / "quality.png", dpi=140)
    plt.close(fig)
    return ["quality.png"]


def build_report(run_dir: Path) -> Path:
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    parts = [f"# tokenbench report — `{run_dir.name}`\n"]

    lt = _load(run_dir, "loadtest.json")
    if lt:
        _loadtest_charts(lt["summaries"], charts)
        parts.append("## Load test\n\n![load test](charts/loadtest.png)\n")
        parts.append(_table(
            ["model", "conc.", "TTFT p50", "TTFT p99", "TPOT p50", "E2E p50", "agg tok/s", "per-req tok/s", "errors", "$/1k req"],
            [[s["model"], str(s["concurrency"]), _f(s["ttft_p50"]), _f(s["ttft_p99"]), _f(s["tpot_p50"], "{:.4f}"),
              _f(s["e2e_p50"]), _f(s["output_tok_per_s"], "{:.0f}"), _f(s["per_request_tok_per_s"], "{:.1f}"),
              f"{s['errors']}/{s['requests']}", _f(s.get("usd_per_1k_requests"), "{:.3f}")] for s in lt["summaries"]],
        ))
        parts.append("")

    ev = _load(run_dir, "eval.json")
    if ev:
        _quality_chart(ev["summary"], charts)
        parts.append("## Quality evals\n\n![quality](charts/quality.png)\n")
        parts.append(_table(
            ["model", "task", "n", "accuracy", "mean latency (s)", "mean out tokens", "cost (USD)", "errors"],
            [[r["model"], r["task"], str(r["n"]), f"{r['accuracy']:.1%}", _f(r["mean_e2e_s"], "{:.2f}"),
              _f(r["mean_output_tokens"], "{:.0f}"), _f(r["total_cost_usd"], "{:.4f}"), str(r["errors"])] for r in ev["summary"]],
        ))
        parts.append("")

    re_ = _load(run_dir, "route_eval.json")
    if re_:
        parts.append("## Routing (replayed on the eval run)\n")
        parts.append(_table(
            ["policy", "accuracy", "mean latency (s)", "cost (USD)", "% sent to large"],
            [[k, f"{v['accuracy']:.1%}", _f(v["mean_e2e_s"], "{:.2f}"), _f(v["total_cost_usd"], "{:.4f}"), f"{v['share_large']:.0%}"]
             for k, v in re_.items()],
        ))
        parts.append("")

    ca = _load(run_dir, "cache.json")
    if ca:
        parts.append("## Prefix caching\n")
        parts.append(_table(
            ["model", "prompt tokens", "TTFT shared prefix", "TTFT busted prefix", "speedup", "cached tokens (shared / busted)"],
            [[r["model"], str(r["prompt_tokens"]), _f(r["ttft_shared_p50"]), _f(r["ttft_busted_p50"]), _f(r["speedup"], "{:.2f}x"),
              f"{_f(r['cached_tokens_shared_mean'], '{:.0f}')} / {_f(r['cached_tokens_busted_mean'], '{:.0f}')}"] for r in ca["summary"]],
        ))
        parts.append("")

    out = run_dir / "REPORT.md"
    out.write_text("\n".join(parts))
    return out
