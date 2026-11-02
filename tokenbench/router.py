"""Model routing: send easy requests to a small cheap model, hard ones to a large model.

Two pieces:
  choose_tier()  a transparent rule-based router (no extra model call, ~0 ms overhead)
  create_app()   an OpenAI-compatible FastAPI proxy that applies it, so any OpenAI SDK client
                 can point at it unchanged
  route_eval()   replays a finished eval run to measure what routing would have cost/scored,
                 without spending a single extra token
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .config import Config

HARD_PATTERNS = [
    (r"step by step|prove|derive|explain why", "asks for reasoning"),
    (r"how (much|many|old|long)|total|percent|%|average|calculate", "quantitative question"),
    (r"```|\bdef \b|\bclass \b|function|stack trace|traceback", "code"),
]
EASY_PATTERNS = [
    (r"reply with only the label|classify", "classification"),
    (r"reply with only a json|extract", "extraction"),
]


def choose_tier(text: str) -> tuple[str, list[str]]:
    """Return ("small" | "large", reasons)."""
    t = text.lower()
    reasons = [why for pat, why in EASY_PATTERNS if re.search(pat, t)]
    if reasons:
        return "small", reasons
    score, reasons = 0, []
    for pat, why in HARD_PATTERNS:
        if re.search(pat, t):
            score += 1
            reasons.append(why)
    if len(text) > 4000:
        score += 1
        reasons.append("long prompt")
    if len(re.findall(r"\d+", t)) >= 3:
        score += 1
        reasons.append("many numbers")
    return ("large" if score >= 2 else "small"), reasons or ["default"]


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def create_app(cfg: Config) -> FastAPI:
    if not (cfg.router_small and cfg.router_large):
        raise SystemExit("Config needs router.small and router.large (model labels)")
    models = {"small": cfg.model(cfg.router_small).id, "large": cfg.model(cfg.router_large).id}
    upstream = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
    app = FastAPI(title="tokenbench router")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "models": models}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        forced = request.headers.get("x-tier")
        if forced in models:
            tier, reasons = forced, ["forced by x-tier header"]
        else:
            tier, reasons = choose_tier(_last_user_text(body.get("messages", [])))
        body["model"] = models[tier]
        route_headers = {"x-routed-model": models[tier], "x-route-reason": "; ".join(reasons)}

        if body.get("stream"):
            req = client.build_request("POST", upstream, json=body, headers=headers)
            resp = await client.send(req, stream=True)
            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                media_type="text/event-stream",
                headers=route_headers,
                background=BackgroundTask(resp.aclose),
            )
        resp = await client.post(upstream, json=body, headers=headers)
        return JSONResponse(resp.json(), status_code=resp.status_code, headers=route_headers)

    return app


def route_eval(cfg: Config, run_dir: Path) -> dict:
    """Score three policies on an existing eval run: always-small, always-large, routed."""
    data = json.loads((run_dir / "eval.json").read_text())
    small, large = cfg.router_small, cfg.router_large
    by_key = {(i["model"], i["task"], i["id"]): i for i in data["items"]}
    keys = sorted({(i["task"], i["id"]) for i in data["items"]})
    missing = [m for m in (small, large) if not any(k[0] == m for k in by_key)]
    if missing:
        raise SystemExit(f"eval.json has no results for {missing}; run `tokenbench eval` with both router models")

    policies = {"always_small": [], "always_large": [], "routed": []}
    for task, iid in keys:
        s, l = by_key[(small, task, iid)], by_key[(large, task, iid)]
        policies["always_small"].append(s)
        policies["always_large"].append(l)
        policies["routed"].append(l if choose_tier(_last_user_text([{"role": "user", "content": s["prompt"]}]))[0] == "large" else s)

    out = {}
    for name, picks in policies.items():
        costs = [p["cost_usd"] for p in picks]
        out[name] = {
            "accuracy": sum(p["score"] for p in picks) / len(picks),
            "mean_e2e_s": sum(p["e2e_s"] for p in picks) / len(picks),
            "total_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "share_large": sum(p["model"] == large for p in picks) / len(picks),
        }
    (run_dir / "route_eval.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        cost = "n/a" if r["total_cost_usd"] is None else f"${r['total_cost_usd']:.4f}"
        print(f"  {name:<14} accuracy={r['accuracy']:.1%}  latency={r['mean_e2e_s']:.2f}s  cost={cost}  large={r['share_large']:.0%}")
    return out
