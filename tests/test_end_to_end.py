"""Run the whole pipeline against the mock server (no network, no API key)."""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from tokenbench.cache_exp import run_cache_experiment
from tokenbench.client import make_client
from tokenbench.config import Config, ModelSpec
from tokenbench.evals import run_evals
from tokenbench.loadtest import sweep
from tokenbench.mock_server import create_app as create_mock
from tokenbench.report import build_report
from tokenbench.router import create_app as create_router
from tokenbench.router import route_eval


def _serve(app) -> tuple[str, uvicorn.Server]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/v1/", server


@pytest.fixture(scope="module")
def cfg():
    url, server = _serve(create_mock())
    yield Config(
        base_url=url,
        api_key_env="UNUSED_KEY",
        models=[ModelSpec("mock-small", "mock-small", 0.02, 0.06), ModelSpec("mock-large", "mock-large", 0.13, 0.40)],
        router_small="mock-small",
        router_large="mock-large",
    )
    server.should_exit = True


def test_full_pipeline(cfg, tmp_path):
    client = make_client(cfg.base_url, cfg.api_key)

    summaries = asyncio.run(sweep(cfg, client, [1, 4], 4, 32, tmp_path))
    assert len(summaries) == 4 and all(s["errors"] == 0 for s in summaries)
    by = {(s["model"], s["concurrency"]): s for s in summaries}
    assert by[("mock-small", 1)]["ttft_p50"] < by[("mock-large", 1)]["ttft_p50"]
    # batching: more concurrency -> more aggregate throughput
    assert by[("mock-large", 4)]["output_tok_per_s"] > by[("mock-large", 1)]["output_tok_per_s"]

    rows = asyncio.run(run_evals(cfg, client, tmp_path, ["classification"], concurrency=8))
    assert {r["model"] for r in rows} == {"mock-small", "mock-large"}

    cache = asyncio.run(run_cache_experiment(cfg, client, tmp_path))
    assert all(r["speedup"] > 1.2 for r in cache)
    assert all(r["cached_tokens_shared_mean"] > 0 for r in cache)
    assert all(r["cached_tokens_busted_mean"] == 0 for r in cache)

    route = route_eval(cfg, tmp_path)
    assert set(route) == {"always_small", "always_large", "routed"}

    report = build_report(tmp_path)
    text = report.read_text()
    assert "Load test" in text and "Prefix caching" in text
    assert (tmp_path / "charts" / "loadtest.png").exists()
    json.loads((tmp_path / "loadtest.json").read_text())


def test_router_proxy(cfg):
    url, server = _serve(create_router(cfg))
    try:
        base = url.removesuffix("v1/")
        r = httpx.post(base + "v1/chat/completions", json={"messages": [{"role": "user", "content": "classify: hi. Reply with only the label"}], "max_tokens": 4})
        assert r.status_code == 200 and r.headers["x-routed-model"] == "mock-small"
        r = httpx.post(base + "v1/chat/completions", headers={"x-tier": "large"}, json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True})
        assert r.headers["x-routed-model"] == "mock-large" and "data: [DONE]" in r.text
    finally:
        server.should_exit = True
