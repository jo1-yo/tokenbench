"""Command line entry point: `tokenbench <command> ...`."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from .config import load_config


def _client(cfg):
    from .client import make_client

    return make_client(cfg.base_url, cfg.api_key)


def _out(args) -> Path:
    return Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d-%H%M%S")


def cmd_models(args):
    cfg = load_config(args.config)

    async def go():
        page = await _client(cfg).models.list()
        for m in page.data:
            print(m.id)

    asyncio.run(go())


def cmd_loadtest(args):
    from .loadtest import sweep

    cfg, out = load_config(args.config), _out(args)
    levels = [int(x) for x in args.levels.split(",")]
    print(f"Load test -> {out}")
    asyncio.run(sweep(cfg, _client(cfg), levels, args.requests, args.max_tokens, out))


def cmd_eval(args):
    from .evals import run_evals

    cfg, out = load_config(args.config), _out(args)
    tasks = args.tasks.split(",") if args.tasks else None
    print(f"Evals -> {out}")
    asyncio.run(run_evals(cfg, _client(cfg), out, tasks, args.concurrency))


def cmd_cache(args):
    from .cache_exp import run_cache_experiment

    cfg, out = load_config(args.config), _out(args)
    print(f"Prefix-cache experiment -> {out}")
    asyncio.run(run_cache_experiment(cfg, _client(cfg), out, args.rounds))


def cmd_route_eval(args):
    from .router import route_eval

    print("Routing policies (replayed on eval results):")
    route_eval(load_config(args.config), Path(args.run))


def cmd_report(args):
    from .report import build_report

    print(f"Wrote {build_report(Path(args.run))}")


def cmd_all(args):
    from .cache_exp import run_cache_experiment
    from .evals import run_evals
    from .loadtest import sweep
    from .report import build_report
    from .router import route_eval

    cfg, out = load_config(args.config), _out(args)
    levels = [int(x) for x in args.levels.split(",")]

    async def go():
        client = _client(cfg)
        print("== Load test"); await sweep(cfg, client, levels, args.requests, args.max_tokens, out)
        print("== Evals"); await run_evals(cfg, client, out)
        print("== Prefix cache"); await run_cache_experiment(cfg, client, out)

    asyncio.run(go())
    if cfg.router_small and cfg.router_large:
        print("== Routing"); route_eval(cfg, out)
    print(f"== Report: {build_report(out)}")


def cmd_router(args):
    import uvicorn

    from .router import create_app

    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port)


def cmd_mock(args):
    import uvicorn

    from .mock_server import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tokenbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, config=True, out=True):
        sp = sub.add_parser(name, help=help_)
        if config:
            sp.add_argument("-c", "--config", default="configs/nebius.yaml")
        if out:
            sp.add_argument("-o", "--out", help="run directory (default: runs/<timestamp>)")
        sp.set_defaults(fn=fn)
        return sp

    add("models", cmd_models, "list model ids the endpoint serves", out=False)

    for name, fn, help_ in [("loadtest", cmd_loadtest, "concurrency sweep: latency + throughput"), ("all", cmd_all, "loadtest + eval + cache + routing + report")]:
        sp = add(name, fn, help_)
        sp.add_argument("--levels", default="1,2,4,8,16", help="comma-separated concurrency levels")
        sp.add_argument("--requests", type=int, default=16, help="requests per level (at least 2x concurrency)")
        sp.add_argument("--max-tokens", type=int, default=256)

    sp = add("eval", cmd_eval, "quality evals (math, extraction, classification)")
    sp.add_argument("--tasks", help="comma-separated subset of tasks")
    sp.add_argument("--concurrency", type=int, default=4)

    sp = add("cache", cmd_cache, "prefix-caching TTFT experiment")
    sp.add_argument("--rounds", type=int, default=1)

    sp = add("route-eval", cmd_route_eval, "replay eval results through the router", out=False)
    sp.add_argument("run", help="run directory containing eval.json")

    sp = add("report", cmd_report, "charts + REPORT.md for a run", config=False, out=False)
    sp.add_argument("run")

    for name, fn, help_, port, cfg in [("router", cmd_router, "serve the OpenAI-compatible routing proxy", 8080, True), ("mock", cmd_mock, "serve the fake OpenAI-compatible endpoint", 8000, False)]:
        sp = add(name, fn, help_, config=cfg, out=False)
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port", type=int, default=port)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
