"""A fake OpenAI-compatible server for offline development and CI.

It is not a model. It imitates the *latency shape* of a real inference server so the
benchmark code can be exercised without spending tokens:

- prefill time grows with prompt length, but blocks of the prompt that were seen before are
  ~free (imitating vLLM-style automatic prefix caching: the prompt is split into fixed-size
  blocks, each identified by a hash of itself *and everything before it*, so only an exact
  shared prefix can hit),
- decode time per token grows with the number of in-flight requests (imitating a batch that
  gets bigger and slower per step, but higher in total throughput).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PROFILES = {
    # base_s: fixed overhead, prefill_s: per input token, decode_s: per output token at batch size 1
    "mock-small": {"base_s": 0.03, "prefill_s": 0.00004, "decode_s": 0.004},
    "mock-large": {"base_s": 0.08, "prefill_s": 0.00015, "decode_s": 0.012},
}
CACHE_BLOCK_CHARS = 256  # ~64 tokens, a typical KV-cache block size is 16 tokens; coarser is fine here
CACHED_PREFILL_FRACTION = 0.1
BATCH_SLOWDOWN = 0.06  # each extra in-flight request makes a decode step 6% slower
FILLER = "the quick brown fox jumps over the lazy dog while the model streams tokens".split()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def cached_blocks(model: str, prompt: str, seen: set) -> int:
    """Count leading full blocks already in the cache, then add this prompt's blocks to it."""
    h = hashlib.sha256(model.encode())
    hits, still_matching = 0, True
    for i in range(len(prompt) // CACHE_BLOCK_CHARS):
        h.update(prompt[i * CACHE_BLOCK_CHARS : (i + 1) * CACHE_BLOCK_CHARS].encode())
        key = h.hexdigest()
        if still_matching and key in seen:
            hits += 1
        else:
            still_matching = False
        seen.add(key)
    return hits


def create_app() -> FastAPI:
    app = FastAPI(title="tokenbench mock")
    state = {"active": 0, "seen": set()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in PROFILES]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        model = body.get("model")
        if model not in PROFILES:
            return JSONResponse({"error": {"message": f"unknown model {model}"}}, status_code=404)
        prof = PROFILES[model]
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        in_tok = approx_tokens(prompt)
        n_out = min(int(body.get("max_tokens") or 64), 64)

        cached = cached_blocks(model, prompt, state["seen"]) * CACHE_BLOCK_CHARS // 4
        prefill = prof["prefill_s"] * (in_tok - cached + cached * CACHED_PREFILL_FRACTION)

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        usage = {
            "prompt_tokens": in_tok,
            "completion_tokens": n_out,
            "total_tokens": in_tok + n_out,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        words = [FILLER[i % len(FILLER)] for i in range(n_out)]

        async def decode_step():
            await asyncio.sleep(prof["decode_s"] * (1 + BATCH_SLOWDOWN * max(0, state["active"] - 1)))

        if not body.get("stream"):
            state["active"] += 1
            try:
                await asyncio.sleep(prof["base_s"] + prefill)
                for _ in words:
                    await decode_step()
            finally:
                state["active"] -= 1
            return {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(words)}, "finish_reason": "length"}],
                "usage": usage,
            }

        include_usage = (body.get("stream_options") or {}).get("include_usage", False)

        def sse(choices, extra=None):
            payload = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": choices}
            payload.update(extra or {})
            return f"data: {json.dumps(payload)}\n\n"

        async def gen():
            state["active"] += 1
            try:
                await asyncio.sleep(prof["base_s"] + prefill)
                yield sse([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
                for i, w in enumerate(words):
                    if i:
                        await decode_step()
                    yield sse([{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}])
                yield sse([{"index": 0, "delta": {}, "finish_reason": "length"}])
                if include_usage:
                    yield sse([], {"usage": usage})
                yield "data: [DONE]\n\n"
            finally:
                state["active"] -= 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
