"""One streaming chat request, timed the way inference engineers measure it.

TTFT  (time to first token)  ~ queueing + prefill. Dominated by prompt length and load.
TPOT  (time per output token) ~ decode speed. Dominated by model size and batch size.
E2E   = TTFT + TPOT * (output_tokens - 1)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from openai import AsyncOpenAI


@dataclass
class RequestResult:
    model: str
    ok: bool
    e2e_s: float
    ttft_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None  # reported by some providers when the prompt prefix hit the KV cache
    text: str = ""
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tpot_s"] = self.tpot_s
        return d


def make_client(base_url: str, api_key: str, timeout: float = 180.0) -> AsyncOpenAI:
    # No automatic retries: a benchmark should see (and count) failures, not hide them.
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


async def stream_chat(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> RequestResult:
    start = time.perf_counter()
    ttft = None
    pieces: list[str] = []
    text_chunks = 0
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            # Reasoning models (e.g. DeepSeek-R1) stream their thinking in a separate field;
            # it still counts as a generated token for latency purposes.
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if content or reasoning:
                if ttft is None:
                    ttft = time.perf_counter() - start
                text_chunks += 1
                pieces.append(content)
    except Exception as e:  # network errors, 429s, 5xx, timeouts
        return RequestResult(
            model=model,
            ok=False,
            e2e_s=time.perf_counter() - start,
            ttft_s=ttft,
            error=f"{type(e).__name__}: {e}"[:300],
        )

    e2e = time.perf_counter() - start
    cached = None
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
    return RequestResult(
        model=model,
        ok=True,
        e2e_s=e2e,
        ttft_s=ttft,
        input_tokens=usage.prompt_tokens if usage else 0,
        # Fall back to counting streamed chunks if the provider doesn't send usage (≈1 token per chunk).
        output_tokens=usage.completion_tokens if usage else text_chunks,
        cached_tokens=cached,
        text="".join(pieces),
    )
