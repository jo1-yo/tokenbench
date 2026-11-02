"""Run a small open model locally with PyTorch + Transformers and measure three optimizations
that serverless inference platforms rely on — from the inside this time:

  1. KV cache      generate with use_cache=True vs False
  2. batching      aggregate tokens/s at batch size 1, 4, 8
  3. precision     float32 vs half precision (weights memory + speed)

Usage:
    pip install -e ".[local]"
    python local/hf_inference_bench.py                       # Qwen2.5-0.5B-Instruct, ~1 GB download
    python local/hf_inference_bench.py --model Qwen/Qwen2.5-1.5B-Instruct --new-tokens 64

Works on CUDA, Apple Silicon (MPS) or CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Write a short poem about autumn leaves.",
    "List three tips for writing clean Python code.",
    "What is the capital of Japan, and what is it known for?",
    "Describe a transformer model to a ten-year-old.",
    "Give two reasons to use batching in model serving.",
    "Summarize the plot of a heist movie in three sentences.",
    "What does quantization do to a neural network?",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load(model_id: str, dtype: torch.dtype, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # decoder-only models must pad on the left for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return tok, model


def weights_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


@torch.inference_mode()
def generate(tok, model, device, prompts, new_tokens, use_cache=True) -> dict:
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    batch = tok(texts, return_tensors="pt", padding=True).to(device)
    sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        **batch,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,  # force equal lengths so runs are comparable
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tok.pad_token_id,
    )
    sync(device)
    dt = time.perf_counter() - t0
    generated = (out.shape[1] - batch["input_ids"].shape[1]) * out.shape[0]
    return {"seconds": dt, "generated_tokens": generated, "tok_per_s": generated / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--out", default="runs/local_hf.json")
    args = ap.parse_args()

    device = pick_device()
    half = torch.float16 if device in ("cuda", "mps") else torch.bfloat16
    print(f"device={device} model={args.model}")
    results = {"device": device, "model": args.model, "new_tokens": args.new_tokens}

    tok, model = load(args.model, half, device)
    generate(tok, model, device, PROMPTS[:1], 8)  # warm-up (kernel compilation, allocator)

    # 1. KV cache
    with_cache = generate(tok, model, device, PROMPTS[:1], args.new_tokens, use_cache=True)
    no_cache = generate(tok, model, device, PROMPTS[:1], args.new_tokens, use_cache=False)
    results["kv_cache"] = {"on": with_cache, "off": no_cache, "speedup": with_cache["tok_per_s"] / no_cache["tok_per_s"]}
    print(f"KV cache: {with_cache['tok_per_s']:.1f} tok/s on vs {no_cache['tok_per_s']:.1f} off "
          f"({results['kv_cache']['speedup']:.1f}x)")

    # 2. batching
    results["batching"] = {}
    for bs in (1, 4, 8):
        r = generate(tok, model, device, PROMPTS[:bs], args.new_tokens)
        results["batching"][bs] = r
        print(f"batch={bs}: {r['tok_per_s']:.1f} aggregate tok/s, {r['seconds']:.2f}s per batch")

    # 3. precision
    results["precision"] = {str(half).replace("torch.", ""): {"weights_gb": weights_gb(model), **generate(tok, model, device, PROMPTS[:4], args.new_tokens)}}
    del model
    _, model32 = load(args.model, torch.float32, device)
    results["precision"]["float32"] = {"weights_gb": weights_gb(model32), **generate(tok, model32, device, PROMPTS[:4], args.new_tokens)}
    for name, r in results["precision"].items():
        print(f"{name}: weights={r['weights_gb']:.2f} GB, {r['tok_per_s']:.1f} tok/s (batch 4)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
