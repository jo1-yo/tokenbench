"""Small, fully automatic quality evals: is the cheaper/faster model good enough for the task?

Three task types that show up constantly in real customer workloads:
  math            multi-step reasoning, exact numeric answer
  extraction      text -> JSON with a fixed schema (scored per field)
  classification  pick one label from a fixed set

Everything is scored by code (no LLM-as-judge), so results are reproducible and cheap.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from statistics import mean

from .client import stream_chat
from .config import Config, ModelSpec

DATA = Path(__file__).resolve().parent.parent / "data" / "evals"
LABELS = ["billing", "bug_report", "feature_request", "account_access", "shipping"]
EXTRACTION_KEYS = ["order_id", "customer", "date", "quantity", "product"]

PROMPTS = {
    "math": (
        "Solve the problem. Think step by step briefly, then put the final numeric answer "
        "on the last line in the form 'Answer: <number>'.\n\nProblem: {input}"
    ),
    "extraction": (
        "Extract these fields from the text: " + ", ".join(EXTRACTION_KEYS) + ". "
        "Reply with only a JSON object. Use a string for order_id (digits only), an integer for "
        "quantity, YYYY-MM-DD for date, and the singular product name.\n\nText: {input}"
    ),
    "classification": (
        "Classify the customer message into exactly one of these labels: " + ", ".join(LABELS) + ". "
        "Reply with only the label.\n\nMessage: {input}"
    ),
}
MAX_TOKENS = {"math": 512, "extraction": 200, "classification": 10}


def load_task(task: str) -> list[dict]:
    path = DATA / f"{task}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------- scorers: return a score in [0, 1] ----------

_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def score_math(response: str, answer: float) -> float:
    m = re.search(r"Answer:\s*\**\s*\$?\s*(" + _NUM + ")", response, re.IGNORECASE)
    nums = [m.group(1)] if m else re.findall(_NUM, response)
    if not nums:
        return 0.0
    try:
        got = float(nums[-1].replace(",", ""))
    except ValueError:
        return 0.0
    return 1.0 if abs(got - float(answer)) <= 1e-6 * max(1.0, abs(float(answer))) + 1e-9 else 0.0


def _parse_json_object(response: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", response)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _norm(v) -> str:
    s = str(v).strip().lower().lstrip("#")
    s = re.sub(r"[^\w\s'-]", "", s)
    return re.sub(r"s$", "", s)  # "mugs" ~ "mug"


def score_extraction(response: str, answer: dict) -> float:
    obj = _parse_json_object(response)
    if obj is None:
        return 0.0
    correct = 0
    for key, expected in answer.items():
        got = obj.get(key)
        if got is None:
            continue
        a, b = _norm(got), _norm(expected)
        if a == b or (key == "product" and (a in b or b in a)):
            correct += 1
    return correct / len(answer)


def score_classification(response: str, answer: str) -> float:
    text = response.strip().lower().replace(" ", "_").replace("-", "_")
    found = sorted((text.find(lbl), lbl) for lbl in LABELS if lbl in text)
    return 1.0 if found and found[0][1] == answer else 0.0


SCORERS = {"math": score_math, "extraction": score_extraction, "classification": score_classification}


def build_prompt(task: str, item: dict) -> str:
    return PROMPTS[task].format(input=item["input"])


# ---------- runner ----------


async def eval_model(client, spec: ModelSpec, tasks: list[str], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(task: str, item: dict) -> dict:
        prompt = build_prompt(task, item)
        async with sem:
            r = await stream_chat(client, spec.id, [{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS[task])
        return {
            "model": spec.label,
            "task": task,
            "id": item["id"],
            "prompt": prompt,
            "score": SCORERS[task](r.text, item["answer"]) if r.ok else 0.0,
            "ok": r.ok,
            "error": r.error,
            "response": r.text,
            "e2e_s": r.e2e_s,
            "ttft_s": r.ttft_s,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": spec.cost_usd(r.input_tokens, r.output_tokens),
        }

    jobs = [one(task, item) for task in tasks for item in load_task(task)]
    return list(await asyncio.gather(*jobs))


def summarize_evals(items: list[dict]) -> list[dict]:
    rows = []
    for model in dict.fromkeys(i["model"] for i in items):
        for task in [*dict.fromkeys(i["task"] for i in items), "all"]:
            sel = [i for i in items if i["model"] == model and (task == "all" or i["task"] == task)]
            costs = [i["cost_usd"] for i in sel]
            rows.append({
                "model": model,
                "task": task,
                "n": len(sel),
                "accuracy": mean(i["score"] for i in sel),
                "errors": sum(not i["ok"] for i in sel),
                "mean_e2e_s": mean(i["e2e_s"] for i in sel),
                "mean_output_tokens": mean(i["output_tokens"] for i in sel),
                "total_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            })
    return rows


async def run_evals(cfg: Config, client, out_dir: Path, tasks: list[str] | None = None, concurrency: int = 4) -> list[dict]:
    tasks = tasks or list(PROMPTS)
    items = []
    for spec in cfg.models:
        items.extend(await eval_model(client, spec, tasks, concurrency))
    rows = summarize_evals(items)
    for row in rows:
        if row["task"] == "all":
            cost = "n/a" if row["total_cost_usd"] is None else f"${row['total_cost_usd']:.4f}"
            print(f"  {row['model']:<20} accuracy={row['accuracy']:.1%}  mean_latency={row['mean_e2e_s']:.2f}s  cost={cost}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps({"summary": rows, "items": items}, indent=2))
    return rows
