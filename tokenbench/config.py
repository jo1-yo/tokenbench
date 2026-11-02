"""YAML config: which endpoint to hit, which models to compare, and their prices."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    id: str  # model id the API expects, e.g. "meta-llama/Llama-3.3-70B-Instruct"
    label: str  # short name used in tables and charts
    price_in: float | None = None  # USD per 1M input tokens
    price_out: float | None = None  # USD per 1M output tokens

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000


@dataclass
class Config:
    base_url: str
    api_key_env: str
    models: list[ModelSpec]
    router_small: str | None = None  # label of the cheap model
    router_large: str | None = None  # label of the strong model
    extra: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if key:
            return key
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return "mock"  # the mock server does not check keys
        raise SystemExit(
            f"Set ${self.api_key_env} (e.g. `export {self.api_key_env}=...`) before running against {self.base_url}"
        )

    def model(self, label: str) -> ModelSpec:
        for m in self.models:
            if m.label == label:
                return m
        raise KeyError(f"No model with label {label!r} in config")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    provider = raw["provider"]
    models = [ModelSpec(**m) for m in raw["models"]]
    router = raw.get("router") or {}
    return Config(
        base_url=provider["base_url"],
        api_key_env=provider.get("api_key_env", "NEBIUS_API_KEY"),
        models=models,
        router_small=router.get("small"),
        router_large=router.get("large"),
    )
