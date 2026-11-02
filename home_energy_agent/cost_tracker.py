"""Append token usage and estimated cost for each agent session."""

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COST_FILE = ROOT / "data" / "session_costs.jsonl"

# USD per 1M tokens. Update these rates if the provider's pricing changes.
MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model.lower(), {"input": 0.0, "output": 0.0})
    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


def record_session_usage(session_id: str, model: str, usage) -> dict:
    """Persist and return aggregate token use and estimated cost for one run."""
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost_usd(model, input_tokens, output_tokens),
    }

    COST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COST_FILE.open("a", encoding="utf-8") as cost_file:
        cost_file.write(json.dumps(entry) + "\n")

    return entry
