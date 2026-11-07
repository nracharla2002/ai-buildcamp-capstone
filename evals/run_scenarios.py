"""Run the Home Energy Helper against evals/scenarios.csv and save results."""

import asyncio
import csv
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from home_energy_agent.agent import run_agent
from home_energy_agent.rag import ROOT, build_index
from home_energy_agent.tools import capture_tool_calls


MODEL = "openai:gpt-4o-mini"
SCENARIOS_FILE = ROOT / "evals" / "scenarios.csv"
RESULTS_FILE = ROOT / "evals" / "results.json"


async def run_all():
    with SCENARIOS_FILE.open(encoding="utf-8", newline="") as csv_file:
        scenarios = list(csv.DictReader(csv_file))

    index = build_index()
    results = []

    for number, scenario in enumerate(scenarios, start=1):
        question = scenario["question"]
        print(f"[{number}/{len(scenarios)}] {question}")

        run_details = {}
        start = time.time()
        try:
            with capture_tool_calls() as tool_calls:
                answer = await run_agent(
                    question,
                    index=index,
                    model=MODEL,
                    run_details=run_details,
                )

            input_tokens = run_details.get("input_tokens", 0)
            output_tokens = run_details.get("output_tokens", 0)
            result = {
                "question": question,
                "category": scenario["category"],
                "type": scenario["type"],
                "output": answer,
                "tool_calls": tool_calls,
                "execution_time": round(time.time() - start, 2),
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "cost": run_details.get("estimated_cost_usd", 0.0),
                "session_id": run_details.get("session_id"),
            }
            print(f"  Done in {result['execution_time']:.1f}s (${result['cost']:.6f})")
        except Exception as exc:
            result = {
                "question": question,
                "category": scenario["category"],
                "type": scenario["type"],
                "error": str(exc),
                "execution_time": round(time.time() - start, 2),
            }
            print(f"  Failed after {result['execution_time']:.1f}s: {exc}")

        results.append(result)

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    total_cost = sum(result.get("cost", 0.0) for result in results)
    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")
    print(f"Total estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(run_all())
