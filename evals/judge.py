"""Use an LLM judge to label Home Energy Helper scenario results."""

import json
import re
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INPUT_FILE = REPO_ROOT / "evals" / "results.json"
OUTPUT_FILE = REPO_ROOT / "evals" / "results_judged.json"
JUDGE_MODEL = "openai:gpt-4o-mini"

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "notebooks" / ".env", override=False)


class JudgeEvaluation(BaseModel):
    reasoning: str
    label: Literal["good", "bad"]


judge_instructions = """
You are an expert evaluator assessing a Home Energy Helper assistant that answers
questions about household electricity bills and energy-saving guidance.

Evaluate each interaction using the user's question, recorded tool calls, and the
agent's final response. Reason through the criteria internally, then provide a
concise explanation tied to evidence in the interaction.

Label the response "good" when it accurately and completely answers the question,
or appropriately explains that required information is missing and asks for it. The
agent should use the available tools appropriately and present its answer in clear,
readable Markdown.

Label the response "bad" when it is factually wrong, misses the request, invents
facts or source links, fails to use required tool results, uses an inappropriate
tool, exposes raw tool output, or is poorly formatted.

Home Energy Helper behavior to apply:
- For bill comparisons, compare_monthly_bills should be called when both periods'
  kWh and bill amounts are provided. If required values are missing, the agent
  should ask for them instead of inventing them.
- Preserve the period labels the user supplied. If two periods have the same month
  label, distinguish them as the first and second bill. Never invent years or dates.
- Bill comparisons show what changed, not what caused it. The agent may suggest
  checks as possibilities when relevant, but must not present a cause as established
  by the bill figures.
- If the user explicitly says billing days are unknown, or asks for a daily or
  normalized comparison, the agent should explain that daily usage cannot be
  compared without the day counts and ask for them. Do not require this caveat for
  raw total changes when the user does not raise billing-day concerns.
- Before applying a billing-day penalty, verify whether the user supplied the day
  counts. Apply the missing-days rule only when the counts are absent and the user
  says they are unknown or asks for a daily/normalized comparison.
- Billing-day calibration examples:
  * Missing days explicitly raised: If the user says, "Compare January at 500 kWh
    and $100 with February at 600 kWh and $120. I don't know how many days each
    bill covered," an answer that reports only raw changes without explaining that
    daily usage cannot be compared or asking for the day counts is bad. Correct
    raw arithmetic does not make it complete.
  * Billing days provided: If the user gives January's figures over 31 days and
    February's over 28 days, do not claim the day counts are missing or penalize
    the answer for omitting a missing-days caveat. Judge the calculations and
    guidance using the information actually supplied.
- If the user asks why a bill is higher or "more than usual" but provides only the
  current bill, the agent must say it cannot compare without a previous bill and ask
  for that bill's kWh and amount. Mark it bad if it speculates about causes before
  asking.
- If the user asserts that an appliance caused a change and asks the agent to
  confirm, the agent must directly say that bill totals do not prove that cause.
  Comparing bills and asking about the appliance is not enough if the answer never
  addresses the claim.
- When the agent gives substantive energy-saving guidance, it should include at
  least one relevant Energy.gov link. Do not require a separate citation for every
  tip. Checking utility rates or fees is a reasonable diagnostic suggestion and
  does not require an Energy.gov citation. A brief optional tip in an otherwise
  correct bill comparison is not by itself a reason to mark the response bad.
- The agent should not invent savings estimates or make unsupported cost
  comparisons. If current information, such as a rebate amount, cannot be verified,
  it is good to say so and direct the user to an authoritative source; never expect
  the agent to guess an amount.
- Judge the whole answer for material failures. Correct arithmetic or useful
  follow-up questions do not excuse invented dates, ignoring explicit missing
  information, or failing to address an unsupported causal claim.
- The agent should politely decline questions outside household energy use, bills,
  and energy-saving guidance.
- A recorded search_energy_guidance call may be part of the application's initial
  retrieval, including on bill-comparison questions. Do not mark a response bad
  solely because that search was recorded; assess whether the needed tool behavior
  and final answer were appropriate.

The results file does not include the full retrieved Energy.gov passages. Do not
pretend to have verified a specific claim against passages that are not provided.
Use the available interaction evidence, and mention this limitation in the
reasoning if it materially affects the decision.
""".strip()


judge_agent = Agent(
    JUDGE_MODEL,
    output_type=JudgeEvaluation,
    instructions=judge_instructions,
)


def build_judge_prompt(row):
    response = row.get("output")
    if not response and row.get("error"):
        response = f"The agent failed to answer: {row['error']}"
    elif not response:
        response = "No final response was recorded."

    return f"""User question:
{row.get('question', 'No question was recorded.')}

Tools used:
{json.dumps(row.get('tool_calls', []), ensure_ascii=False)}

Agent final response:
{response}
""".strip()


def missing_billing_days_requirement_met(response):
    response = response.lower()
    mentions_limitation = any(
        re.search(pattern, response)
        for pattern in (
            r"(?:cannot|can't|can not|unable to|not possible to).{0,60}"
            r"(?:compare|calculate|normalize|adjust).{0,60}"
            r"(?:daily|per.day|day count|billing day|billing period)",
            r"(?:daily|per.day|normalized).{0,60}"
            r"(?:cannot|can't|can not|unable|not possible|not adjusted|not normalized)",
            r"(?:without|until|unless).{0,50}"
            r"(?:day count|billing day|number of days).{0,50}"
            r"(?:cannot|can't|can not|not possible|compare|calculate|adjust)",
        )
    )
    asks_for_day_counts = re.search(
        r"\b(?:how many|how long|what (?:is|are|were) the (?:number|day count)|"
        r"please provide|could you (?:provide|share|tell me)|"
        r"can you (?:provide|share|tell me))\b"
        r"[^?]{0,100}\b(?:days?|billing period)\b[^?]*\?",
        response,
    )
    return mentions_limitation and asks_for_day_counts is not None


def run_all():
    with INPUT_FILE.open(encoding="utf-8") as file:
        results = json.load(file)

    for number, row in enumerate(results, start=1):
        evaluation = judge_agent.run_sync(build_judge_prompt(row)).output
        label = evaluation.label
        reasoning = evaluation.reasoning

        if (
            row.get("category") == "missing_billing_days"
            and not missing_billing_days_requirement_met(row.get("output", ""))
        ):
            label = "bad"
            reasoning = (
                "Category-specific check failed: the response must explain that "
                "daily or normalized usage cannot be compared without billing-day "
                "counts and ask the user for those counts. "
                f"LLM judge reasoning: {evaluation.reasoning}"
            )

        row["judge_label"] = label
        row["judge_reasoning"] = reasoning
        print(f"[{number}/{len(results)}] {label}: {row.get('question', '')}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved judged results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_all()
