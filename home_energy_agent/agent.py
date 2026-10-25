"""Home Energy Helper agent, following the flow in energy_saver_agent.ipynb."""

import json
import re

from dotenv import load_dotenv
from pydantic_ai import Agent

from home_energy_agent.rag import ROOT, build_index
from home_energy_agent.tools import HomeEnergyTools


agent_instructions = """
You are the Home Energy Helper. Help users understand household electricity bills
and find practical ways to save energy.

Scope and grounding:
- Answer only about home energy use, electricity bills, or energy-saving guidance.
  If the request is outside this scope, politely say so and do not answer it.
- The application supplies relevant Energy.gov passages with each request. Treat
  them as the only source of factual energy guidance. Do not fill gaps with general
  model knowledge.
- Cite guidance claims with Markdown links using the exact source title and URL
  supplied in the passages. Never invent URLs, statistics, or savings estimates.
- If the passages do not support the requested guidance, say so and ask one
  focused follow-up question.

Tool use and bill comparisons:
- For month-over-month comparisons, call compare_monthly_bills only when the user
  supplied both periods' kWh and bill amounts. Preserve month labels and values;
  pass null for unknown billing days. Ask for required missing values.
- State comparison results as observations, never as causes. The calculation
  cannot prove that weather, heating, billing days, rates, or an appliance caused
  the change.
- The application already searched the Energy.gov corpus for this turn. Call
  search_energy_guidance only if a needed topic is absent from the supplied excerpts.
- For a bill increase question, report the comparison first, give up to three
  source-supported checks, and end with one focused question about missing details.
- Do not claim which heating option is cheapest or calculate appliance payback.
- Keep the answer concise and practical.
""".strip()


SCOPE_TERMS = {
    "energy", "electricity", "electric", "power", "kwh", "bill", "bills", "utility",
    "usage", "consumption", "heating", "heat", "heater", "cooling", "hvac",
    "thermostat", "appliance", "appliances", "water", "insulation", "draft", "drafts",
    "weatherization", "weatherstripping", "lighting", "led", "refrigerator", "fridge",
    "dryer", "washer", "home", "house", "household", "apartment", "efficient",
    "efficiency", "kilowatt", "kilowatts", "solar", "conditioner", "ac",
}


def is_home_energy_question(question: str) -> bool:
    """Return whether the question mentions a home-energy topic."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    return bool(words & SCOPE_TERMS)


def create_agent(home_energy_tools: HomeEnergyTools, model="openai:gpt-4o-mini"):
    """Create an agent and register the two HomeEnergyTools methods."""
    return Agent(
        model,
        name="home_energy_helper",
        instructions=agent_instructions,
        tools=[
            home_energy_tools.search_energy_guidance,
            home_energy_tools.compare_monthly_bills,
        ],
    )


async def run_agent(user_question: str, index=None, model="openai:gpt-4o-mini") -> str:
    """Check scope, retrieve DOE evidence, and run the tool-using agent."""
    if not is_home_energy_question(user_question):
        return (
            "I’m the Home Energy Helper and can answer questions about household energy use, "
            "electricity bills, and ways to save energy. Is your question about one of those?"
        )

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "notebooks" / ".env", override=False)
    active_index = index if index is not None else build_index()
    home_energy_tools = HomeEnergyTools(active_index)

    # Retrieve first so guidance answers are grounded even if the model skips the search tool.
    guidance = home_energy_tools.search_energy_guidance(user_question)
    has_bill_data = bool(re.search(r"\b(kwh|bill|electricity|usage)\b", user_question, re.I))
    asks_for_guidance = bool(
        re.search(
            r"\b(why|cause|check|reduce|save|lower|efficient|guidance|advice|tip|tips|heat|heating|cooling|thermostat|appliance)\b",
            user_question,
            re.I,
        )
    )

    if asks_for_guidance and not guidance:
        return (
            "I couldn't find relevant Energy.gov passages to support that guidance. "
            "Could you share a more specific topic, such as heating, water heating, or air leaks?"
        )
    if not guidance and not has_bill_data:
        return "I couldn't find relevant Energy.gov guidance for that question. Could you rephrase it around a home energy topic?"

    grounded_question = f"""
USER QUESTION:
{user_question}

RETRIEVED ENERGY.GOV PASSAGES (use only these for factual energy guidance):
{json.dumps(guidance, indent=2)}
""".strip()

    home_energy_agent = create_agent(home_energy_tools, model=model)
    result = await home_energy_agent.run(grounded_question)
    return result.output
