import pytest

from home_energy_agent.tests.judge import assert_criteria
from home_energy_agent.tests.utils import run_agent_test


@pytest.mark.asyncio
async def test_zero_baseline_bill_comparison_is_accurate(agent):
    user_prompt = (
        "Compare 0 kWh and $0 in January with 100 kWh and $20 in February; "
        "both bills covered 30 days."
    )
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(
        result,
        [
            "Calls compare_monthly_bills with the supplied bill data; does not "
            "call search_energy_guidance.",
            "Reports January-to-February increases of 100 kWh and $20.",
            "Reports daily averages: January 0 kWh/$0; February about "
            "3.33 kWh/$0.67.",
            "Does not calculate a percentage increase from the zero baseline.",
        ],
    )
