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
            "The agent calls compare_monthly_bills with January as the previous "
            "period (0 kWh, $0, 30 days) and February as the current period "
            "(100 kWh, $20, 30 days).",
            "The answer clearly reports the increase of 100 kWh and $20, and "
            "does not state a percentage increase because the January baseline "
            "is zero.",
            "The answer treats the comparison as a change in the bills and does "
            "not claim to know what caused the increase.",
        ],
    )
