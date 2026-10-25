import pytest

from home_energy_agent.tests.utils import collect_tools, run_agent_test


@pytest.mark.asyncio
async def test_agent_compares_bills_from_zero_baseline(agent):
    user_prompt = (
        "Compare 0 kWh and $0 in January with 100 kWh and $20 in February; "
        "both bills covered 30 days."
    )
    result = await run_agent_test(agent, user_prompt)

    tool_calls = collect_tools(result.new_messages())
    tool_names = [tool_call.name for tool_call in tool_calls]

    assert "compare_monthly_bills" in tool_names

    comparison_call = next(
        tool_call for tool_call in tool_calls
        if tool_call.name == "compare_monthly_bills"
    )
    assert comparison_call.args["previous_period"]["kwh"] == 0
    assert comparison_call.args["current_period"]["kwh"] == 100

    answer = result.output
    assert "100" in answer
    assert "$20" in answer
