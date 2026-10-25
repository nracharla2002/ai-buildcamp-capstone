from pydantic import BaseModel, Field
from pydantic_ai import Agent

from home_energy_agent.tests.utils import collect_tools


judge_instructions = """
You evaluate whether a Home Energy Helper agent followed the exact criteria.
Use only the agent's final answer and recorded tool calls as evidence. Do not
give credit for information that is not present in that evidence.
""".strip()


class JudgeCriterion(BaseModel):
    criterion_description: str = Field(
        description="The exact behavior being evaluated."
    )
    passed: bool = Field(
        description="Whether the agent met this criterion."
    )
    judgement: str = Field(
        description="Evidence from the agent's answer or tool calls for the decision."
    )


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion]
    feedback: str


def create_judge_agent():
    return Agent(
        model="openai:gpt-4o-mini",
        name="home_energy_test_judge",
        instructions=judge_instructions,
        output_type=JudgeFeedback,
    )


judge_user_prompt_template = """
Evaluate the agent against each criterion below.

CRITERIA:
{criteria}

AGENT FINAL ANSWER:
{output}

TOOL CALLS:
{tool_calls}
""".strip()


async def assert_criteria(result, criteria):
    tool_calls = collect_tools(result.new_messages())
    judge_prompt = judge_user_prompt_template.format(
        criteria="\n".join(f"- {criterion}" for criterion in criteria),
        output=str(result.output),
        tool_calls="\n".join(str(tool_call) for tool_call in tool_calls),
    )

    judge_result = await create_judge_agent().run(judge_prompt)

    print("Judge feedback:")
    print(judge_result.output.feedback)
    for criterion in judge_result.output.criteria:
        print(f"{criterion.criterion_description}: {criterion.judgement}")
        assert criterion.passed, (
            f"{criterion.criterion_description}: {criterion.judgement}"
        )
