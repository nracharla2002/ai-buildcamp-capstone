from dataclasses import dataclass
import json


@dataclass
class ToolCall:
    name: str
    args: dict


def collect_tools(messages):
    tool_calls = []

    for message in messages:
        for part in message.parts:
            if part.part_kind != "tool-call":
                continue
            if part.tool_name == "final_result":
                continue

            args = part.args
            if isinstance(args, str):
                args = json.loads(args)

            tool_calls.append(ToolCall(part.tool_name, args))

    return tool_calls


async def run_agent_test(agent, user_prompt, message_history=None):
    if message_history is None:
        message_history = []

    return await agent.run(user_prompt, message_history=message_history)
