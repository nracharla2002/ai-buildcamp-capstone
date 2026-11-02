"""Simple Streamlit chat interface for the Home Energy Helper."""

import asyncio

import logfire
import streamlit as st
from dotenv import load_dotenv

from home_energy_agent.agent import run_agent
from home_energy_agent.monitoring import configure_monitoring
from home_energy_agent.rag import ROOT, build_index
from home_energy_agent.tools import capture_tool_calls


load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "notebooks" / ".env", override=False)
configure_monitoring()

st.set_page_config(page_title="Home Energy Helper", page_icon="⚡")
st.title("⚡ Home Energy Helper")
st.caption("Compare electricity bills and get Energy.gov-backed guidance.")


@st.cache_resource(show_spinner="Loading Energy.gov guidance…")
def load_index():
    return build_index()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


with st.sidebar:
    st.subheader("About")
    st.caption("Each interaction is traced with Logfire and records token usage.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Interaction details"):
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    st.markdown("**Tools called**")
                    for tool_name in tool_calls:
                        st.write(f"- `{tool_name}`")
                else:
                    st.write("No tools were called.")

                details = message.get("run_details", {})
                if details.get("session_id"):
                    st.caption(f"Trace session: `{details['session_id']}`")
                    st.caption(
                        f"Tokens: {details.get('input_tokens', 0)} input / "
                        f"{details.get('output_tokens', 0)} output · "
                        f"Estimated cost: "
                        f"${details.get('estimated_cost_usd', 0):.6f}"
                    )


if prompt := st.chat_input("Ask about your electricity use or home energy savings…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        with st.spinner("Checking your question…"):
            tool_calls = []
            run_details = {}
            try:
                with logfire.span(
                    "home_energy.streamlit_interaction",
                    user_question=prompt,
                ):
                    index = load_index()
                    with capture_tool_calls() as captured_tool_calls:
                        tool_calls = captured_tool_calls
                        answer = asyncio.run(
                            run_agent(
                                prompt,
                                index=index,
                                message_history=st.session_state.agent_history,
                                run_details=run_details,
                            )
                        )
                answer_placeholder.markdown(answer)
                st.session_state.agent_history = run_details.get(
                    "message_history", st.session_state.agent_history
                )
            except Exception as exc:
                answer = f"I couldn’t complete that request. Details: {exc}"
                answer_placeholder.error(answer)

        with st.expander("Interaction details"):
            if tool_calls:
                st.markdown("**Tools called**")
                for tool_name in tool_calls:
                    st.write(f"- `{tool_name}`")
            else:
                st.write("No tools were called.")

            if run_details.get("session_id"):
                st.caption(f"Trace session: `{run_details['session_id']}`")
                st.caption(
                    f"Tokens: {run_details.get('input_tokens', 0)} input / "
                    f"{run_details.get('output_tokens', 0)} output · "
                    f"Estimated cost: "
                    f"${run_details.get('estimated_cost_usd', 0):.6f}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tool_calls": list(tool_calls),
            "run_details": run_details,
        }
    )
