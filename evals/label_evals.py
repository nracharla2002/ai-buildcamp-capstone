"""Streamlit UI for labeling Home Energy Helper evaluation results."""

import csv
import json
import sys
from pathlib import Path

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from home_energy_agent.agent import agent_instructions


RESULTS_FILE = REPO_ROOT / "evals" / "results.json"
LEGACY_LABELED_FILE = REPO_ROOT / "evals" / "results_labeled.json"


def load_data(file_path):
    with file_path.open(encoding="utf-8") as file:
        if file_path.suffix.lower() != ".csv":
            return json.load(file)

        data = list(csv.DictReader(file))

    for item in data:
        for field, default in (("tool_calls", []), ("tokens", {})):
            value = item.get(field)
            item[field] = json.loads(value) if value else default

        for field in ("execution_time", "cost"):
            value = item.get(field)
            item[field] = float(value) if value else 0.0

        if not item.get("label"):
            item["label"] = None

    return data


def save_data(data, file_path):
    fieldnames = list(dict.fromkeys(key for item in data for key in item))
    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else "" if value is None else value
                for key, value in item.items()
            }
            writer.writerow(row)


def get_labeled_path(input_path):
    return input_path.with_name(f"{input_path.stem}_labeled.csv")


def find_next_unlabelled(data, start_index):
    for offset in range(1, len(data)):
        index = (start_index + offset) % len(data)
        if data[index].get("label") is None:
            return index
    return None


st.set_page_config(layout="wide", page_title="Home Energy Evaluation Labeling")
st.title("Home Energy Helper Evaluation Labeling")

if not RESULTS_FILE.exists():
    st.error("No evals/results.json found. Run run_scenarios.py first.")
    st.stop()

labeled_file = get_labeled_path(RESULTS_FILE)
if st.session_state.get("results_file") != str(RESULTS_FILE):
    st.session_state.results_file = str(RESULTS_FILE)
    if labeled_file.exists():
        source_file = labeled_file
    elif LEGACY_LABELED_FILE.exists():
        source_file = LEGACY_LABELED_FILE
    else:
        source_file = RESULTS_FILE
    st.session_state.data = load_data(source_file)
    for item in st.session_state.data:
        item.setdefault("label", None)
        item.setdefault("comments", "")
    st.session_state.current_index = next(
        (i for i, item in enumerate(st.session_state.data) if item["label"] is None),
        0,
    )

data = st.session_state.data
if not data:
    st.error("No evaluation results found in evals/results.json.")
    st.stop()

st.sidebar.header("Scenario navigation")
selection = st.sidebar.selectbox(
    "Select scenario",
    range(len(data)),
    index=st.session_state.current_index,
    format_func=lambda i: (
        f"{'✅' if data[i].get('label') == 'good' else '❌' if data[i].get('label') == 'bad' else '⬜'} "
        f"{i + 1}: {data[i].get('question', 'No question')[:55]}"
    ),
)

if selection != st.session_state.current_index:
    st.session_state.current_index = selection
    st.rerun()

labeled_count = sum(item.get("label") is not None for item in data)
st.sidebar.progress(labeled_count / len(data))
st.sidebar.write(f"Labeled: {labeled_count} / {len(data)}")

item = data[st.session_state.current_index]
st.header(f"Scenario {st.session_state.current_index + 1} of {len(data)}")

st.subheader("User question")
st.chat_message("user").write(item.get("question", "No question found."))

st.subheader("Agent response")
if item.get("error"):
    st.error(item["error"])
else:
    st.chat_message("assistant").markdown(item.get("output", "No response found."))

meta1, meta2, meta3, meta4 = st.columns(4)
meta1.metric("Category", item.get("category", "N/A"))
meta2.metric("Scenario type", item.get("type", "N/A"))
meta3.metric("Execution time", f"{item.get('execution_time', 0)} s")
meta4.metric("Estimated cost", f"${item.get('cost', 0):.6f}")

tokens = item.get("tokens", {})
if tokens:
    st.caption(
        f"Tokens — input: {tokens.get('input_tokens', 0)}, "
        f"output: {tokens.get('output_tokens', 0)}, "
        f"total: {tokens.get('total_tokens', 0)}"
    )

tool_calls = item.get("tool_calls", [])
st.write("**Tools called:** " + (", ".join(tool_calls) if tool_calls else "None recorded"))

with st.expander("View agent instructions"):
    st.markdown(agent_instructions)

st.divider()
st.subheader("Label this response")
label_col, comments_col = st.columns([1, 2])

with label_col:
    choices = ["Not Selected", "Good", "Not Good"]
    choice_for_label = {"good": "Good", "bad": "Not Good"}
    current_choice = choice_for_label.get(item.get("label"), "Not Selected")
    decision = st.radio(
        "Does the agent response meet expectations?",
        choices,
        index=choices.index(current_choice),
        key=f"label_{st.session_state.current_index}",
    )

with comments_col:
    comments = st.text_area(
        "Comments",
        value=item.get("comments", ""),
        height=100,
        key=f"comments_{st.session_state.current_index}",
    )

if st.button("Save label and next", type="primary"):
    item["label"] = {"Good": "good", "Not Good": "bad"}.get(decision)
    item["comments"] = comments
    save_data(data, labeled_file)

    next_index = find_next_unlabelled(data, st.session_state.current_index)
    if next_index is not None:
        st.session_state.current_index = next_index
    st.rerun()

st.sidebar.divider()
if st.sidebar.button("Reset labels"):
    if labeled_file.exists():
        labeled_file.unlink()
    st.session_state.pop("data", None)
    st.session_state.pop("current_index", None)
    st.session_state.pop("results_file", None)
    st.rerun()
