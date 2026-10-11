# Home Energy Helper

A Streamlit home energy helper that helps households understand how their electricity use and costs change over time, and find practical ways to save energy.

## The Problem

People can see the amount of electricity they used and the price they paid, but those numbers alone don’t explain why a bill changed or what to do next. The Home Energy Helper makes comparisons clear, offers evidence-based energy-saving guidance, and asks for missing details instead of guessing.

## What It Does

Users enter monthly electricity usage in kWh and the bill amount. The app calculates and compares usage and costs across months, highlighting changes over time. Its assistant answers questions about common ways to reduce energy use using knowledge retrieved from [energy.gov](https://www.energy.gov/), and includes links to reliable sources with every guidance-based response. When important details are missing, it asks follow-up questions so its explanations stay accurate and clear.

For example, if a user says, “My electricity use went from 620 kWh in January to 790 kWh in February. Why did it go up, and what should I check?”, the assistant can reply: “Your usage increased by 170 kWh (about 27%). The bills show the change, but they don’t reveal its cause. Was February’s billing period longer, or did you use more heating or a space heater?”

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.
