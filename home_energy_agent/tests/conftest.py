import pytest
from dotenv import load_dotenv

from home_energy_agent.agent import create_agent
from home_energy_agent.rag import ROOT, build_index
from home_energy_agent.tools import HomeEnergyTools


load_dotenv(ROOT / ".env")


@pytest.fixture(scope="module")
def agent():
    index = build_index()
    home_energy_tools = HomeEnergyTools(index)
    return create_agent(home_energy_tools)
