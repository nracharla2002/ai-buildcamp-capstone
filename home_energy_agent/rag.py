"""Load, chunk, index, and search curated Energy.gov guidance."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from minsearch import Index
from openai import OpenAI
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "energygov_cache.json"
CACHE_MAX_AGE = timedelta(days=7)
CACHE_VERSION = 2

ENERGY_GOV_SOURCES = [
    {
        "title": "5 Tips for Energy-Saving Renters",
        "topic": "low-cost energy savings, heating, thermostat settings, air leaks, LED lighting",
        "url": "https://www.energy.gov/articles/5-tips-energy-saving-renters",
    },
    {
        "title": "Renters",
        "topic": "low-cost energy savings, renters, thermostats, drafts, water heating",
        "url": "https://www.energy.gov/save/renters",
    },
    {
        "title": "Whole-House Weatherization",
        "topic": "weatherization, heating, insulation, air sealing, energy audits",
        "url": "https://www.energy.gov/cmei/scep/wap/whole-house-weatherization",
    },
    {
        "title": "Consumer Water Heaters",
        "topic": "water heating, heat pump water heaters, efficiency",
        "url": "https://www.energy.gov/cmei/buildings/consumer-water-heaters",
    },
    {
        "title": "Consumer Guide to Air Sealing the Home",
        "topic": "weatherization, air leaks, drafts, caulking, weatherstripping",
        "url": "https://www.energy.gov/sites/default/files/2021-08/ES-Air%20Sealing_080421.pdf",
    },
]


class TextExtractor(HTMLParser):
    """Extract readable text from an HTML page without another dependency."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header", "noscript", "svg"}:
            self.skip_depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "section"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header", "noscript", "svg"}:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "section"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def extract_source_text(source, response_bytes):
    """Extract readable text from an Energy.gov HTML page or PDF."""
    if source["url"].lower().split("?")[0].endswith(".pdf"):
        reader = PdfReader(BytesIO(response_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    parser = TextExtractor()
    parser.feed(response_bytes.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def fetch_energy_gov_sources():
    """Fetch curated DOE pages, skipping individual pages that fail."""
    documents = []
    failures = []
    for source in ENERGY_GOV_SOURCES:
        try:
            request = Request(
                source["url"],
                headers={"User-Agent": "HomeEnergyHelper/0.1 (educational project)"},
            )
            with urlopen(request, timeout=25) as response:
                text = extract_source_text(source, response.read())
            if len(text.split()) < 50:
                raise ValueError("page contained too little readable text")
            documents.append(
                {**source, "source_name": "U.S. Department of Energy", "text": text}
            )
            print(f"Fetched: {source['title']}")
        except Exception as exc:
            failures.append(f"{source['title']}: {exc}")
            print(f"Could not fetch {source['title']}: {exc}")
    return documents, failures


def chunk_documents(documents, chunk_words=450, overlap_words=75):
    """Chunk page text and keep the source title and URL with every passage."""
    chunks = []
    step = chunk_words - overlap_words
    for document in documents:
        words = document["text"].split()
        for start in range(0, len(words), step):
            content = " ".join(words[start : start + chunk_words]).strip()
            if content:
                chunks.append(
                    {
                        "title": document["title"],
                        "topic": document["topic"],
                        "content": content,
                        "source_name": document["source_name"],
                        "source_url": document["url"],
                    }
                )
    return chunks


def load_energy_chunks(force_refresh=False):
    """Use a fresh local cache or fetch and cache the Energy.gov pages."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if (
                not force_refresh
                and cached.get("version") == CACHE_VERSION
                and datetime.now(timezone.utc) - fetched_at < CACHE_MAX_AGE
            ):
                print(f"Using cached Energy.gov text fetched {fetched_at.date()}.")
                return cached["chunks"]
        except (OSError, ValueError, KeyError, TypeError):
            cached = None

    documents, failures = fetch_energy_gov_sources()
    if documents:
        chunks = chunk_documents(documents)
        cache = {
            "version": CACHE_VERSION,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "chunks": chunks,
        }
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        if failures:
            print(f"Fetched {len(documents)} source(s); {len(failures)} source(s) failed.")
        print(f"Created {len(chunks)} chunks in {CACHE_PATH}.")
        return chunks

    if cached and cached.get("version") == CACHE_VERSION and cached.get("chunks"):
        print("All refreshes failed; using the existing cached Energy.gov content.")
        return cached["chunks"]
    raise RuntimeError(
        "No Energy.gov pages could be fetched and no current cache exists. "
        "Check your internet connection, then rerun this cell."
    )


def build_index(chunks=None):
    """Index Energy.gov passages using the same fields as the project notebook."""
    if chunks is None:
        chunks = load_energy_chunks()
    index = Index(
        text_fields=["title", "topic", "content"],
        keyword_fields=["source_name", "source_url"],
    )
    index.fit(chunks)
    return index


def search(query, index, num_results=5):
    """Search the supplied Energy.gov index."""
    return index.search(query, num_results=num_results)


rag_instructions = """
You are a careful Home Energy Helper. Answer only questions about household energy use,
electricity bills, and practical energy-saving guidance. For guidance, use only the
supplied Energy.gov excerpts; do not answer from general model knowledge. Cite factual
recommendations with Markdown links using the exact title and source URL from an excerpt.
If the excerpts do not support an answer, say so and ask one focused follow-up question.
Do not guess the cause of a bill change: usage records show what changed, not why.
Distinguish calculations from explanations, and never invent source links or savings claims.
""".strip()


def build_prompt(query, search_results):
    return json.dumps(
        {"question": query, "energy_gov_context": search_results},
        indent=2,
    )


def llm(user_prompt, instructions=None, model="gpt-4o-mini"):
    """Call OpenAI for the standalone RAG baseline."""
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "notebooks" / ".env", override=False)
    messages = []
    if instructions is not None:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_prompt})
    response = OpenAI().responses.create(model=model, input=messages)
    return response.output_text


def rag(query, index, model="gpt-4o-mini"):
    """Run the simple retrieve-prompt-answer baseline from the notebook."""
    search_results = search(query, index)
    if not search_results:
        return "I couldn't find relevant Energy.gov guidance for that question. Could you rephrase it around a home energy topic?"
    prompt = build_prompt(query, search_results)
    return llm(prompt, instructions=rag_instructions, model=model)
