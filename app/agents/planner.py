import json

from ..config import settings
from ..llm import chat_json

SYSTEM = """You are the Planner agent in a retrieval system that answers precise questions over a corpus of financial research documents.
You are given the question and an overview of the corpus (files, sections, and entities such as tickers).
Break the question into at most 5 focused sub-queries for a hybrid (keyword + semantic) search engine.

Rules:
- Each sub-query must be self-contained: name the company/ticker, the metric, and the period, using the document's own vocabulary (e.g. "CET1 ratio", "Q3 FY2026").
- For comparisons or rankings across companies, issue one sub-query per company, or one aimed at the sector snapshot/screening table.
- If the period is ambiguous ("latest", "last quarter"), search for the most recent period AND the ones adjacent to it.
- When the question asks for a specific number, add a cross-check sub-query aimed at a different section (e.g. the snapshot table vs. the company deep dive vs. the quarterly history) so disagreements between sources surface.
- "entities" must be values copied exactly from the corpus overview's entity lists, or an empty list if the question is sector-wide or the entity is unclear.
- Do not answer the question.

Also classify the question:
- "identify": it asks WHICH entity (company/bank/miner...) satisfies a set of descriptive conditions ("Which bank did X while also Y?", "Which company ...?"). For these, list the atomic, independently checkable conditions in "clues" (2-6 short self-contained statements, each preserving the specifics: numbers, periods, events, qualifiers), and put in "scope_docs" the file names from the overview whose subject matter fits (e.g. the sector the question is about); use [] if unclear.
- "lookup": anything else (a figure, a comparison, a summary). Use "clues": [] and "scope_docs": [].

Return JSON: {"question_type": "identify"|"lookup", "clues": [str], "scope_docs": [str], "sub_queries": [{"query": str, "entities": [str], "purpose": str}], "interpretation": str}"""


async def plan(question: str, overview: dict) -> dict:
    user = f"Corpus overview:\n{json.dumps(overview, ensure_ascii=False)}\n\nQuestion: {question}"
    out = await chat_json(SYSTEM, user, settings.chat_model)
    subs = [s for s in out.get("sub_queries", []) if isinstance(s, dict) and s.get("query")][:5]
    if not subs:
        subs = [{"query": question, "entities": [], "purpose": "direct search"}]
    valid = {e for d in overview.get("documents", []) for e in d.get("entities", [])}
    for s in subs:
        s["entities"] = [e for e in s.get("entities", []) if e in valid]
    files = {d["file"] for d in overview.get("documents", [])}
    clues = [c.strip() for c in out.get("clues", []) if isinstance(c, str) and c.strip()][:6]
    qtype = "identify" if out.get("question_type") == "identify" and len(clues) >= 2 else "lookup"
    scope = [f for f in out.get("scope_docs", []) if f in files]
    return {
        "question_type": qtype,
        "clues": clues if qtype == "identify" else [],
        "scope_docs": scope if qtype == "identify" else [],
        "sub_queries": subs,
        "interpretation": out.get("interpretation", ""),
    }
