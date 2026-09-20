"""The analyst team for identification questions ("Which company did X while also Y?").

The Leader (planner) splits the question into clues. One analyst agent per company then grades every
clue against that company's OWN passages, all in parallel, and returns a verdict per clue with the
passage numbers it relied on. Analysts never see each other's companies, so they can't be steered by a
rival's evidence; the leader's job (plain code here) is to line the verdicts up into a matrix.

Members are derived from the corpus overview (entities found in headings), so a new dataset gets a new
team without code changes. If a dataset has no detectable entities, the caller falls back to the
ordinary retrieval flow."""
import asyncio
import re

from ..chunker import Chunk
from ..config import settings
from ..corpus import Corpus
from ..llm import chat_json, embed
from ..observability import agent_span
from .common import format_evidence

GRADE_SYSTEM = """You are the analyst agent for {entity} ({name}) on a research team. The team leader must identify which single company satisfies ALL of a set of clues. You only see passages about YOUR company; judge only that.

For each numbered clue, decide using ONLY the numbered passages:
- "supported": a passage states or clearly implies it for this company. Respect qualifiers: exact numbers, the period, and wording like "record", "fifth consecutive", "first time", "only".
- "contradicted": a passage shows it is false for this company (a different value or the opposite outcome).
- "no_evidence": the passages neither support nor contradict it.
Be strict: a clue with a specific number, period or event is "supported" only if a passage matches it. Never use outside knowledge.

Return JSON: {{"clues": [{{"id": <clue number>, "verdict": "supported"|"contradicted"|"no_evidence", "passages": [<passage numbers>], "note": "<= 20 words"}}]}}"""

SCORE = {"supported": 1, "contradicted": -2, "no_evidence": 0}
PER_CLUE = 3  # passages kept per clue per analyst
MAX_MEMBERS = 24
MAX_PERIOD_PASSAGES = 10


def members(corpus: Corpus, scope_docs: list[str]) -> list[dict]:
    """One analyst per entity found in the routed documents (all documents if routing is empty)."""
    out = []
    for d in corpus.overview["documents"]:
        if scope_docs and d["file"] not in scope_docs:
            continue
        for e in d["entities"]:
            out.append({"entity": e, "name": corpus.entity_name(e), "doc": d["file"]})
    return out


def short_period(label: str) -> str:
    """"Q2 FY2026 (Apr 30, 2026) — Full Summary" -> "Q2 FY2026"."""
    return re.split(r"\s+[—(–]|\s+-\s", label, maxsplit=1)[0].strip()


async def _grade(member: dict, clues: list[str], passages: list[Chunk], period: str = "", model: str | None = None) -> dict:
    entity, name = member["entity"], member["name"]
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clues, 1))
    scope = f" The passages come from the reporting period {period} (plus a few undated context passages)." if period else ""
    user = f"Clues:\n{numbered}\n\nPassages about {entity}:{scope}\n{format_evidence(passages, 1200)}"
    graded = await chat_json(GRADE_SYSTEM.format(entity=entity, name=name), user, model or settings.chat_model)
    by_id = {}
    for item in graded.get("clues", []):
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            by_id[item["id"]] = item
    verdicts = []
    for i, clue in enumerate(clues, 1):
        item = by_id.get(i, {})
        verdict = item.get("verdict") if item.get("verdict") in SCORE else "no_evidence"
        nums = [n for n in item.get("passages", []) if isinstance(n, int) and 1 <= n <= len(passages)]
        verdicts.append({"id": i, "clue": clue, "verdict": verdict, "passages": [passages[n - 1].id for n in nums], "note": str(item.get("note", ""))[:160]})
    return {**member, "period": period, "verdicts": verdicts, "score": sum(SCORE[v["verdict"]] for v in verdicts), "passages": passages}


async def analyze(corpus: Corpus, member: dict, clues: list[str]) -> dict:
    """Entity-level analyst: grades every clue against the best passages about one company, any period."""
    entity, name, doc = member["entity"], member["name"], member["doc"]
    with agent_span(f"analyst {entity}", entity=entity, clues=len(clues)):
        queries = [f"{entity} {name} {c}" for c in clues]
        qvecs = await embed(queries)
        per_clue = await asyncio.gather(
            *[asyncio.to_thread(corpus.store.search, q, v, PER_CLUE * 2, [entity], [doc]) for q, v in zip(queries, qvecs)]
        )
        passages: list[Chunk] = []
        for res in per_clue:
            for cid, _ in res[:PER_CLUE]:
                c = corpus.by_id.get(cid)
                if c and c not in passages:
                    passages.append(c)
        return await _grade(member, clues, passages)


async def refine(corpus: Corpus, results: list[dict], clues: list[str], top: int = 2) -> list[dict]:
    """Riddle clues normally describe ONE reporting period. For the leading candidates, run a period
    analyst per quarter that grades all clues against just that quarter's passages, so the team finds the
    (company, quarter) pair that fits instead of mixing evidence from different quarters."""
    tasks = []
    for r in ranked(results)[:top]:
        periods: dict[str, list[Chunk]] = {}
        for c in corpus.chunks:
            if c.entity == r["entity"] and c.period:
                periods.setdefault(short_period(c.period), []).append(c)
        shared = [c for c in r["passages"] if not c.period][:6]  # undated context: tables, deep dives, news
        for label, chunks in periods.items():
            passages = [c for c in chunks if c.kind != "table_row"][:MAX_PERIOD_PASSAGES] + shared
            tasks.append(_period_task(r, label, passages, clues))
    return list(await asyncio.gather(*tasks))


async def _period_task(member: dict, label: str, passages: list[Chunk], clues: list[str]) -> dict:
    with agent_span(f"period analyst {member['entity']} {label}", entity=member["entity"], period=label):
        return await _grade(member, clues, passages, label, settings.fast_model)  # many small calls: use the fast model


def best_per_entity(pairs: list[dict]) -> list[dict]:
    """Each entity represented by its best (entity, period) result, ranked."""
    best: dict[str, dict] = {}
    for r in pairs:
        key = (r["score"], sum(v["verdict"] == "supported" for v in r["verdicts"]))
        cur = best.get(r["entity"])
        if cur is None or key > (cur["score"], sum(v["verdict"] == "supported" for v in cur["verdicts"])):
            best[r["entity"]] = r
    return ranked(list(best.values()))


def ranked(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda r: (r["score"], sum(v["verdict"] == "supported" for v in r["verdicts"])), reverse=True)


def build_evidence(results: list[dict], cap: int = 24, top: int = 3) -> list[Chunk]:
    """Evidence for the verifier/writer: cited passages of the leading analysts first, in rank order."""
    evidence: list[Chunk] = []
    seen: set[str] = set()

    def add(c: Chunk | None):
        if c and c.id not in seen and len(evidence) < cap:
            seen.add(c.id)
            evidence.append(c)

    lead = ranked(results)[:top]
    for r in lead:
        cited = [pid for v in r["verdicts"] if v["verdict"] == "supported" for pid in v["passages"]]
        by_id = {c.id: c for c in r["passages"]}
        for pid in cited:
            add(by_id.get(pid))
    for r in lead:
        for c in r["passages"]:
            add(c)
    return evidence


def summarize(results: list[dict], evidence: list[Chunk], clues: list[str], top: int = 4) -> dict:
    """Leader's matrix in the evidence's numbering, so downstream agents can cite by [n]."""
    index = {c.id: n for n, c in enumerate(evidence, 1)}
    rows = []
    for r in ranked(results)[:top]:
        rows.append(
            {
                "entity": r["entity"],
                "name": r["name"],
                "period": r.get("period", ""),
                "score": r["score"],
                "verdicts": [
                    {"id": v["id"], "verdict": v["verdict"], "passages": [index[p] for p in v["passages"] if p in index], "note": v["note"]}
                    for v in r["verdicts"]
                ],
            }
        )
    full = [r for r in ranked(results) if all(v["verdict"] == "supported" for v in r["verdicts"])]
    return {
        "clues": clues,
        "candidates": rows,
        "members": len(results),
        "leader": rows[0]["entity"] if rows else None,
        "unique_full_match": full[0]["entity"] if len(full) == 1 else None,
        "full_matches": [r["entity"] for r in full],
        "leader_period": rows[0]["period"] if rows else "",
    }
