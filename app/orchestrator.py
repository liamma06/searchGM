"""Deterministic pipeline. Two flows share the same verify -> write tail:

  lookup    plan -> parallel retrieve -> verify (-> one follow-up round) -> write
  identify  plan -> analyst team (one agent per company, in parallel) -> cross-model verify -> write

The LLM agents do the reasoning; plain Python decides control flow, which keeps latency and failure
modes predictable. Every step is emitted as an event so the UI can show a live trace and so each
answer is auditable."""
import asyncio
import time
from typing import AsyncIterator

from .agents import planner, team, verifier, writer
from .agents.common import short_section
from .agents.retriever import retrieve
from .chunker import Chunk
from .config import settings
from .corpus import Corpus
from .observability import agent_span, log_pipeline

MAX_EVIDENCE = {"answer": 22, "brief": 30}


def hit_dict(c: Chunk, score: float) -> dict:
    return {
        "id": c.id,
        "doc": c.doc,
        "section": short_section(c),
        "entity": c.entity,
        "kind": c.kind,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "score": round(score, 4),
        "snippet": c.text[:220],
    }


def merge_evidence(per_query: list[list[tuple[Chunk, float]]], existing: list[Chunk], cap: int) -> list[Chunk]:
    """Round-robin across sub-queries so every sub-question is represented, deduped by chunk id."""
    out = list(existing)
    seen = {c.id for c in out}
    depth = max((len(r) for r in per_query), default=0)
    for rank in range(depth):
        for results in per_query:
            if rank < len(results) and len(out) < cap:
                c = results[rank][0]
                if c.id not in seen:
                    seen.add(c.id)
                    out.append(c)
    return out


async def lookup_flow(corpus: Corpus, question: str, plan: dict, cap: int, st: dict, ms, followups: bool = True, clues: list[str] | None = None):
    yield {"type": "stage", "agent": "retriever", "status": "running", "t": ms()}
    with agent_span("retriever", sub_queries=len(plan["sub_queries"]), backend=corpus.backend):
        results = await asyncio.gather(*[retrieve(corpus, s) for s in plan["sub_queries"]])
    for sub, res in zip(plan["sub_queries"], results):
        yield {"type": "retrieval", "query": sub["query"], "hits": [hit_dict(c, s) for c, s in res], "t": ms()}
    evidence = merge_evidence(list(results), [], cap)

    yield {"type": "stage", "agent": "verifier", "status": "running", "t": ms()}
    with agent_span("verifier", evidence=len(evidence), round=1):
        audit = await verifier.verify(question, evidence, clues=clues)
    yield {"type": "verification", "round": 1, "audit": audit, "t": ms()}

    follow_ups = [f for f in audit["follow_up_queries"] if isinstance(f, dict) and f.get("query")][:3]
    if followups and (not audit["answerable"] or audit["gaps"]) and follow_ups:
        valid = {e for d in corpus.overview["documents"] for e in d["entities"]}
        for f in follow_ups:
            f["entities"] = [e for e in f.get("entities", []) if e in valid]
        yield {"type": "stage", "agent": "retriever", "status": "follow-up", "t": ms()}
        with agent_span("retriever", sub_queries=len(follow_ups), followup=True):
            more = await asyncio.gather(*[retrieve(corpus, f) for f in follow_ups])
        for f, res in zip(follow_ups, more):
            yield {"type": "retrieval", "query": f["query"], "followup": True, "hits": [hit_dict(c, s) for c, s in res], "t": ms()}
        evidence = merge_evidence(list(more), evidence, cap + 6)
        yield {"type": "stage", "agent": "verifier", "status": "running", "t": ms()}
        with agent_span("verifier", evidence=len(evidence), round=2):
            audit = await verifier.verify(question, evidence)
        yield {"type": "verification", "round": 2, "audit": audit, "t": ms()}
    st.update(evidence=evidence, audit=audit)


async def team_flow(corpus: Corpus, question: str, plan: dict, roster: list[dict], cap: int, st: dict, ms):
    clues = plan["clues"]
    yield {"type": "stage", "agent": "analysts", "status": "running", "t": ms(), "members": len(roster)}
    tasks = [asyncio.create_task(team.analyze(corpus, m, clues)) for m in roster]
    results = []
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        yield {
            "type": "analyst",
            "entity": r["entity"],
            "name": r["name"],
            "score": r["score"],
            "verdicts": [{"id": v["id"], "verdict": v["verdict"], "note": v["note"], "passages": v["passages"]} for v in r["verdicts"]],
            "t": ms(),
        }
    yield {"type": "stage", "agent": "periods", "status": "running", "t": ms()}
    with agent_span("period analysts", candidates=3):
        pairs = await team.refine(corpus, results, clues)
    best = team.best_per_entity(pairs + results)
    for r in best[:3]:
        yield {"type": "period", "entity": r["entity"], "period": r["period"], "score": r["score"], "t": ms()}
    evidence = team.build_evidence(best, cap)
    summary = team.summarize(best, evidence, clues)
    yield {"type": "team", "summary": summary, "t": ms()}

    yield {"type": "stage", "agent": "verifier", "status": "running", "t": ms()}
    with agent_span("verifier", evidence=len(evidence), cross_model=True):
        audit = await verifier.verify(question, evidence, summary)
    yield {"type": "verification", "round": 1, "audit": audit, "t": ms()}
    st.update(evidence=evidence, audit=audit, team=summary)


async def run(corpus: Corpus, question: str, mode: str = "answer", flow: str | None = None) -> AsyncIterator[dict]:
    t0 = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        if not corpus.chunks:
            yield {"type": "error", "message": "No dataset loaded. Load one from the MCP server first."}
            return
        cap = MAX_EVIDENCE.get(mode, 22)

        yield {"type": "stage", "agent": "planner", "status": "running", "t": ms()}
        with agent_span("planner"):
            plan = await planner.plan(question, corpus.overview)
        roster = team.members(corpus, plan["scope_docs"]) if plan["question_type"] == "identify" else []
        use_team = 2 <= len(roster) <= team.MAX_MEMBERS and mode == "answer"
        prefer = flow or settings.identify_flow
        yield {
            "type": "plan",
            "sub_queries": plan["sub_queries"],
            "interpretation": plan["interpretation"],
            "question_type": "identify" if use_team else "lookup",
            "flow": prefer if use_team else "lookup",
            "clues": plan["clues"] if use_team else [],
            "roster": [m["entity"] for m in roster] if use_team else [],
            "t": ms(),
        }

        st: dict = {}
        if use_team and prefer == "team":
            async for ev in team_flow(corpus, question, plan, roster, cap, st, ms):
                yield ev
        elif use_team and prefer == "auto":
            # fast path first; the analyst team only runs when the lookup can't show one company matching every clue
            async for ev in lookup_flow(corpus, question, plan, cap, st, ms, followups=False, clues=plan["clues"]):
                yield ev
            unsure = not st["audit"]["answerable"] or bool(st["audit"]["gaps"])
            if unsure:
                reason = (st["audit"]["gaps"] or ["evidence did not show one company satisfying every clue"])[0]
                yield {"type": "escalate", "reason": reason, "t": ms()}
                async for ev in team_flow(corpus, question, plan, roster, cap, st, ms):
                    yield ev
        else:
            async for ev in lookup_flow(corpus, question, plan, cap, st, ms):
                yield ev
        evidence, audit, squad = st["evidence"], st["audit"], st.get("team")

        yield {"type": "stage", "agent": "writer", "status": "running", "t": ms()}
        answer = ""
        with agent_span("writer", evidence=len(evidence), mode=mode):
            async for tok in writer.write(question, evidence, audit, mode, squad, identify=use_team):
                answer += tok
                yield {"type": "token", "text": tok}

        valid, invalid = writer.used_citations(answer, len(evidence))
        yield {
            "type": "citations",
            "items": [
                {
                    "n": n,
                    "id": evidence[n - 1].id,
                    "doc": evidence[n - 1].doc,
                    "section": short_section(evidence[n - 1]),
                    "period": evidence[n - 1].period,
                    "kind": evidence[n - 1].kind,
                    "start_line": evidence[n - 1].start_line,
                    "end_line": evidence[n - 1].end_line,
                    "text": evidence[n - 1].text,
                }
                for n in valid
            ],
        }
        ungrounded, derived = writer.check_numbers(answer, evidence)
        log_pipeline(mode, len(evidence), audit, ungrounded, derived, invalid, ms())
        yield {
            "type": "grounding",
            "invalid_citations": invalid,
            "ungrounded_numbers": ungrounded,
            "derived_numbers": derived,
            "conflicts": len(audit["conflicts"]),
            "gaps": audit["gaps"],
            "answerable": audit["answerable"],
            "team": {
                "team_pick": squad["leader"],
                "unique_full_match": squad["unique_full_match"],
                "verifier_winner": audit.get("winner"),
                "agrees": audit.get("agrees_with_team"),
                "verifier_model": audit.get("model"),
            }
            if squad
            else None,
        }
        yield {"type": "done", "t": ms()}
    except Exception as e:  # surfaced in the UI rather than a dead stream
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
