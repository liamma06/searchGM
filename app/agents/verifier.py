from ..chunker import Chunk
from ..config import settings
from ..llm import chat_json, cross_model_available
from .common import format_evidence

SYSTEM = """You are the Verifier agent. You audit retrieved evidence BEFORE an answer is written.
You are given a question and numbered evidence passages [n] taken from financial research documents (each with file, line range, section and reporting period).

Do the following, using ONLY the evidence:
1. Extract the facts needed to answer the question, each tied to the passage numbers that state it. Note the reporting period and whether the value is exact or approximate ("~").
2. Detect CONFLICTS: the same metric, entity AND period reported with different values in different passages or sections (e.g. a snapshot table vs a deep dive). Different periods are NOT conflicts. Rounding or "~" approximations of the same number are NOT conflicts but must be noted as approximate.
3. Decide "answerable": true only if the evidence directly supports a precise answer. If a needed figure, entity or period is absent, list it under "gaps" and propose targeted "follow_up_queries" (with entities copied from the evidence metadata when known).
4. Never invent values, and never fill gaps from outside knowledge.
5. Take a passage's period only from its own section label or text. If a passage states a value without a period, record the period as "not stated" instead of inferring it from other passages; a value that appears only as an approximation ("~") for the latest period while the exact figure is stated for an earlier one should be reported as exactly that.

Return JSON:
{"answerable": bool,
 "facts": [{"claim": str, "value": str, "period": str, "exact": bool, "sources": [int]}],
 "conflicts": [{"topic": str, "values": [{"value": str, "sources": [int]}], "assessment": str}],
 "gaps": [str],
 "follow_up_queries": [{"query": str, "entities": [str]}]}"""


TEAM_ADDENDUM = """

This is an IDENTIFICATION question ("which company ..."). A team of analyst agents, one per company, graded every clue for their own company; their matrix follows, citing the evidence passages by number. Audit it independently of them:
- Check each "supported" verdict against the passages it cites. Treat it as NOT holding if the number, period or wording in the passage does not actually match the clue.
- Clues in a question like this normally all describe ONE reporting period. Each candidate is shown at the single period where the most clues hold together; judge the company at that period, and do not reject it merely because some clue would also hold in a different quarter.
- Decide the winner: the single entity for which ALL clues hold, or null if none or more than one does.
Also return: "winner": <entity id or null>, "agrees_with_team": <true if your winner equals the team's unique full match>, "adjudication": "<= 40 words, explaining any disagreement with the team>". Return an empty "follow_up_queries"."""


def team_block(team: dict) -> str:
    lines = ["Clues:"] + [f"  {i}. {c}" for i, c in enumerate(team["clues"], 1)]
    lines.append(f"Team leader's pick: {team['leader']} (unique full match: {team['unique_full_match']}; full matches: {team['full_matches']})")
    for row in team["candidates"]:
        lines.append(f"Analyst {row['entity']} ({row['name']}), best reporting period {row.get('period') or 'n/a'}, score {row['score']}:")
        for v in row["verdicts"]:
            lines.append(f"  clue {v['id']}: {v['verdict']} {v['passages']} {v['note']}")
    return "\n".join(lines)


LOOKUP_IDENTIFY_ADDENDUM = """

The question asks WHICH company satisfies a set of clues:
{clues}
Set "answerable" to true only if the evidence shows ONE company satisfying every clue (ideally in the same reporting period). Otherwise set it to false and list each unresolved clue under "gaps"."""


async def verify(question: str, evidence: list[Chunk], team: dict | None = None, clues: list[str] | None = None) -> dict:
    user = f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}"
    system, model = SYSTEM, settings.chat_model
    if clues and not team:
        numbered = "\n".join(f"  {i}. {c}" for i, c in enumerate(clues, 1))
        system += LOOKUP_IDENTIFY_ADDENDUM.format(clues=numbered)
    if team:
        system += TEAM_ADDENDUM
        user += f"\n\nTeam analysis:\n{team_block(team)}"
    out = None
    if team and cross_model_available():  # a different model family adjudicates, so errors are less likely to be shared
        try:
            out = await chat_json(
                system,
                user,
                settings.verifier_model,
                provider="openrouter",
                timeout=settings.verifier_timeout,
                extra_body={"reasoning": {"enabled": False}},
            )
            model = settings.verifier_model
        except Exception:
            out = None
    if out is None:
        out = await chat_json(system, user, settings.chat_model)
    out["model"] = model
    n = len(evidence)

    def clean(srcs):
        return [s for s in (srcs or []) if isinstance(s, int) and 1 <= s <= n]

    for f in out.get("facts", []):
        f["sources"] = clean(f.get("sources"))
    for c in out.get("conflicts", []):
        for v in c.get("values", []):
            v["sources"] = clean(v.get("sources"))
    out.setdefault("facts", [])
    out.setdefault("conflicts", [])
    out.setdefault("gaps", [])
    out.setdefault("follow_up_queries", [])
    out["answerable"] = bool(out.get("answerable"))
    if team:
        out["winner"] = out.get("winner") if out.get("winner") in {r["entity"] for r in team["candidates"]} else None
        out["agrees_with_team"] = bool(out.get("agrees_with_team"))
        out["adjudication"] = str(out.get("adjudication", ""))[:300]
    return out
