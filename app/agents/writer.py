import json
import re
from typing import AsyncIterator

from ..chunker import Chunk
from ..llm import chat_stream
from .common import format_evidence

RULES = """You are the Writer agent. Write the final answer using ONLY the numbered evidence passages and the verifier's audit.

Rules:
- Cite every factual claim with its passage number in square brackets immediately after the claim, e.g. "CET1 was 13.5% [3]". Use one bracket per source: [2][5], never [2, 5]. Never cite a number that is not in the evidence.
- State the reporting period next to each figure. Quote figures exactly as written in the source, including "~".
- If the verifier found conflicts, say so explicitly: state each conflicting value with its citation and which one you consider more authoritative and why (e.g. primary company disclosure or the more recent/specific section over a summary table), or say it cannot be resolved.
- Never speculate about WHY figures differ or what a value "probably" refers to. Explain a difference only if a passage states the reason (e.g. different period, adjusted vs reported); otherwise say the sources differ and the reason is not stated. Never attach a citation to a claim that passage does not make.
- If the evidence does not contain what was asked, say exactly what is missing instead of guessing. A partial, honest answer is better than a complete-looking one.
- Be concise and direct. Lead with the answer."""

ANSWER = RULES + "\n- Format: a short direct answer, then (only if useful) a few bullet points of supporting detail."

IDENTIFY = (
    RULES
    + """
- This is an identification question. A team of analyst agents graded each clue per company (see "Team analysis"), and an independent verifier model audited them. Always lead with a named answer: the verifier's "winner", or if that is null, the team's best-supported company (say "best match"). Then go through each clue in turn: state how that company satisfies it, with citations. Only AFTER that, note plainly any caveat: a clue that is unsupported, clues that fall in different reporting periods, or where the verifier disagreed with the team. Do not open with a refusal."""
)


IDENTIFY_LITE = ANSWER + "\n- This asks WHICH company satisfies several clues. Name the company first, then show how each clue is met, with citations, and only then note any caveat (an unsupported clue, or clues in different reporting periods). Do not open with a refusal."


async def write(
    question: str, evidence: list[Chunk], audit: dict, mode: str = "answer", team: dict | None = None, identify: bool = False
) -> AsyncIterator[str]:
    system = IDENTIFY if team else (IDENTIFY_LITE if identify else ANSWER)
    user = (
        f"Question: {question}\n\nVerifier audit (JSON):\n{json.dumps(audit, ensure_ascii=False)}\n\n"
        + (f"Team analysis (JSON):\n{json.dumps(team, ensure_ascii=False)}\n\n" if team else "")
        + f"Evidence:\n{format_evidence(evidence)}"
    )
    async for tok in chat_stream(system, user):
        yield tok


CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"[$~]?\d[\d,]*\.\d+%?|[$~]?\d[\d,]*%|\$\d[\d,]*")


def used_citations(answer: str, n_evidence: int) -> tuple[list[int], list[int]]:
    """Return (valid cited indices in first-use order, invalid indices the writer made up)."""
    valid, invalid = [], []
    for m in CITE_RE.finditer(answer):
        k = int(m.group(1))
        (valid if 1 <= k <= n_evidence else invalid).append(k)
    return list(dict.fromkeys(valid)), list(dict.fromkeys(invalid))


ANY_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def check_numbers(answer: str, evidence: list[Chunk]) -> tuple[list[str], list[str]]:
    """Deterministic audit of figures (%, $, decimals) in the answer.

    Returns (ungrounded, derived). A figure found verbatim in the evidence is grounded. One that is
    the sum or difference of two evidence figures (e.g. a margin change) is "derived": legitimate
    arithmetic, reported separately. Anything else is "ungrounded" and should be treated as suspect."""
    corpus_text = " ".join(c.text for c in evidence).replace(",", "").replace("~", "")
    values = {round(float(x), 2) for x in ANY_NUM_RE.findall(corpus_text)}
    ungrounded, derived = [], []
    for m in NUM_RE.finditer(CITE_RE.sub("", answer)):
        tok = m.group(0)
        norm = tok.replace(",", "").replace("~", "").lstrip("$")
        if norm in corpus_text or tok in ungrounded or tok in derived:
            continue
        try:
            v = round(float(norm.rstrip("%")), 2)
        except ValueError:
            ungrounded.append(tok)
            continue
        if any(round(a - v, 2) in values or round(v - a, 2) in values for a in values):
            derived.append(tok)
        else:
            ungrounded.append(tok)
    return ungrounded, derived
