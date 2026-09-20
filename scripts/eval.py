"""Regression eval: python -m scripts.eval [eval/questions.json]

A case passes when every `expect` substring appears in a cited answer with valid citations and no
ungrounded figures, or, for `unanswerable` cases, when the system reports missing data instead of
answering. Add the RBC prize questions to eval/questions.json to track them the same way."""
import asyncio
import json
import sys
from pathlib import Path

from app.config import ROOT, settings
from app.corpus import Corpus
from app.orchestrator import run


async def run_case(corpus: Corpus, case: dict) -> dict:
    answer, grounding, cites = "", {}, []
    async for ev in run(corpus, case["q"]):
        if ev["type"] == "token":
            answer += ev["text"]
        elif ev["type"] == "grounding":
            grounding = ev
        elif ev["type"] == "citations":
            cites = ev["items"]
        elif ev["type"] == "error":
            return {"pass": False, "why": ev["message"], "answer": answer}
    if case.get("unanswerable"):
        ok = not grounding.get("answerable", True) or bool(grounding.get("gaps"))
        return {"pass": ok, "why": "" if ok else "answered an unanswerable question", "answer": answer}
    missing = [e for e in case.get("expect", []) if e.lower() not in answer.lower()]
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if grounding.get("ungrounded_numbers"):
        problems.append(f"ungrounded {grounding['ungrounded_numbers']}")
    if grounding.get("invalid_citations"):
        problems.append(f"invalid citations {grounding['invalid_citations']}")
    if not cites:
        problems.append("no citations")
    return {"pass": not problems, "why": "; ".join(problems), "answer": answer}


async def main(path: Path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    corpus = Corpus()
    await corpus.load_from_mcp(settings.mcp_url)
    results = await asyncio.gather(*[run_case(corpus, c) for c in cases])
    for c, r in zip(cases, results):
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {c['q']}" + (f"\n      {r['why']}\n      {r['answer'][:300]!r}" if not r["pass"] else ""))
    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "questions.json"))
