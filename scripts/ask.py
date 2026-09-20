"""CLI: python -m scripts.ask "question" [--brief]"""
import asyncio
import json
import sys

from app.config import settings
from app.corpus import Corpus
from app.orchestrator import run


async def main(question: str, mode: str):
    corpus = Corpus()
    await corpus.load_from_mcp(settings.mcp_url)
    print(f"[{corpus.backend}] {len(corpus.chunks)} chunks | {corpus.notice}\n")
    async for ev in run(corpus, question, mode):
        t = ev["type"]
        if t == "stage":
            print(f"[{ev['t'] / 1000:5.1f}s] stage: {ev['agent']}")
        elif t == "period":
            print(f"PERIOD {ev['entity']:7} best={ev['period']!r} score={ev['score']}")
        elif t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "plan":
            print("PLAN:", json.dumps(ev["sub_queries"], ensure_ascii=False), "\n")
        elif t == "retrieval":
            print(f"RETRIEVE {'(follow-up) ' if ev.get('followup') else ''}{ev['query']!r}: " + ", ".join(f"{h['entity']}:L{h['start_line']}" for h in ev["hits"]))
        elif t == "verification":
            a = ev["audit"]
            if "winner" in a:
                print(f"CROSS-MODEL VERIFIER ({a.get('model')}): winner={a['winner']} agrees_with_team={a['agrees_with_team']} | {a['adjudication']}")
            print(f"VERIFY r{ev['round']}: answerable={a['answerable']} facts={len(a['facts'])} conflicts={json.dumps(a['conflicts'], ensure_ascii=False)} gaps={a['gaps']}\n")
        elif t == "analyst":
            mark = {"supported": "Y", "contradicted": "X", "no_evidence": "-"}
            print(f"ANALYST {ev['entity']:7} score={ev['score']:>3} " + " ".join(mark[v["verdict"]] for v in ev["verdicts"]))
        elif t == "team":
            sm = ev["summary"]
            print(f"TEAM leader={sm['leader']} unique_full_match={sm['unique_full_match']} full_matches={sm['full_matches']}")
        elif t == "citations":
            print("\n\nCITATIONS:")
            for c in ev["items"]:
                print(f"  [{c['n']}] {c['doc']} L{c['start_line']}-{c['end_line']} | {c['section']}")
        elif t in ("grounding", "error", "done"):
            print(t.upper(), {k: v for k, v in ev.items() if k != "type"})


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    asyncio.run(main(" ".join(args), "brief" if "--brief" in sys.argv else "answer"))
