"""Compare the lookup / team / auto flows on the RBC prize questions.

    python -m scripts.compare_flows [out.json] [--concurrency N] [--limit N]

For every question and flow it records the company the answer names, end-to-end latency, and whether
auto-routing escalated to the analyst team. There are no published gold answers, so agreement between
flows is the signal, and disagreements are the ones worth reading against the source."""
import asyncio
import json
import statistics
import sys
import time

from app.config import ROOT, settings
from app.corpus import Corpus
from app.llm import chat_json
from app.orchestrator import run

FLOWS = ["lookup", "team", "auto"]


async def pick_entity(answer: str, entities: list[str]) -> str | None:
    out = await chat_json(
        f'Which single company does this answer name as THE answer to the question? Choose from: {entities}. '
        'If it names none or says it cannot tell, use null. Return JSON {"ticker": <ticker or null>}.',
        answer[:1500],
        settings.fast_model,
    )
    t = out.get("ticker")
    return t if t in entities else None


async def one(corpus: Corpus, q: dict, flow: str, entities: list[str], sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.perf_counter()
        answer, escalated, err, grounding = "", False, None, {}
        async for ev in run(corpus, q["q"], "answer", flow):
            if ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "escalate":
                escalated = True
            elif ev["type"] == "grounding":
                grounding = ev
            elif ev["type"] == "error":
                err = ev["message"]
        secs = round(time.perf_counter() - t0, 1)
    return {
        "q": q["q"], "sector": q["sector"], "flow": flow, "secs": secs, "escalated": escalated, "error": err,
        "pick": await pick_entity(answer, entities) if answer else None,
        "ungrounded": grounding.get("ungrounded_numbers", []), "answer": answer,
    }


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


async def main(out_path: str, concurrency: int, limit: int | None):
    questions = json.loads((ROOT / "eval" / "rbc_prize_questions.json").read_text(encoding="utf-8"))[:limit]
    corpus = Corpus()
    await corpus.load_from_mcp(settings.mcp_url)
    entities = [e for d in corpus.overview["documents"] for e in d["entities"]]
    sem = asyncio.Semaphore(concurrency)
    results = {}
    for flow in FLOWS:
        rows = await asyncio.gather(*[one(corpus, q, flow, entities, sem) for q in questions])
        results[flow] = rows
        secs = [r["secs"] for r in rows]
        print(f"{flow:7} median={statistics.median(secs):5.1f}s  p90={pct(secs, .9):5.1f}s  max={max(secs):5.1f}s  "
              f"errors={sum(bool(r['error']) for r in rows)}  no_pick={sum(r['pick'] is None for r in rows)}"
              + (f"  escalated={sum(r['escalated'] for r in rows)}/{len(rows)}" if flow == "auto" else ""), flush=True)
        json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print()
    for a, b in [("lookup", "team"), ("lookup", "auto"), ("team", "auto")]:
        same = sum(x["pick"] == y["pick"] for x, y in zip(results[a], results[b]))
        print(f"agreement {a} vs {b}: {same}/{len(questions)}")
    print("\nquestions where the flows disagree:")
    for i, (l, t, a) in enumerate(zip(results["lookup"], results["team"], results["auto"]), 1):
        if len({l["pick"], t["pick"], a["pick"]}) > 1:
            print(f"{i:2}. lookup={l['pick']} team={t['pick']} auto={a['pick']}{' (escalated)' if a['escalated'] else ''} | {l['q'][:110]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    conc = int(sys.argv[sys.argv.index("--concurrency") + 1]) if "--concurrency" in sys.argv else 4
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out = next((a for a in args if a.endswith(".json")), str(ROOT / "data" / "cache" / "flow_comparison.json"))
    asyncio.run(main(out, conc, lim))
