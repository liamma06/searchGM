"""GPTZero integration: flags AI-written passages in the source corpus and in cited evidence, so a
data-quality signal ("this analyst blurb looks machine-generated") sits next to every citation.

Disabled (every call returns a `skipped` result) unless GPTZERO_API_KEY is set. Errors from the API
are returned as `{"error": ...}` and never raised, so a GPTZero outage can't break an answer."""
import asyncio
import hashlib

import httpx

from .config import settings
from .corpus import Corpus

URL = "https://api.gptzero.me/v2/predict/text"
MIN_CHARS = 250  # GPTZero needs a minimum amount of text to classify
FLAG_AT = 50  # ai_pct at or above this is shown as flagged

_cache: dict[str, dict] = {}
_sem: tuple | None = None  # (loop, semaphore): concurrency cap shared by the badge lookups and the scan
MAX_CONCURRENT = 3
RETRIES = 3


def _semaphore() -> asyncio.Semaphore:
    global _sem
    loop = asyncio.get_running_loop()
    if _sem is None or _sem[0] is not loop:
        _sem = (loop, asyncio.Semaphore(MAX_CONCURRENT))
    return _sem[1]


def enabled() -> bool:
    return bool(settings.gptzero_api_key)


def parse(doc: dict) -> dict:
    probs = doc.get("class_probabilities") or {}
    ai = probs.get("ai")
    if ai is None:
        ai = doc.get("completely_generated_prob", doc.get("average_generated_prob", 0.0))
    label = doc.get("predicted_class") or ("ai" if ai >= 0.5 else "human")
    flagged = [s.get("sentence", "")[:160] for s in doc.get("sentences", []) if s.get("generated_prob", 0) >= 0.8][:3]
    return {"label": label, "ai_pct": round(float(ai) * 100), "flagged_sentences": flagged}


async def detect(text: str, client: httpx.AsyncClient | None = None) -> dict:
    if not enabled():
        return {"skipped": "GPTZERO_API_KEY not set"}
    if len(text) < MIN_CHARS:
        return {"skipped": f"under {MIN_CHARS} characters"}
    key = hashlib.sha1(text.encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    headers = {"x-api-key": settings.gptzero_api_key, "Accept": "application/json", "Content-Type": "application/json"}
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        for attempt in range(RETRIES + 1):
            async with _semaphore():
                r = await client.post(URL, headers=headers, json={"document": text, "multilingual": False})
            if r.status_code == 429 or r.status_code >= 500:  # transient: rate limit or gateway error
                if attempt < RETRIES:
                    await asyncio.sleep(0.6 * 2**attempt)
                    continue
            break
        if r.status_code != 200:
            return {"error": f"GPTZero {r.status_code}: {r.text[:120]}"}
        result = parse(r.json()["documents"][0])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if own:
            await client.aclose()
    _cache[key] = result
    return result


class Scan:
    """Background scan of every long-enough text passage in the corpus."""

    def __init__(self):
        self.running = False
        self.total = 0
        self.done = 0
        self.errors = 0
        self.results: dict[str, dict] = {}
        self.first_error: str | None = None

    async def run(self, corpus: Corpus, concurrency: int = 4) -> None:
        targets = [c for c in corpus.chunks if c.kind == "text" and len(c.text) >= MIN_CHARS]
        self.running, self.total, self.done, self.errors, self.results, self.first_error = True, len(targets), 0, 0, {}, None
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=30) as client:

            async def one(c):
                async with sem:
                    if self.first_error and self.errors >= 3:  # e.g. bad key or out of credits: stop burning requests
                        self.done += 1
                        return
                    res = await detect(c.text, client)
                    if "error" in res:
                        self.errors += 1
                        self.first_error = self.first_error or res["error"]
                    else:
                        self.results[c.id] = res
                    self.done += 1

            await asyncio.gather(*[one(c) for c in targets])
        self.running = False

    def summary(self, corpus: Corpus, top: int = 12) -> dict:
        flagged = sorted(
            ((cid, r) for cid, r in self.results.items() if r.get("ai_pct", 0) >= FLAG_AT),
            key=lambda kv: kv[1]["ai_pct"],
            reverse=True,
        )
        items = []
        for cid, r in flagged[:top]:
            c = corpus.by_id.get(cid)
            if c:
                items.append({"id": cid, "doc": c.doc, "section": c.section.split(" > ", 1)[-1], "start_line": c.start_line,
                              "end_line": c.end_line, "ai_pct": r["ai_pct"], "sample": (r["flagged_sentences"] or [c.text[:160]])[0]})
        return {
            "enabled": enabled(),
            "running": self.running,
            "total": self.total,
            "done": self.done,
            "scanned": len(self.results),
            "flagged": len(flagged),
            "errors": self.errors,
            "error": self.first_error,
            "items": items,
        }


scan = Scan()
