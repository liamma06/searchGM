import asyncio
import json

import httpx
import sentry_sdk

from app import gptzero, observability
from app.chunker import Chunk
from app.config import settings

LONG = "Analysts noted the significant improvement in profitability and operating leverage across all segments. " * 4


def _chunk(cid="c1", text=LONG):
    return Chunk(cid, "d.md", "T", "S", "X", "text", text, 1, 2)


# ---------- GPTZero ----------
def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_gptzero_disabled_without_key(monkeypatch):
    monkeypatch.setattr(settings, "gptzero_api_key", "")
    assert asyncio.run(gptzero.detect(LONG)) == {"skipped": "GPTZERO_API_KEY not set"}


def test_gptzero_parses_caches_and_skips_short(monkeypatch):
    monkeypatch.setattr(settings, "gptzero_api_key", "k")
    gptzero._cache.clear()
    calls = []

    def handler(req):
        calls.append(req)
        assert req.headers["x-api-key"] == "k"
        body = {"documents": [{"predicted_class": "ai", "class_probabilities": {"ai": 0.91, "human": 0.09},
                               "sentences": [{"sentence": "Robotic sentence.", "generated_prob": 0.95}, {"sentence": "ok", "generated_prob": 0.1}]}]}
        return httpx.Response(200, json=body)

    async def go():
        async with _client(handler) as c:
            first = await gptzero.detect(LONG, c)
            again = await gptzero.detect(LONG, c)
            short = await gptzero.detect("too short", c)
        return first, again, short

    first, again, short = asyncio.run(go())
    assert first == {"label": "ai", "ai_pct": 91, "flagged_sentences": ["Robotic sentence."]}
    assert again == first and len(calls) == 1  # cached
    assert "skipped" in short


def test_gptzero_api_errors_are_returned_not_raised(monkeypatch):
    monkeypatch.setattr(settings, "gptzero_api_key", "k")
    gptzero._cache.clear()
    res = asyncio.run(_run_detect(lambda req: httpx.Response(402, text="out of credits")))
    assert res["error"].startswith("GPTZero 402")


async def _run_detect(handler):
    async with _client(handler) as c:
        return await gptzero.detect(LONG, c)


def test_gptzero_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(settings, "gptzero_api_key", "k")
    monkeypatch.setattr(gptzero, "RETRIES", 2)
    gptzero._cache.clear()
    codes = iter([502, 429, 200])

    def handler(req):
        code = next(codes)
        if code != 200:
            return httpx.Response(code, text="busy")
        return httpx.Response(200, json={"documents": [{"predicted_class": "human", "class_probabilities": {"ai": 0.02}, "sentences": []}]})

    async def go():
        real_sleep = asyncio.sleep

        async def fast(_):
            await real_sleep(0)

        monkeypatch.setattr(gptzero.asyncio, "sleep", fast)
        return await _run_detect(handler)

    assert asyncio.run(go()) == {"label": "human", "ai_pct": 2, "flagged_sentences": []}


def test_scan_flags_ai_passages_and_stops_on_repeated_errors(monkeypatch):
    from app.corpus import Corpus

    monkeypatch.setattr(settings, "gptzero_api_key", "k")
    gptzero._cache.clear()
    corpus = Corpus()
    corpus.chunks = [_chunk(f"c{i}", LONG + str(i)) for i in range(3)] + [_chunk("short", "tiny")]
    corpus.by_id = {c.id: c for c in corpus.chunks}

    async def fake_detect(text, client=None):
        return {"label": "ai", "ai_pct": 80, "flagged_sentences": []}

    monkeypatch.setattr(gptzero, "detect", fake_detect)
    s = gptzero.Scan()
    asyncio.run(s.run(corpus))
    out = s.summary(corpus)
    assert out["total"] == 3 and out["flagged"] == 3 and not out["running"]

    async def failing(text, client=None):
        return {"error": "GPTZero 401: bad key"}

    monkeypatch.setattr(gptzero, "detect", failing)
    corpus.chunks = [_chunk(f"c{i}", LONG + str(i)) for i in range(20)]
    corpus.by_id = {c.id: c for c in corpus.chunks}
    s2 = gptzero.Scan()
    asyncio.run(s2.run(corpus, concurrency=1))
    assert s2.summary(corpus)["error"].startswith("GPTZero 401") and s2.errors <= 4  # stopped early


# ---------- Sentry ----------
def test_sentry_is_noop_without_dsn(monkeypatch):
    monkeypatch.setattr(settings, "sentry_dsn", "")
    monkeypatch.setattr(observability, "_enabled", False)
    assert observability.init() is False
    with observability.agent_span("planner"):  # must not raise
        pass
    observability.log_pipeline("answer", 3, {"answerable": True, "conflicts": [], "gaps": []}, [], [], [], 10)


def test_sentry_sends_agent_spans_and_warns_on_ungrounded(monkeypatch):
    from sentry_sdk.transport import Transport

    sent = []

    class Capture(Transport):
        def capture_envelope(self, envelope):
            for item in envelope.items:
                sent.append((item.type, item.payload.json))

    monkeypatch.setattr(settings, "sentry_dsn", "https://key@o0.ingest.sentry.io/0")
    monkeypatch.setattr(observability, "_enabled", False)
    assert observability.init(transport=Capture()) is True
    audit = {"answerable": True, "conflicts": [1], "gaps": []}
    with sentry_sdk.start_transaction(op="http.server", name="POST /api/ask"):
        with observability.agent_span("planner", sub_queries=2):
            pass
        with observability.agent_span("writer", evidence=5):
            pass
        observability.log_pipeline("answer", 5, audit, ["$9.99"], [], [], 1234)
    sentry_sdk.flush()
    # this SDK streams spans as separate "span" envelope items
    spans = {sp["name"]: sp for t, p in sent if t == "span" for sp in p["items"]}
    assert set(spans) == {"invoke_agent planner", "invoke_agent writer"}
    for sp in spans.values():
        assert sp["attributes"]["sentry.op"]["value"] == "gen_ai.invoke_agent"
        assert sp["parent_span_id"]  # nested under the request transaction
    assert spans["invoke_agent planner"]["attributes"]["gen_ai.agent.name"]["value"] == "planner"
    events = [p for t, p in sent if t == "event"]
    assert any("not backed by evidence" in (e.get("message") or "") and e["level"] == "warning" for e in events)
    sentry_sdk.get_client().close()
    monkeypatch.setattr(observability, "_enabled", False)
