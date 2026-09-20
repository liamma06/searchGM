import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import gptzero, observability
from .config import ROOT, settings
from .corpus import Corpus
from .orchestrator import run

observability.init()  # before the app is created so Sentry's FastAPI integration can patch it

corpus = Corpus()
state = {"loading": False, "error": None}
FRONTEND = ROOT / "frontend"


async def load(url: str) -> None:
    state.update(loading=True, error=None)
    try:
        await corpus.load_from_mcp(url)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["loading"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(load(settings.mcp_url))  # index in the background so the UI is up immediately
    yield
    task.cancel()


app = FastAPI(title="signal", lifespan=lifespan)


class AskBody(BaseModel):
    question: str
    flow: str | None = None  # "team" | "auto" | "lookup"; None uses IDENTIFY_FLOW


class BriefBody(BaseModel):
    entity: str


class IngestBody(BaseModel):
    mcp_url: str


def sse(events):
    async def gen():
        async for ev in events:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/status")
def status():
    return {**corpus.status(), "loading": state["loading"], "error": state["error"], "models": {"chat": settings.chat_model, "fast": settings.fast_model, "embed": settings.embed_model},
            "integrations": {"sentry": observability.enabled(), "gptzero": gptzero.enabled()}}


@app.post("/api/ingest")
async def ingest(body: IngestBody):
    if state["loading"]:
        raise HTTPException(409, "A dataset is already loading")
    await load(body.mcp_url.strip())
    if state["error"]:
        raise HTTPException(502, state["error"])
    return status()


@app.post("/api/ask")
async def ask(body: AskBody):
    flow = body.flow if body.flow in ("team", "auto", "lookup") else None
    return sse(run(corpus, body.question.strip(), "answer", flow))


@app.post("/api/brief")
async def brief(body: BriefBody):
    entity = body.entity.strip()
    question = (
        f"Produce an analyst brief for {entity}: its latest-quarter results and key metrics, how they trended over the "
        f"trailing quarters, upcoming catalysts and risks, insider/institutional activity, and flag any conflicting, "
        f"approximate or missing data."
    )
    return sse(run(corpus, question, "brief"))


@app.get("/api/chunk/{chunk_id:path}")
def chunk(chunk_id: str, radius: int = 4):
    ctx = corpus.source_context(chunk_id, min(max(radius, 0), 20))
    if not ctx:
        raise HTTPException(404, "unknown chunk")
    return ctx


@app.get("/api/gptzero/chunk/{chunk_id:path}")
async def gptzero_chunk(chunk_id: str):
    c = corpus.by_id.get(chunk_id)
    if not c:
        raise HTTPException(404, "unknown chunk")
    if c.kind != "text":  # detectors score prose; table rows would just read as "human"
        return {"skipped": "structured table data"}
    return await gptzero.detect(c.text)


@app.post("/api/gptzero/scan")
async def gptzero_scan_start():
    if not gptzero.enabled():
        raise HTTPException(400, "GPTZERO_API_KEY not set")
    if not gptzero.scan.running:
        asyncio.create_task(gptzero.scan.run(corpus))
    return gptzero.scan.summary(corpus)


@app.get("/api/gptzero/scan")
def gptzero_scan_status():
    return gptzero.scan.summary(corpus)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
