import asyncio
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import auth, db, gptzero, observability
from .auth import admin_user, current_user, optional_user, require_xhr
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
    db.init()
    task = asyncio.create_task(load(settings.mcp_url))  # index in the background so the UI is up immediately
    yield
    task.cancel()


app = FastAPI(title="signal", lifespan=lifespan)
if not settings.session_secret:
    print("WARNING: SESSION_SECRET is not set; using a temporary one, so sign-ins reset whenever the server restarts.")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret or secrets.token_urlsafe(32),
    session_cookie="signal_session",
    same_site="lax",
    https_only=settings.cookie_secure,
    max_age=7 * 24 * 3600,
)
app.include_router(auth.router)


class AskBody(BaseModel):
    question: str
    flow: str | None = None  # "team" | "auto" | "lookup"; None uses IDENTIFY_FLOW
    chat_id: str | None = None  # continue an existing chat; None starts a new one


class BriefBody(BaseModel):
    entity: str
    chat_id: str | None = None


class IngestBody(BaseModel):
    mcp_url: str


def sse(events):
    async def gen():
        async for ev in events:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def persisted(events, chat_id: str, question: str, kind: str):
    """Pass events through to the client and save the finished run. Tokens are folded into one answer string;
    everything else is kept as the event stream so a past chat replays exactly."""
    yield {"type": "chat", "id": chat_id}
    collected, answer, t0, saved = [], "", time.time(), False
    try:
        async for ev in events:
            if ev["type"] == "token":
                answer += ev["text"]
            else:
                collected.append(ev)
            yield ev
        if any(e["type"] in ("done", "error") for e in collected):
            await asyncio.to_thread(db.save_run, chat_id, question, kind, answer, collected, time.time() - t0)
            saved = True
    finally:  # the browser disconnected mid-run: save what finished, without blocking
        if not saved and any(e["type"] in ("done", "error") for e in collected):
            threading.Thread(target=db.save_run, args=(chat_id, question, kind, answer, collected, time.time() - t0), daemon=True).start()


def start_chat(user: dict, chat_id: str | None, title: str) -> str:
    if chat_id:
        if not db.get_chat(chat_id, user["id"]):
            raise HTTPException(404, "Chat not found")
        return chat_id
    return db.create_chat(user["id"], title)


# ---------- app ----------
@app.get("/api/status")
def status(user: dict = Depends(current_user)):
    return {**corpus.status(), "loading": state["loading"], "error": state["error"], "models": {"chat": settings.chat_model, "fast": settings.fast_model, "embed": settings.embed_model},
            "integrations": {"sentry": observability.enabled(), "gptzero": gptzero.enabled()}, "is_admin": user["is_admin"]}


@app.post("/api/ingest", dependencies=[Depends(require_xhr)])
async def ingest(body: IngestBody, user: dict = Depends(admin_user)):
    url = body.mcp_url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "The MCP URL must start with http:// or https://")
    if state["loading"]:
        raise HTTPException(409, "A dataset is already loading")
    await load(url)
    if state["error"]:
        raise HTTPException(502, state["error"])
    return status(user)


@app.post("/api/ask", dependencies=[Depends(require_xhr)])
async def ask(body: AskBody, user: dict = Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Ask a question")
    auth.check_ask_rate(user)
    flow = body.flow if body.flow in ("team", "auto", "lookup") else None
    chat_id = await asyncio.to_thread(start_chat, user, body.chat_id, question)
    return sse(persisted(run(corpus, question, "answer", flow), chat_id, question, "ask"))


@app.post("/api/brief", dependencies=[Depends(require_xhr)])
async def brief(body: BriefBody, user: dict = Depends(current_user)):
    entity = body.entity.strip()
    if not entity:
        raise HTTPException(400, "Pick a company")
    auth.check_ask_rate(user)
    question = (
        f"Produce an analyst brief for {entity}: its latest-quarter results and key metrics, how they trended over the "
        f"trailing quarters, upcoming catalysts and risks, insider/institutional activity, and flag any conflicting, "
        f"approximate or missing data."
    )
    chat_id = await asyncio.to_thread(start_chat, user, body.chat_id, f"Analyst brief: {entity}")
    return sse(persisted(run(corpus, question, "brief"), chat_id, f"Analyst brief: {entity}", "brief"))


# ---------- saved chats ----------
@app.get("/api/chats")
def chats(user: dict = Depends(current_user)):
    return db.list_chats(user["id"])


@app.get("/api/chats/{chat_id}")
def chat_detail(chat_id: str, user: dict = Depends(current_user)):
    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        raise HTTPException(404, "Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}", dependencies=[Depends(require_xhr)])
def chat_delete(chat_id: str, user: dict = Depends(current_user)):
    if not db.delete_chat(chat_id, user["id"]):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.post("/api/chats/{chat_id}/share", dependencies=[Depends(require_xhr)])
def chat_share(chat_id: str, user: dict = Depends(current_user)):
    try:
        return {"token": db.set_share(chat_id, user["id"], True)}
    except LookupError:
        raise HTTPException(404, "Chat not found")


@app.delete("/api/chats/{chat_id}/share", dependencies=[Depends(require_xhr)])
def chat_unshare(chat_id: str, user: dict = Depends(current_user)):
    try:
        db.set_share(chat_id, user["id"], False)
    except LookupError:
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.get("/api/shared/{token}")
def shared(token: str):
    """Read-only view of a shared chat. No sign-in needed; the unguessable token is the permission."""
    chat = db.get_shared(token)
    if not chat:
        raise HTTPException(404, "This link is no longer shared")
    return {"title": chat["title"], "runs": chat["runs"]}


# ---------- sources and GPTZero ----------
@app.get("/api/chunk/{chunk_id:path}")
def chunk(chunk_id: str, radius: int = 4, share: str | None = Query(None), user: dict | None = Depends(optional_user)):
    if user is None and not (share and db.get_shared(share)):  # a shared chat may read the sources it cites
        raise HTTPException(401, "Sign in required")
    ctx = corpus.source_context(chunk_id, min(max(radius, 0), 20))
    if not ctx:
        raise HTTPException(404, "unknown chunk")
    return ctx


@app.get("/api/gptzero/chunk/{chunk_id:path}")
async def gptzero_chunk(chunk_id: str, user: dict = Depends(current_user)):
    c = corpus.by_id.get(chunk_id)
    if not c:
        raise HTTPException(404, "unknown chunk")
    if c.kind != "text":  # detectors score prose; table rows would just read as "human"
        return {"skipped": "structured table data"}
    return await gptzero.detect(c.text)


@app.post("/api/gptzero/scan", dependencies=[Depends(require_xhr)])
async def gptzero_scan_start(user: dict = Depends(admin_user)):
    if not gptzero.enabled():
        raise HTTPException(400, "GPTZERO_API_KEY not set")
    if not gptzero.scan.running:
        asyncio.create_task(gptzero.scan.run(corpus))
    return gptzero.scan.summary(corpus)


@app.get("/api/gptzero/scan")
def gptzero_scan_status(user: dict = Depends(current_user)):
    return gptzero.scan.summary(corpus)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
