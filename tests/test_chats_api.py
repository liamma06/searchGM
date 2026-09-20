import json

import pytest
from fastapi.testclient import TestClient

from app import auth, db
from app.config import settings

H = {"X-Requested-With": "fetch"}


async def fake_run(corpus, question, mode="answer", flow=None):
    yield {"type": "plan", "sub_queries": [], "interpretation": "", "question_type": "lookup", "flow": "lookup", "clues": [], "roster": [], "t": 1}
    yield {"type": "token", "text": "TD's CET1 was **14.3%** [1]. "}
    yield {"type": "token", "text": "It rose [2]."}
    yield {"type": "done", "t": 2}


@pytest.fixture
def env(monkeypatch):
    import app.main as main

    db.init("")
    monkeypatch.setattr(settings, "auth_required", True)
    monkeypatch.setattr(settings, "rate_limit_per_hour", 100)
    monkeypatch.setattr(main, "run", fake_run)
    for lim in (auth.login_failures, auth.signups, auth.asks):
        lim.hits.clear()
    a = TestClient(main.app)
    a.post("/auth/signup", json={"email": "a@example.com", "password": "correct horse"}, headers=H)
    b = TestClient(main.app)
    b.post("/auth/signup", json={"email": "b@example.com", "password": "correct horse"}, headers=H)
    return a, b


def ask(c, q, chat_id=None):
    c.post("/api/ask", json={"question": q, "chat_id": chat_id}, headers=H)
    return c.get("/api/chats").json()


def test_list_shows_a_clean_preview_and_counts(env):
    a, _ = env
    chats = ask(a, "What was TD's CET1?")
    assert chats[0]["preview"] == "TD's CET1 was 14.3%. It rose."  # citations, markdown and stray spaces removed
    assert chats[0]["runs"] == 1 and chats[0]["created_at"] <= chats[0]["updated_at"]


def test_rename(env):
    a, b = env
    cid = ask(a, "first")[0]["id"]
    assert a.patch(f"/api/chats/{cid}", json={"title": "  Q3   capital   review  "}, headers=H).json() == {"title": "Q3 capital review"}
    assert a.get("/api/chats").json()[0]["title"] == "Q3 capital review"
    assert a.patch(f"/api/chats/{cid}", json={"title": "   "}, headers=H).status_code == 400
    assert a.patch(f"/api/chats/{cid}", json={"title": "x"}).status_code == 403  # CSRF header required
    assert b.patch(f"/api/chats/{cid}", json={"title": "hijack"}, headers=H).status_code == 404  # not yours
    assert a.get("/api/chats").json()[0]["title"] == "Q3 capital review"
    long = a.patch(f"/api/chats/{cid}", json={"title": "y" * 500}, headers=H).json()["title"]
    assert len(long) == 120


def test_markup_in_titles_is_stored_as_plain_text(env):
    a, _ = env
    cid = ask(a, "<img src=x onerror=alert(1)>")[0]["id"]
    assert a.get("/api/chats").json()[0]["title"] == "<img src=x onerror=alert(1)>"  # escaped by the page when shown, never trusted as HTML


def test_bulk_delete_only_removes_your_own_chats(env):
    a, b = env
    ids = [ask(a, f"q{i}")[0]["id"] for i in range(3)]
    other = ask(b, "b's chat")[0]["id"]
    r = a.post("/api/chats/delete", json={"ids": [ids[0], ids[1], other, "nonexistent"]}, headers=H)
    assert r.json() == {"deleted": 2}  # b's chat and the unknown id are ignored
    assert [c["id"] for c in a.get("/api/chats").json()] == [ids[2]]
    assert [c["id"] for c in b.get("/api/chats").json()] == [other]
    assert a.get(f"/api/chats/{ids[0]}").status_code == 404
    assert db._d().runs.count_documents({"chat_id": ids[0]}) == 0  # the questions inside go too
    assert a.post("/api/chats/delete", json={"ids": []}, headers=H).status_code == 400
    assert a.post("/api/chats/delete", json={"ids": ["x"] * 201}, headers=H).status_code == 400
    assert a.post("/api/chats/delete", json={"ids": ids}).status_code == 403


def test_shared_link_stops_working_when_the_chat_is_deleted(env):
    a, _ = env
    cid = ask(a, "share me")[0]["id"]
    token = a.post(f"/api/chats/{cid}/share", headers=H).json()["token"]
    anon = TestClient(a.app)
    assert anon.get(f"/api/shared/{token}").status_code == 200
    a.delete(f"/api/chats/{cid}", headers=H)
    assert anon.get(f"/api/shared/{token}").status_code == 404


def test_every_api_route_refuses_anonymous_visitors_except_the_public_ones(env):
    a, _ = env
    anon = TestClient(a.app)
    public = {"/api/me", "/api/shared/{token}", "/api/chunk/{chunk_id:path}"}  # chunk is open only with a valid share token
    for route in a.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/") or path in public:
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            url = path.replace("{chat_id}", "x").replace("{slug}", "slack").replace("{chunk_id:path}", "x")
            r = anon.request(method, url, headers=H, json={} if method in ("POST", "PATCH", "PUT") else None)
            assert r.status_code == 401, f"{method} {path} returned {r.status_code} for a signed-out visitor"
    assert anon.get("/api/chunk/x").status_code == 401  # and the chunk route needs sign-in or a real share token
    assert anon.get("/api/chunk/x?share=not-a-token").status_code == 401
