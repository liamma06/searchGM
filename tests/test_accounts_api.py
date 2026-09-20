import json

import pytest
from fastapi.testclient import TestClient

from app import auth, db
from app.config import settings

H = {"X-Requested-With": "fetch"}


async def fake_run(corpus, question, mode="answer", flow=None):
    """Stands in for the pipeline: same event shapes, no models."""
    yield {"type": "stage", "agent": "planner", "status": "running", "t": 0}
    yield {"type": "plan", "sub_queries": [{"query": question, "entities": [], "purpose": ""}], "interpretation": "", "question_type": "lookup", "flow": "lookup", "clues": [], "roster": [], "t": 5}
    yield {"type": "token", "text": "TD's CET1 was "}
    yield {"type": "token", "text": "14.3% [1]."}
    yield {"type": "citations", "items": [{"n": 1, "id": "d:1-1", "doc": "d.md", "section": "S", "period": "", "kind": "text", "start_line": 1, "end_line": 1, "text": "x"}]}
    yield {"type": "done", "t": 9}


@pytest.fixture
def client(monkeypatch):
    import app.main as main

    db.init("")  # fresh in-memory database
    monkeypatch.setattr(settings, "auth_required", True)
    monkeypatch.setattr(settings, "rate_limit_per_hour", 3)
    monkeypatch.setattr(main, "run", fake_run)
    for limiter in (auth.login_failures, auth.signups, auth.asks):
        limiter.hits.clear()
    return TestClient(main.app)  # not entered as a context manager, so the corpus is not loaded


def signup(c, email="a@example.com", password="correct horse", name="Ann"):
    return c.post("/auth/signup", json={"email": email, "password": password, "name": name}, headers=H)


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_signup_login_logout_and_admin_flag(client):
    r = signup(client)
    assert r.status_code == 200 and r.json()["is_admin"] is True  # the first account is the admin
    assert client.get("/api/me").json()["user"]["email"] == "a@example.com"
    assert client.post("/auth/logout", headers=H).status_code == 200
    assert client.get("/api/me").json()["user"] is None
    assert client.post("/auth/login", json={"email": "A@Example.com", "password": "correct horse"}, headers=H).status_code == 200  # email is case-insensitive
    other = TestClient(client.app)
    assert signup(other, "b@example.com").json()["is_admin"] is False


def test_signup_validation_and_hashing(client):
    assert signup(client, email="nope").status_code == 400
    assert signup(client, password="short").status_code == 400
    assert signup(client).status_code == 200
    assert signup(TestClient(client.app)).status_code == 409  # duplicate email
    assert db.user_by_email("a@example.com")["pw_hash"].startswith("$argon2")  # never stored in plaintext


def test_wrong_password_and_throttling(client):
    signup(client)
    fresh = TestClient(client.app)
    for _ in range(8):
        assert fresh.post("/auth/login", json={"email": "a@example.com", "password": "wrong-password"}, headers=H).status_code == 401
    assert fresh.post("/auth/login", json={"email": "a@example.com", "password": "correct horse"}, headers=H).status_code == 429  # locked out, even with the right password


def test_csrf_header_is_required_for_unsafe_requests(client):
    assert client.post("/auth/login", json={"email": "a@example.com", "password": "x"}).status_code == 403
    signup(client)
    assert client.post("/api/ask", json={"question": "q"}).status_code == 403


def test_endpoints_need_sign_in(client):
    for path in ("/api/status", "/api/chats", "/api/gptzero/scan"):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/ask", json={"question": "q"}, headers=H).status_code == 401


def test_ask_is_saved_and_replayable(client):
    signup(client)
    resp = client.post("/api/ask", json={"question": "What was TD's CET1?"}, headers=H)
    evs = events(resp)
    assert evs[0]["type"] == "chat"  # the client learns the chat id first
    chats = client.get("/api/chats").json()
    assert len(chats) == 1 and chats[0]["title"] == "What was TD's CET1?" and chats[0]["runs"] == 1
    saved = client.get(f"/api/chats/{chats[0]['id']}").json()["runs"][0]
    assert saved["answer"] == "TD's CET1 was 14.3% [1]."  # tokens folded into one answer
    assert [e["type"] for e in saved["events"]] == ["stage", "plan", "citations", "done"]  # tokens are not stored twice
    # a follow-up joins the same chat
    client.post("/api/ask", json={"question": "and ROE?", "chat_id": chats[0]["id"]}, headers=H)
    assert client.get("/api/chats").json()[0]["runs"] == 2


def test_chats_are_private_to_their_owner(client):
    signup(client)
    client.post("/api/ask", json={"question": "mine"}, headers=H)
    cid = client.get("/api/chats").json()[0]["id"]
    other = TestClient(client.app)
    signup(other, "b@example.com")
    assert other.get("/api/chats").json() == []
    assert other.get(f"/api/chats/{cid}").status_code == 404
    assert other.delete(f"/api/chats/{cid}", headers=H).status_code == 404
    assert other.post("/api/ask", json={"question": "x", "chat_id": cid}, headers=H).status_code == 404  # can't write into it either
    assert client.delete(f"/api/chats/{cid}", headers=H).status_code == 200
    assert client.get("/api/chats").json() == []


def test_share_link_is_public_and_revocable(client):
    signup(client)
    client.post("/api/ask", json={"question": "shared one"}, headers=H)
    cid = client.get("/api/chats").json()[0]["id"]
    token = client.post(f"/api/chats/{cid}/share", headers=H).json()["token"]
    anon = TestClient(client.app)
    view = anon.get(f"/api/shared/{token}")
    assert view.status_code == 200 and view.json()["title"] == "shared one" and view.json()["runs"][0]["answer"]
    assert anon.get("/api/chats").status_code == 401  # a share link grants only that chat
    client.delete(f"/api/chats/{cid}/share", headers=H)
    assert anon.get(f"/api/shared/{token}").status_code == 404


def test_only_admins_can_load_datasets_or_scan(client):
    signup(client)  # admin
    other = TestClient(client.app)
    signup(other, "b@example.com")
    assert other.post("/api/ingest", json={"mcp_url": "http://x"}, headers=H).status_code == 403
    assert other.post("/api/gptzero/scan", headers=H).status_code == 403
    assert client.post("/api/ingest", json={"mcp_url": "file:///etc/passwd"}, headers=H).status_code == 400  # admin, but only http(s) URLs


def test_rate_limit_per_user(client):
    signup(client)
    codes = [client.post("/api/ask", json={"question": f"q{i}"}, headers=H).status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
