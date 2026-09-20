from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from app import auth, db, integrations
from app.config import settings

H = {"X-Requested-With": "fetch"}


class FakeComposio:
    """Just enough of the Composio SDK: connections, auth configs and tool execution."""

    def __init__(self):
        self.calls, self.accounts = [], []
        self.connected_accounts = NS(list=self._list, link=self._link, delete=self._delete)
        self.auth_configs = NS(list=lambda: NS(items=[]), create=lambda toolkit, options: NS(id=f"ac_{toolkit}"))
        self.tools = NS(execute=self._execute)

    def connect(self, user_id, slug):
        self.accounts.append(NS(id=f"ca_{slug}_{len(self.accounts)}", user_id=user_id, toolkit=NS(slug=slug)))

    def _list(self, user_ids=None, toolkit_slugs=None, statuses=None):
        return NS(items=[a for a in self.accounts if a.user_id in (user_ids or [a.user_id])])

    def _link(self, user_id, auth_config_id, callback_url=None):
        self.calls.append(("link", user_id, auth_config_id, callback_url))
        return NS(redirect_url=f"https://connect.composio.dev/link/{auth_config_id}", id="ca_pending")

    def _delete(self, account_id):
        self.accounts = [a for a in self.accounts if a.id != account_id]

    def _execute(self, tool, args, user_id=None):
        self.calls.append((tool, user_id, args))
        slug = tool.split("_")[0].lower()
        slug = {"googledocs": "googledocs", "googleslides": "googleslides"}.get(slug, slug)
        if not any(a.user_id == user_id and a.toolkit.slug == slug for a in self.accounts):
            raise Exception("Error code: 404 - {'error': {'message': 'No connected account found', 'slug': 'ActionExecute_ConnectedAccountNotFound'}}")
        data = {
            "SLACK_LIST_ALL_CHANNELS": {"channels": [{"id": "C2", "name": "research"}, {"id": "C1", "name": "alerts", "is_private": True}]},
            "NOTION_SEARCH_NOTION_PAGE": {"results": [{"id": "p1", "properties": {"Name": {"type": "title", "title": [{"plain_text": "Team wiki"}]}}}]},
            "NOTION_CREATE_NOTION_PAGE": {"id": "newpage", "url": "https://notion.so/newpage"},
            "GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN": {"response_data": {"documentId": "DOC1"}},
            "GOOGLESLIDES_CREATE_SLIDES_MARKDOWN": {"presentationId": "DECK1"},
        }.get(tool, {})
        return {"successful": True, "data": data, "error": None}


@pytest.fixture
def env(monkeypatch):
    import app.main as main

    db.init("")
    fake = FakeComposio()
    monkeypatch.setattr(settings, "auth_required", True)
    monkeypatch.setattr(settings, "composio_api_key", "test-key")
    monkeypatch.setattr(integrations, "_composio", fake)
    integrations._auth_ids.clear()
    for lim in (auth.login_failures, auth.signups, auth.asks, integrations.sends):
        lim.hits.clear()
    client = TestClient(main.app)
    user = client.post("/auth/signup", json={"email": "a@example.com", "password": "correct horse", "name": "Ann"}, headers=H).json()
    return client, fake, user


def uid(user):
    return f"signal:{user['id']}"


def test_needs_sign_in_and_the_csrf_header(env):
    client, fake, user = env
    anon = TestClient(client.app)
    assert anon.get("/api/integrations").status_code == 401
    assert client.post("/api/integrations/slack/connect").status_code == 403  # no X-Requested-With header
    assert client.post("/api/integrations/slack/send", json={"markdown": "x", "target": "C1"}).status_code == 403


def test_status_reflects_connections_and_is_per_user(env):
    client, fake, user = env
    apps = {a["slug"]: a["connected"] for a in client.get("/api/integrations").json()["apps"]}
    assert apps == {"slack": False, "notion": False, "googledocs": False, "googleslides": False}
    fake.connect(uid(user), "slack")
    fake.connect("signal:someone-else", "notion")  # another user's connection must not show up here
    apps = {a["slug"]: a["connected"] for a in client.get("/api/integrations").json()["apps"]}
    assert apps["slack"] is True and apps["notion"] is False


def test_connect_returns_the_hosted_link_for_this_user(env):
    client, fake, user = env
    r = client.post("/api/integrations/notion/connect", headers=H)
    assert r.json()["url"].startswith("https://connect.composio.dev/")
    call = next(c for c in fake.calls if c[0] == "link")
    assert call[1] == uid(user) and call[3].endswith("#app")
    assert client.post("/api/integrations/nope/connect", headers=H).status_code == 404


def test_disconnect_removes_only_that_app(env):
    client, fake, user = env
    fake.connect(uid(user), "slack"); fake.connect(uid(user), "notion")
    assert client.post("/api/integrations/slack/disconnect", headers=H).json() == {"removed": 1}
    apps = {a["slug"]: a["connected"] for a in client.get("/api/integrations").json()["apps"]}
    assert apps["slack"] is False and apps["notion"] is True


def test_targets_for_slack_and_notion(env):
    client, fake, user = env
    fake.connect(uid(user), "slack"); fake.connect(uid(user), "notion")
    assert client.get("/api/integrations/slack/targets").json()["targets"] == [{"id": "C1", "name": "🔒 alerts"}, {"id": "C2", "name": "#research"}]
    assert client.get("/api/integrations/notion/targets").json()["targets"] == [{"id": "p1", "name": "Team wiki"}]
    assert client.get("/api/integrations/googledocs/targets").json()["targets"] == []


def test_slack_send_uses_markdown_text_and_requires_a_channel(env):
    client, fake, user = env
    fake.connect(uid(user), "slack")
    assert client.post("/api/integrations/slack/send", json={"markdown": "hi"}, headers=H).status_code == 400  # no channel picked
    r = client.post("/api/integrations/slack/send", json={"markdown": "x" * 20000, "target": "C2"}, headers=H)
    assert r.status_code == 200 and r.json()["ok"]
    _, user_id, args = next(c for c in fake.calls if c[0] == "SLACK_SEND_MESSAGE")
    assert user_id == uid(user) and args["channel"] == "C2" and len(args["markdown_text"]) == integrations.SLACK_LIMIT


def test_not_connected_gives_a_clear_409(env):
    client, fake, user = env
    r = client.post("/api/integrations/googledocs/send", json={"title": "t", "markdown": "hello"}, headers=H)
    assert r.status_code == 409 and "Connect" in r.json()["detail"]


def test_google_docs_and_slides_return_links(env):
    client, fake, user = env
    fake.connect(uid(user), "googledocs"); fake.connect(uid(user), "googleslides")
    doc = client.post("/api/integrations/googledocs/send", json={"title": "Brief", "markdown": "# Brief\n\nText"}, headers=H).json()
    deck = client.post("/api/integrations/googleslides/send", json={"title": "Deck", "markdown": "# A\n\n---\n\n# B"}, headers=H).json()
    assert doc["url"] == "https://docs.google.com/document/d/DOC1/edit"
    assert deck["url"] == "https://docs.google.com/presentation/d/DECK1/edit"


def test_notion_send_creates_a_page_then_appends_blocks_in_batches_of_100(env):
    client, fake, user = env
    fake.connect(uid(user), "notion")
    md = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(150))
    r = client.post("/api/integrations/notion/send", json={"title": "Notes", "markdown": md, "target": "p1"}, headers=H).json()
    assert r["url"] == "https://notion.so/newpage"
    create = next(c for c in fake.calls if c[0] == "NOTION_CREATE_NOTION_PAGE")
    assert create[2] == {"title": "Notes", "parent_id": "p1"}
    appends = [c for c in fake.calls if c[0] == "NOTION_APPEND_BLOCK_CHILDREN"]
    assert [len(c[2]["children"]) for c in appends] == [100, 51] and appends[0][2]["block_id"] == "newpage"


def test_input_limits_and_disabled_state(env, monkeypatch):
    client, fake, user = env
    fake.connect(uid(user), "slack")
    assert client.post("/api/integrations/slack/send", json={"markdown": "   ", "target": "C1"}, headers=H).status_code == 400
    assert client.post("/api/integrations/slack/send", json={"markdown": "x" * 30001, "target": "C1"}, headers=H).status_code == 400
    assert client.post("/api/integrations/nope/send", json={"markdown": "x"}, headers=H).status_code == 404
    for _ in range(30):
        integrations.sends.hit(f"u{user['id']}")
    assert client.post("/api/integrations/slack/send", json={"markdown": "x", "target": "C1"}, headers=H).status_code == 429
    monkeypatch.setattr(settings, "composio_api_key", "")
    assert client.get("/api/integrations").json() == {"enabled": False, "apps": []}
    assert client.post("/api/integrations/slack/connect", headers=H).status_code == 503


def test_markdown_to_notion_blocks():
    blocks = integrations.md_to_notion_blocks("# Head\n\nSome **bold** and `code` text.\n\n- one\n- two\n\n1. first\n\n> quoted\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n---\n")
    kinds = [b["type"] for b in blocks]
    assert kinds == ["heading_1", "paragraph", "bulleted_list_item", "bulleted_list_item", "numbered_list_item", "quote", "code", "divider"]
    para = blocks[1]["paragraph"]["rich_text"]
    assert any(t.get("annotations", {}).get("bold") and t["text"]["content"] == "bold" for t in para)
    assert any(t.get("annotations", {}).get("code") for t in para)
    long = integrations.md_to_notion_blocks("y" * 4500)[0]["paragraph"]["rich_text"]
    assert [len(t["text"]["content"]) for t in long] == [2000, 2000, 500]  # Notion's per-text limit
