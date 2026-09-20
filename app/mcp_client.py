"""Minimal MCP (streamable HTTP) client that pulls markdown documents from any MCP server.

Deliberately schema-agnostic so a Phase 2 dataset served through a different tool set still
ingests: every tool that needs no required arguments is called, and whatever documents come
back (structuredContent.documents, or plain text content) are normalised to {filename, content}.
"""
import json

import httpx

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def parse_response(resp: httpx.Response) -> dict:
    """A streamable-HTTP reply is either plain JSON or an SSE stream of `data:` lines."""
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        last = None
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    last = json.loads(payload)
        if last is None:
            raise RuntimeError("MCP server returned an empty event stream")
        return last
    return resp.json()


def extract_documents(tool_name: str, result: dict) -> list[dict]:
    """Normalise a tools/call result into a list of {filename, content} documents."""
    structured = result.get("structuredContent") or {}
    docs = structured.get("documents")
    if isinstance(docs, list) and docs:
        out = []
        for i, d in enumerate(docs):
            content = d.get("content") or d.get("text") or ""
            name = d.get("filename") or d.get("name") or f"{tool_name}-{i}.md"
            if content.strip():
                out.append({"filename": name, "content": content})
        if out:
            return out
    out = []
    for i, part in enumerate(result.get("content") or []):
        text = part.get("text") if isinstance(part, dict) else None
        if text and text.strip():
            out.append({"filename": f"{tool_name}-{i}.md", "content": text})
    return out


def fetch_documents(url: str, timeout: float = 120.0) -> list[dict]:
    with httpx.Client(timeout=timeout) as client:
        init = client.post(
            url,
            headers=HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "signal-agent", "version": "0.1"},
                },
            },
        )
        init.raise_for_status()
        headers = dict(HEADERS)
        sid = init.headers.get("mcp-session-id")
        if sid:
            headers["mcp-session-id"] = sid
        client.post(url, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

        listing = parse_response(
            client.post(url, headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        )
        tools = listing.get("result", {}).get("tools", [])

        documents: list[dict] = []
        seen = set()
        for i, tool in enumerate(tools):
            if tool.get("inputSchema", {}).get("required"):
                continue  # can't call blindly
            resp = parse_response(
                client.post(
                    url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 10 + i,
                        "method": "tools/call",
                        "params": {"name": tool["name"], "arguments": {}},
                    },
                )
            )
            result = resp.get("result", {})
            if result.get("isError"):
                continue
            for doc in extract_documents(tool["name"], result):
                if doc["filename"] not in seen:
                    seen.add(doc["filename"])
                    documents.append(doc)
        if not documents:
            raise RuntimeError(f"No documents returned by any tool at {url}")
        return documents
