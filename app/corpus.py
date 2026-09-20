"""The indexed dataset: documents, chunks, embeddings and the search backend."""
import asyncio
import hashlib
import json

import numpy as np

from .chunker import Chunk, chunk_documents, corpus_overview
from .config import settings
from .llm import embed
from .mcp_client import fetch_documents
from .store import make_store


class Corpus:
    def __init__(self):
        self.docs: list[dict] = []
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.overview: dict = {"documents": []}
        self.store = None
        self.backend = "not loaded"
        self.notice: str | None = None
        self.source: str = ""

    async def load_from_mcp(self, url: str) -> None:
        docs = await asyncio.to_thread(fetch_documents, url)
        self.source = url
        await self.build(docs)

    async def build(self, docs: list[dict]) -> None:
        chunks = chunk_documents(docs)
        texts = [c.embed_text for c in chunks]
        key = hashlib.sha256(("\n".join(texts) + settings.embed_model).encode()).hexdigest()[:16]
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = settings.cache_dir / f"emb-{key}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = await embed(texts)
            np.save(cache, vecs)
        store, notice = make_store()
        await asyncio.to_thread(store.index, chunks, vecs)
        self.docs, self.chunks = docs, chunks
        self.by_id = {c.id: c for c in chunks}
        self.overview = corpus_overview(docs, chunks)
        self.store, self.backend, self.notice = store, store.name, notice

    def entity_name(self, entity: str) -> str:
        """Display name for an entity id, from its heading ("RY — Royal Bank of Canada")."""
        for c in self.chunks:
            if c.entity == entity:
                for part in c.section.split(" > "):
                    if part.startswith(f"{entity} —") or part.startswith(f"{entity} -"):
                        return part.split("—", 1)[-1].split(" - ", 1)[-1].split("—")[0].strip()
                    if f"({entity})" in part:  # "Royal Bank of Canada (RY)"
                        return part.split("(")[0].strip()
        return entity

    def source_context(self, chunk_id: str, radius: int = 3) -> dict | None:
        c = self.by_id.get(chunk_id)
        if not c:
            return None
        doc = next(d for d in self.docs if d["filename"] == c.doc)
        lines = doc["content"].splitlines()
        lo, hi = max(1, c.start_line - radius), min(len(lines), c.end_line + radius)
        # markdown to render: the passage itself, or for a table row the whole table so it shows with its header
        md_lo, md_hi = c.start_line, c.end_line
        if c.kind == "table_row":
            table = next((t for t in self.chunks if t.kind == "table" and t.doc == c.doc and t.start_line <= c.start_line <= t.end_line), None)
            if table:
                md_lo, md_hi = table.start_line, table.end_line
        return {
            "kind": c.kind,
            "markdown": "\n".join(lines[md_lo - 1 : md_hi]),
            "md_start": md_lo,
            "id": c.id,
            "doc": c.doc,
            "section": c.section,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "lines": [{"n": n, "text": lines[n - 1], "hit": c.start_line <= n <= c.end_line} for n in range(lo, hi + 1)],
        }

    def status(self) -> dict:
        return {
            "loaded": bool(self.chunks),
            "source": self.source,
            "backend": self.backend,
            "notice": self.notice,
            "documents": self.overview["documents"],
            "chunks": len(self.chunks),
        }
