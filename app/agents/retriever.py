import asyncio

from ..chunker import Chunk
from ..config import settings
from ..corpus import Corpus
from ..llm import chat_json, embed
from ..store import rrf
from .common import short_section

RERANK_SYSTEM = """You are a relevance judge for a financial-research search engine.
Given a search query and numbered candidate passages, return the indices of the passages that best help answer the query, most useful first.
Prefer passages that contain the exact metric, entity and period asked for, and passages that state the figure explicitly.
Include a passage from each distinct source section that states the same figure (so disagreements are visible).
Return JSON: {"ranked": [int, ...]} with at most %d indices."""


async def retrieve(corpus: Corpus, sub_query: dict, pool: int = 24, keep: int = 8) -> list[tuple[Chunk, float]]:
    query = sub_query["query"]
    entities = sub_query.get("entities") or None
    qvec = (await embed([query]))[0]

    searches = [asyncio.to_thread(corpus.store.search, query, qvec, pool, None)]
    if entities:  # entity-filtered search guarantees coverage; unfiltered one catches cross-entity context
        searches.append(asyncio.to_thread(corpus.store.search, query, qvec, pool, entities))
    results = await asyncio.gather(*searches)
    fused = rrf([[cid for cid, _ in r] for r in results])[:pool]
    candidates = [(corpus.by_id[cid], score) for cid, score in fused if cid in corpus.by_id]
    if len(candidates) <= keep:
        return candidates

    listing = "\n".join(f"{i}: ({short_section(c)}) {c.text[:380]}" for i, (c, _) in enumerate(candidates))
    try:
        out = await chat_json(RERANK_SYSTEM % keep, f"Query: {query}\n\nCandidates:\n{listing}", settings.fast_model)
        order = [i for i in out.get("ranked", []) if isinstance(i, int) and 0 <= i < len(candidates)]
        seen: list[int] = []
        for i in order:
            if i not in seen:
                seen.append(i)
        if seen:
            return [candidates[i] for i in seen[:keep]]
    except Exception:
        pass
    return candidates[:keep]
