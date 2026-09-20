"""Hybrid search backends. Both return (chunk_id, fused_score) fused with Reciprocal Rank Fusion
over a BM25 ranking and a dense-vector ranking. ElasticStore is used when ELASTIC_URL is set;
LocalStore is an in-process fallback with the same interface."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .config import settings

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class LocalStore:
    name = "local (BM25 + vectors, RRF)"

    def __init__(self):
        self.ids: list[str] = []
        self.entities: list[str] = []
        self.docs: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.vecs: np.ndarray | None = None

    def index(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        self.ids = [c.id for c in chunks]
        self.entities = [c.entity for c in chunks]
        self.docs = [c.doc for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
        self.vecs = vecs

    def search(self, query: str, qvec: np.ndarray, k: int = 30, entities: list[str] | None = None, docs: list[str] | None = None):
        mask = np.ones(len(self.ids), dtype=bool)
        if entities:
            wanted = set(entities)
            mask &= np.array([e in wanted for e in self.entities])
        if docs:
            wanted_docs = set(docs)
            mask &= np.array([d in wanted_docs for d in self.docs])
        bm = self.bm25.get_scores(tokenize(query))
        dense = self.vecs @ qvec
        pool = 100
        bm_rank = [self.ids[i] for i in np.argsort(-np.where(mask, bm, -1e9))[:pool] if mask[i]]
        dn_rank = [self.ids[i] for i in np.argsort(-np.where(mask, dense, -1e9))[:pool] if mask[i]]
        return rrf([bm_rank, dn_rank])[:k]


class ElasticStore:
    def __init__(self):
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(settings.elastic_url, api_key=settings.elastic_api_key, request_timeout=60)
        self.index_name = settings.elastic_index
        info = self.es.info()
        self.name = f"elastic {info['version']['number']} (BM25 + kNN, RRF)"

    def index(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        from elasticsearch import helpers

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "section": {"type": "text"},
                    "doc": {"type": "keyword"},
                    "entity": {"type": "keyword"},
                    "kind": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": int(vecs.shape[1]),
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )
        actions = (
            {
                "_index": self.index_name,
                "_id": c.id,
                "_source": {
                    "text": c.embed_text,
                    "section": c.section,
                    "doc": c.doc,
                    "entity": c.entity,
                    "kind": c.kind,
                    "vector": vecs[i].tolist(),
                },
            }
            for i, c in enumerate(chunks)
        )
        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)

    def search(self, query: str, qvec: np.ndarray, k: int = 30, entities: list[str] | None = None, docs: list[str] | None = None):
        flt = ([{"terms": {"entity": entities}}] if entities else []) + ([{"terms": {"doc": docs}}] if docs else [])
        pool = 100
        bm = self.es.search(
            index=self.index_name,
            size=pool,
            query={
                "bool": {
                    "must": {"multi_match": {"query": query, "fields": ["text", "section^0.5"]}},
                    "filter": flt,
                }
            },
            source=False,
        )
        kn = self.es.search(
            index=self.index_name,
            size=pool,
            knn={
                "field": "vector",
                "query_vector": qvec.tolist(),
                "k": pool,
                "num_candidates": pool * 2,
                "filter": flt,
            },
            source=False,
        )
        return rrf([[h["_id"] for h in bm["hits"]["hits"]], [h["_id"] for h in kn["hits"]["hits"]]])[:k]


def make_store():
    """Prefer Elastic when configured and reachable; otherwise fall back to the local store."""
    if settings.elastic_url:
        try:
            return ElasticStore(), None
        except Exception as e:  # unreachable, bad key, license, etc.
            return LocalStore(), f"Elastic unavailable ({type(e).__name__}: {e}); using local store"
    return LocalStore(), "ELASTIC_URL not set; using local store"
