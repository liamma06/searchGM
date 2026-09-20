import asyncio
import json
from typing import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

from .config import settings

_client: AsyncOpenAI | None = None
_or_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def openrouter() -> AsyncOpenAI:
    global _or_client
    if _or_client is None:
        _or_client = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    return _or_client


def cross_model_available() -> bool:
    return bool(settings.openrouter_api_key)


async def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    provider: str = "openai",
    timeout: float | None = None,
    extra_body: dict | None = None,
) -> dict:
    api = openrouter() if provider == "openrouter" else client()
    kwargs = {}
    if timeout:
        kwargs["timeout"] = timeout
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = await api.chat.completions.create(
        model=model or settings.chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def chat_stream(system: str, user: str, model: str | None = None) -> AsyncIterator[str]:
    stream = await client().chat.completions.create(
        model=model or settings.chat_model,
        temperature=0,
        stream=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    async for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


async def embed(texts: list[str], batch: int = 128, concurrency: int = 4) -> np.ndarray:
    """Embed texts (order preserved). Vectors are L2-normalised so dot product == cosine."""
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: list[str]) -> list[list[float]]:
        async with sem:
            r = await client().embeddings.create(model=settings.embed_model, input=[t[:8000] for t in chunk])
            return [d.embedding for d in r.data]

    parts = await asyncio.gather(*[one(texts[i : i + batch]) for i in range(0, len(texts), batch)])
    mat = np.array([v for part in parts for v in part], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-9, None)
