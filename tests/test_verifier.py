import asyncio

from app.agents import verifier
from app.chunker import Chunk

EVIDENCE = [Chunk("a", "d.md", "T", "S", "RY", "text", "EPS adjusted 4.28, diluted 4.23; ROE 17.9%", 1, 1)]


def test_different_measures_are_not_reported_as_conflicts(monkeypatch):
    async def fake(system, user, model=None, provider="openai", timeout=None, extra_body=None):
        return {
            "answerable": True, "facts": [], "gaps": [], "follow_up_queries": [],
            "conflicts": [
                {"topic": "EPS", "values": [{"value": "$4.28 adjusted", "sources": [1]}, {"value": "$4.23 diluted", "sources": [1]}], "assessment": "", "different_measures": True},
                {"topic": "ROE", "values": [{"value": "17.9%", "sources": [1]}, {"value": "18.1%", "sources": [1]}], "assessment": "", "different_measures": False},
            ],
        }

    monkeypatch.setattr(verifier, "chat_json", fake)
    audit = asyncio.run(verifier.verify("q", EVIDENCE))
    assert [c["topic"] for c in audit["conflicts"]] == ["ROE"]  # the genuine disagreement stays


def test_identical_values_and_real_differences(monkeypatch):
    async def fake(system, user, model=None, provider="openai", timeout=None, extra_body=None):
        def c(topic, *vals):
            return {"topic": topic, "values": [{"value": v, "sources": [1]} for v in vals], "assessment": "", "different_measures": False}
        return {"answerable": True, "facts": [], "gaps": [], "follow_up_queries": [],
                "conflicts": [c("CIBC EPS", "$2.73", "$2.73"), c("TD ROE", "16.0%", "16%"), c("ROE", "17.9%", "18%"), c("EPS", "$4.28", "$4.23")]}

    monkeypatch.setattr(verifier, "chat_json", fake)
    audit = asyncio.run(verifier.verify("q", EVIDENCE))
    assert [c["topic"] for c in audit["conflicts"]] == ["ROE", "EPS"]  # same-value pairs dropped; 17.9% vs 18% still counts
