from app.chunker import chunk_document
from app.mcp_client import extract_documents
from app.store import rrf

DOC = """# Sector Report

## 3. Snapshot

| Ticker | Company | ROE |
|---|---|---|
| RY | Royal Bank | 17.9% |
| TD | TD Bank | 16.0% |

## 6. History

### RY — Royal Bank

<details>
<summary><b>Q2 FY2026 (Apr 30, 2026) — Full Summary</b></summary>

**Balance sheet & capital:**
CET1 ratio of 13.5%.

</details>

<details>
<summary><b>Q1 FY2026 (Jan 31, 2026) — Full Summary</b></summary>

**Balance sheet & capital:**
CET1 ratio of 13.2%.

</details>
"""


def test_table_rows_carry_headers_entity_and_line_numbers():
    rows = [c for c in chunk_document("r.md", DOC) if c.kind == "table_row"]
    assert len(rows) == 2
    ry = rows[0]
    assert ry.entity == "RY"
    assert "ROE: 17.9%" in ry.text and "Company: Royal Bank" in ry.text
    assert DOC.splitlines()[ry.start_line - 1].startswith("| RY")


def test_period_is_tracked_and_details_tags_do_not_leak():
    text = [c for c in chunk_document("r.md", DOC) if c.kind == "text"]
    assert all("<details>" not in c.text and "<summary>" not in c.text for c in text)
    q2 = next(c for c in text if "13.5%" in c.text)
    q1 = next(c for c in text if "13.2%" in c.text)
    assert q2.period.startswith("Q2 FY2026") and "Q2 FY2026" in q2.section
    assert q1.period.startswith("Q1 FY2026")
    assert q2.entity == "RY"


def test_extract_documents_structured_and_plain():
    structured = {"structuredContent": {"documents": [{"filename": "a.md", "content": "# A"}]}}
    assert extract_documents("t", structured) == [{"filename": "a.md", "content": "# A"}]
    plain = {"content": [{"type": "text", "text": "hello"}]}
    assert extract_documents("t", plain) == [{"filename": "t-0.md", "content": "hello"}]


def test_number_audit_separates_grounded_derived_and_invented():
    from app.agents.writer import check_numbers
    from app.chunker import Chunk

    ev = [Chunk("a", "d.md", "T", "S", "X", "text", "Gross margin was 16.7% in Q3 2024 and 46.3% in Q2 2026. EPS $4.28", 1, 1)]
    ungrounded, derived = check_numbers("Margin rose from 16.7% to 46.3%, a 29.6% change; EPS was $4.28 and $9.99 [1]", ev)
    assert derived == ["29.6%"]
    assert ungrounded == ["$9.99"]


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert {fused[0][0], fused[1][0]} == {"a", "b"}
    assert fused[-1][0] in {"c", "d"}


def _entities(doc: str) -> dict:
    from app.chunker import chunk_document

    return {c.entity for c in chunk_document("r.md", doc) if c.kind == "text" and c.entity}


def _report(h3_a: str, h3_b: str) -> str:
    return f"""# Sector Report

## 5. Deep Dives

### {h3_a}

Thesis: strong quarter with record revenue and margin expansion across every segment of the business.

### {h3_b}

Thesis: weaker quarter with declining revenue and margin compression across the business.

## 6. Quarterly History

### {h3_a}

Revenue grew twelve percent year over year on strong demand and pricing power in the core franchise.

### {h3_b}

Revenue fell four percent year over year on softer demand and higher costs in the core franchise.
"""


def test_entities_are_detected_whatever_the_heading_style():
    assert _entities(_report("RY — Royal Bank", "TD — TD Bank")) == {"RY", "TD"}  # ticker first
    assert _entities(_report("Royal Bank (RY)", "TD Bank (TD)")) == {"RY", "TD"}  # ticker in parentheses
    assert _entities(_report("Royal Bank", "TD Bank")) == {"Royal Bank", "TD Bank"}  # name only, repeated across sections


def test_one_off_headings_and_periods_are_not_entities():
    doc = "# R\n\n## A\n\n### Overview\n\nSome long enough narrative text about the whole sector and its outlook for next year.\n\n### FY2027 — Outlook\n\nMore long enough narrative text about the next fiscal year and what management expects.\n"
    assert _entities(doc) == set()
