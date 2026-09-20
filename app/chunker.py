"""Structure-aware markdown chunking.

Keeps the heading path, source line range, and table headers on every chunk so a citation can
point at an exact place in an exact document. Tables are indexed twice: once per row (precise
numeric lookups) and once whole (comparisons across rows).
"""
import re
from dataclasses import asdict, dataclass

MAX_TEXT_CHARS = 1400
MAX_TABLE_CHARS = 6000
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(\{#[^}]*\})?\s*$")
ENTITY_RE = re.compile(r"^([A-Z][A-Z0-9.]{0,7})\b\s*[—–-]")
PAREN_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.]{0,7})\)")  # "Royal Bank of Canada (RY)"
PERIOD_LIKE_RE = re.compile(r"(FY|CY|Q|H)\d")
NAME_SPLIT_RE = re.compile(r"\s+[—–]\s+|\s+-\s+")
SUMMARY_RE = re.compile(r"^\s*<summary>(.*?)</summary>\s*$")
DETAILS_TAG_RE = re.compile(r"^\s*</?details>\s*$")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    id: str
    doc: str
    doc_title: str
    section: str
    entity: str
    kind: str  # text | table_row | table
    text: str
    start_line: int
    end_line: int
    period: str = ""

    @property
    def embed_text(self) -> str:
        head = f"{self.doc_title} > {self.section}"
        if self.entity:
            head += f" [{self.entity}]"
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


def _heading_key(title: str) -> str:
    """Leading name of a heading without trailing qualifiers: "RY — Royal Bank — Quarterly" -> "RY"."""
    return NAME_SPLIT_RE.split(title)[0].strip()


def _repeating_keys(lines: list[str]) -> set[str]:
    """H3 headings that recur under two or more different H2 sections. Reports that describe the same
    companies in several sections (deep dive, quarterly history, ...) repeat each company's heading, so
    repetition identifies entities without depending on how the headings are worded."""
    h2, seen = "", {}
    for line in lines:
        m = HEADING_RE.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2)
        if level == 2:
            h2 = title
        elif level == 3:
            seen.setdefault(_heading_key(title), set()).add(h2)
    return {k for k, sections in seen.items() if len(sections) >= 2 and not PERIOD_LIKE_RE.match(k)}


def _entity_from_heading(title: str, repeating: set[str]) -> str:
    """Entity id for an H3 heading, trying the most specific convention first."""
    em = ENTITY_RE.match(title)  # "RY — Royal Bank of Canada"
    if em and not PERIOD_LIKE_RE.match(em.group(1)):
        return em.group(1)
    pm = PAREN_TICKER_RE.search(title)  # "Royal Bank of Canada (RY)"
    if pm and not PERIOD_LIKE_RE.match(pm.group(1)):
        return pm.group(1)
    key = _heading_key(title)  # "Royal Bank of Canada" repeated across sections
    return key if key in repeating else ""


def chunk_document(filename: str, content: str) -> list[Chunk]:
    lines = content.splitlines()
    repeating = _repeating_keys(lines)
    dslug = slug(filename.rsplit(".", 1)[0])
    doc_title = next((HEADING_RE.match(l).group(2) for l in lines if HEADING_RE.match(l) and l.startswith("# ")), filename)
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []
    entity = ""
    period = ""  # current <details><summary> label, e.g. "Q2 FY2026 (Apr 30, 2026) — Full Summary"
    buf: list[tuple[int, str]] = []  # (line_no, text)

    def section() -> str:
        parts = [t for _, t in path] or [doc_title]
        if period:
            parts.append(period)
        return " > ".join(parts)

    def add(kind: str, text: str, start: int, end: int, row_entity: str = ""):
        text = text.strip()
        if not text:
            return
        cid = f"{dslug}:{start}-{end}" + ("" if kind != "table_row" else f"#{len(chunks)}")
        chunks.append(Chunk(cid, filename, doc_title, section(), row_entity or entity, kind, text, start, end, period))

    def flush_text():
        nonlocal buf
        if not buf:
            return
        cur: list[tuple[int, str]] = []
        size = 0
        for ln, txt in buf:
            if size + len(txt) > MAX_TEXT_CHARS and cur:
                add("text", "\n".join(t for _, t in cur), cur[0][0], cur[-1][0])
                cur, size = [], 0
            cur.append((ln, txt))
            size += len(txt) + 1
        if cur:
            add("text", "\n".join(t for _, t in cur), cur[0][0], cur[-1][0])
        buf = []

    i = 0
    while i < len(lines):
        line = lines[i]
        ln = i + 1
        sm = SUMMARY_RE.match(line)
        if sm:
            flush_text()
            period = TAG_RE.sub("", sm.group(1)).strip()
            i += 1
            continue
        if DETAILS_TAG_RE.match(line):
            flush_text()
            if "/" in line:
                period = ""
            i += 1
            continue
        m = HEADING_RE.match(line)
        if m:
            flush_text()
            period = ""
            level, title = len(m.group(1)), m.group(2)
            path = [(l, t) for l, t in path if l < level] + [(level, title)]
            if level == 3:
                entity = _entity_from_heading(title, repeating)
            elif level <= 2:
                entity = ""
            i += 1
            continue
        if _is_table_line(line):
            flush_text()
            j = i
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            table = lines[i:j]
            header = _split_row(table[0])
            body = [(i + 1 + k, r) for k, r in enumerate(table) if k > 0 and not _is_separator(r)]
            whole = "\n".join(table)
            add("table", whole[:MAX_TABLE_CHARS], ln, j)
            for rln, row in body:
                cells = _split_row(row)
                pairs = "; ".join(f"{h}: {c}" for h, c in zip(header, cells) if c)
                first = cells[0] if cells else ""
                row_entity = first if re.fullmatch(r"[A-Z][A-Z0-9.]{0,7}", first) and not re.match(r"(FY|CY|Q|H)\d", first) else ""
                add("table_row", f"{pairs}\n(raw) {row.strip()}", rln, rln, row_entity)
            i = j
            continue
        if line.strip() and not line.strip().startswith("---"):
            buf.append((ln, line))
        elif not line.strip():
            if buf and sum(len(t) for _, t in buf) > MAX_TEXT_CHARS * 0.6:
                flush_text()
        i += 1
    flush_text()
    # A table's first-column ticker only counts as an entity if headings also identify it; otherwise
    # ids would mix tickers (rows) with names (headings) and split one company into two.
    known = {c.entity for c in chunks if c.kind == "text" and c.entity}
    if known:
        for c in chunks:
            if c.kind == "table_row" and c.entity and c.entity not in known:
                c.entity = ""
    return chunks


def chunk_documents(docs: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d["filename"], d["content"]))
    return out


def corpus_overview(docs: list[dict], chunks: list[Chunk]) -> dict:
    """Compact description of the dataset so the planner can adapt to unseen data."""
    overview = []
    for d in docs:
        cs = [c for c in chunks if c.doc == d["filename"]]
        sections, entities = [], []
        for c in cs:
            top = c.section.split(" > ")
            if len(top) > 1 and top[1] not in sections:
                sections.append(top[1])
            if c.entity and c.entity not in entities:
                entities.append(c.entity)
        overview.append(
            {
                "file": d["filename"],
                "title": cs[0].doc_title if cs else d["filename"],
                "sections": sections,
                "entities": entities,
                "chunks": len(cs),
            }
        )
    return {"documents": overview}
