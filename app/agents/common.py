from ..chunker import Chunk


def short_section(c: Chunk) -> str:
    """Section path without the document title, which is redundant next to the filename."""
    parts = c.section.split(" > ")
    return " > ".join(parts[1:]) if len(parts) > 1 else c.section


def format_evidence(evidence: list[Chunk], max_chars: int = 1600) -> str:
    blocks = []
    for n, c in enumerate(evidence, 1):
        body = c.text if len(c.text) <= max_chars else c.text[:max_chars] + " …"
        blocks.append(f"[{n}] {c.doc} L{c.start_line}-{c.end_line} | {short_section(c)}\n{body}")
    return "\n\n".join(blocks)
