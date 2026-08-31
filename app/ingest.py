"""
Ingestion: parse knowledge-base markdown into metadata-tagged chunks.

Each file has YAML front matter (document_id, status, policy_authority,
audience, supersedes/superseded_by, effective_date, ...). We chunk by
H2 (##) heading so citations can point at a specific heading, not just
a filename.
"""
from __future__ import annotations

import re
import glob
import os
from dataclasses import dataclass, field

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    heading: str
    text: str
    doc_id: str
    status: str
    policy_authority: str
    audience: str
    title: str
    effective_date: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    extra: dict = field(default_factory=dict)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_text, body = m.group(1), m.group(2)
    meta = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    return meta, body


def _split_headings(body: str) -> list[tuple[str, str]]:
    """Split a document body into (heading, section_text) by H2 (##)."""
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # top-level title; skip storing as its own chunk, keep as context
            continue
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    out = []
    for heading, ls in sections:
        text = "\n".join(ls).strip()
        if text:
            out.append((heading, text))
    return out


def load_chunks(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = _parse_front_matter(raw)
        sections = _split_headings(body)
        doc_id = meta.get("document_id", filename)
        for i, (heading, text) in enumerate(sections):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i}::{heading}",
                    filename=filename,
                    heading=heading,
                    text=text,
                    doc_id=doc_id,
                    status=meta.get("status", "unknown"),
                    policy_authority=meta.get("policy_authority", "unknown"),
                    audience=meta.get("audience", "unknown"),
                    title=meta.get("title", filename),
                    effective_date=meta.get("effective_date"),
                    supersedes=meta.get("supersedes"),
                    superseded_by=meta.get("superseded_by"),
                    extra=meta,
                )
            )
    return chunks


if __name__ == "__main__":
    cs = load_chunks(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
    for c in cs:
        print(c.chunk_id, "|", c.status, c.policy_authority, c.audience)
