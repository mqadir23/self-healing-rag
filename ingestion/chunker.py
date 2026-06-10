"""
chunker.py — Hybrid text chunker for the Self-Healing RAG pipeline.

Strategy:
  1. Structure-aware: If the document has clear headings (markdown-style # or
     ALL-CAPS lines), split on those first, then recursively split large sections.
  2. Recursive character: For unstructured text, split by paragraphs → sentences
     → words with configurable overlap.

Each chunk inherits its parent document's metadata plus its own chunk_index.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with inherited document metadata and chunk-level info."""
    content: str
    metadata: dict = field(default_factory=dict)


# Heading patterns: Markdown headings or ALL-CAPS lines (min 4 chars)
_HEADING_PATTERN = re.compile(
    r"(?:^|\n)(?=#{1,6}\s.+|[A-Z][A-Z \-]{3,}\n)"
)


def _has_structure(text: str) -> bool:
    """Detect if the text has clear heading structure."""
    headings = re.findall(r"(?:^|\n)(#{1,6}\s.+|[A-Z][A-Z \-]{3,})(?:\n|$)", text)
    return len(headings) >= 2


def _split_by_structure(text: str) -> list[str]:
    """Split text into sections based on detected headings."""
    parts = _HEADING_PATTERN.split(text)
    # Filter out empty parts
    sections = [p.strip() for p in parts if p.strip()]
    return sections


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively split text using a hierarchy of separators:
    double newline (paragraphs) → single newline → sentence-ending punctuation → space.
    """
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that actually appears in the text
    chosen_sep = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    if chosen_sep is None:
        # No separator found — hard split
        return _hard_split(text, chunk_size, chunk_overlap)

    # Split by the chosen separator
    parts = text.split(chosen_sep)
    chunks = []
    current_chunk = ""

    for part in parts:
        # Add separator back (except for space which is implicit)
        candidate = part if chosen_sep == " " else part + chosen_sep

        if not current_chunk:
            current_chunk = candidate
        elif len(current_chunk) + len(candidate) <= chunk_size:
            current_chunk += candidate
        else:
            # Current chunk is full — save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of the previous chunk
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + candidate

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Recursively split any chunks that are still too large
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            final_chunks.extend(_recursive_split(chunk, chunk_size, chunk_overlap))
        else:
            final_chunks.append(chunk)

    return final_chunks


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Last-resort character-level split for text with no separators."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return [c for c in chunks if c]


def chunk_documents(
    documents: list,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Chunk a list of LoadedDocument objects into smaller pieces.

    Uses structure-aware splitting when headings are detected,
    otherwise falls back to recursive character splitting.

    Args:
        documents: List of LoadedDocument objects (from loader.py).
        chunk_size: Target maximum characters per chunk. Default 512.
        chunk_overlap: Number of overlapping characters between chunks. Default 100.

    Returns:
        A flat list of Chunk objects with content and metadata.
    """
    all_chunks = []

    for doc in documents:
        text = doc.content

        if not text.strip():
            continue

        # Choose splitting strategy
        if _has_structure(text):
            # Structure-aware: split by headings first, then recursive split large sections
            sections = _split_by_structure(text)
            raw_chunks = []
            for section in sections:
                if len(section) > chunk_size:
                    raw_chunks.extend(_recursive_split(section, chunk_size, chunk_overlap))
                else:
                    raw_chunks.append(section)
        else:
            # Pure recursive character splitting
            raw_chunks = _recursive_split(text, chunk_size, chunk_overlap)

        # Build Chunk objects with metadata
        for i, chunk_text in enumerate(raw_chunks):
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            }
            all_chunks.append(Chunk(content=chunk_text, metadata=chunk_metadata))

    print(f"[Chunker] Created {len(all_chunks)} chunk(s) from {len(documents)} document block(s)")
    return all_chunks
