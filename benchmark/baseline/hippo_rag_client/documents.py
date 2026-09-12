"""Document loading and token-window chunking for the HippoRAG baseline.

HippoRAG's default preprocessor keeps one chunk per document; guideline
files are far too long for that, so this module reuses the unified corpus
input and aggregates paragraphs into roughly token-sized chunks that carry
the manifest ``source_id`` (the same identifier the shared scorer matches
gold sources against).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Callable

from benchmark.common.unified_corpus import corpus_input_dir

_DEFAULT_CHUNK_TOKENS = 1200
# Markdown page-break markers (``## Page 7``) are layout noise: as standalone
# paragraphs they became 175 stub chunks that poisoned retrieval and the
# OpenIE stage (the model cannot extract entities from 9 characters).
_PAGE_MARKER = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*Page[ \t]+\d+[ \t]*$")
_PARAGRAPH_SEPARATORS = ("\n\n", "\n", ". ")
_MIN_CHUNK_CHARS = 50


def _token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(encoding.encode(text))

        return count
    except Exception:  # noqa: BLE001 - fall back to a rough estimate
        return lambda text: max(1, len(text) // 4)


@dataclass(frozen=True)
class CorpusChunk:
    """One token-window chunk tied to its unified-corpus source."""

    content: str
    source_id: str
    file_path: str
    chunk_index: int


def load_corpus_chunks(
    corpus_dir: Path,
    *,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
) -> list[CorpusChunk]:
    """Load the unified corpus and split each file into token windows.

    Splitting honours paragraph boundaries first and falls back to
    sentence boundaries, so chunks stay readable for the OpenIE stage.
    """

    count_tokens = _token_counter()
    input_dir = corpus_input_dir(corpus_dir)
    chunks: list[CorpusChunk] = []
    for path in sorted(input_dir.glob("*.txt")):
        source_id = path.stem.split("--", 1)[0]
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        text = re.sub(r"\n{3,}", "\n\n", _PAGE_MARKER.sub("", text)).strip()
        if not text:
            continue
        pieces = [
            piece
            for piece in _split(text, chunk_tokens, count_tokens)
            if len(piece) >= _MIN_CHUNK_CHARS
        ]
        for index, piece in enumerate(pieces):
            chunks.append(
                CorpusChunk(
                    content=piece,
                    source_id=source_id,
                    file_path=path.name,
                    chunk_index=index,
                )
            )
    if not chunks:
        raise ValueError(f"no supported documents under {input_dir}")
    return chunks


def _split(
    text: str,
    chunk_tokens: int,
    count_tokens: Callable[[str], int],
    _depth: int = 0,
) -> list[str]:
    if count_tokens(text) <= chunk_tokens:
        return [text]
    separator = _PARAGRAPH_SEPARATORS[min(_depth, len(_PARAGRAPH_SEPARATORS) - 1)]
    if separator not in text:
        return _hard_split(text, chunk_tokens, count_tokens)
    parts = text.split(separator)
    pieces: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{separator}{part}" if buffer else part
        if count_tokens(candidate) <= chunk_tokens:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer.strip())
            buffer = ""
        if count_tokens(part) <= chunk_tokens:
            buffer = part
        else:
            pieces.extend(
                _split(part, chunk_tokens, count_tokens, _depth + 1)
            )
    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece.strip()]


def _hard_split(text: str, chunk_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    words = text.split(" ")
    pieces: list[str] = []
    buffer: list[str] = []
    tokens = 0
    for word in words:
        word_tokens = count_tokens(word)
        if buffer and tokens + word_tokens > chunk_tokens:
            pieces.append(" ".join(buffer))
            buffer, tokens = [], 0
        buffer.append(word)
        tokens += word_tokens
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces
