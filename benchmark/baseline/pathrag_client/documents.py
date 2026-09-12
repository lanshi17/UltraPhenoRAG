"""Document input normalization."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    text: str
    file_path: str


def load_documents(root_dir: Path, input_documents: Any = None) -> list[DocumentRecord]:
    """Load documents from an iterable, a DataFrame, or the filesystem.

    When ``input_documents`` is None the unified corpus input directory is
    preferred (see :func:`_resolve_input_directory`).
    """
    records: list[DocumentRecord] = []
    if input_documents is not None:
        rows = (
            (row.to_dict() for _, row in input_documents.iterrows())
            if hasattr(input_documents, "iterrows")
            else input_documents
            if isinstance(input_documents, Iterable)
            else []
        )
        for index, row in enumerate(rows):
            value = row if isinstance(row, Mapping) else {"text": str(row)}
            text = str(
                value.get("text", value.get("content", value.get("document", ""))) or ""
            ).strip()
            if not text:
                continue
            title = str(
                value.get("title", value.get("file_path", value.get("path", ""))) or ""
            )
            document_id = str(
                value.get("id", value.get("document_id", ""))
                or Path(title).stem
                or f"document-{index + 1}"
            )
            records.append(DocumentRecord(document_id, text, title or document_id))
        return records

    directory = _resolve_input_directory(root_dir)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                records.append(DocumentRecord(path.stem, text, str(path.resolve())))
    return records


def _resolve_input_directory(root_dir: Path) -> Path:
    """Locate the input directory, preferring the unified corpus layout.

    Resolution order:
    1. ``<root_dir>/input`` when it exists and contains text files
       (explicit per-baseline input wins, keeps tmp_path tests working).
    2. The shared corpus directory ``<repo>/benchmark/data/corpus/input``.
    3. ``root_dir`` itself (legacy behaviour for loose directories).
    """
    candidate = root_dir / "input"
    if candidate.is_dir() and any(candidate.glob("*.txt")):
        return candidate
    # root_dir 形如 .../benchmark/data/path_rag
    unified = root_dir.parent / "corpus" / "input"
    if unified.is_dir() and any(unified.glob("*.txt")):
        return unified
    return candidate if candidate.is_dir() else root_dir
