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


def load_documents(
    root_dir: Path,
    input_documents: Any = None,
    *,
    input_dir: Path | None = None,
) -> list[DocumentRecord]:
    """Load documents from an iterable, a DataFrame, or the filesystem.

    When ``input_documents`` is None the unified corpus input directory is
    preferred (see :func:`_resolve_input_directory`).  An explicit
    ``input_dir`` takes precedence over the project and unified-corpus
    defaults, which lets benchmark callers point a project at an alternate
    corpus without copying documents into its storage directory.
    """
    records: list[DocumentRecord] = []
    if input_documents is not None:
        if hasattr(input_documents, "iterrows"):
            rows = (row.to_dict() for _, row in input_documents.iterrows())
        elif isinstance(input_documents, Mapping):
            rows = [input_documents]
        elif isinstance(input_documents, (str, bytes)):
            # Treat one string as one document rather than an iterable of
            # single-character documents.
            rows = [{"text": _as_text(input_documents)}]
        elif isinstance(input_documents, Iterable):
            rows = input_documents
        else:
            raise TypeError(
                "input_documents must be an iterable of documents, a mapping, "
                "a string, or a DataFrame-like object"
            )
        for index, row in enumerate(rows):
            value = (
                row
                if isinstance(row, Mapping)
                else {
                    "text": _as_text(row) if isinstance(row, (str, bytes)) else str(row)
                }
            )
            raw_text = value.get(
                "text", value.get("content", value.get("document", ""))
            )
            text = (
                _as_text(raw_text)
                if isinstance(raw_text, (str, bytes))
                else str(raw_text or "")
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

    directory = _resolve_input_directory(root_dir, input_dir=input_dir)
    if not directory.is_dir():
        return records
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                records.append(DocumentRecord(path.stem, text, str(path.resolve())))
    return records


def _resolve_input_directory(root_dir: Path, *, input_dir: Path | None = None) -> Path:
    """Locate the input directory, preferring the unified corpus layout.

    Resolution order:
    1. an explicit ``input_dir`` supplied by the caller;
    2. ``<root_dir>/input`` when it exists and contains text files
       (explicit per-baseline input wins, keeps tmp_path tests working).
    3. The shared corpus directory ``<repo>/benchmark/data/corpus/input``.
    4. ``root_dir`` itself (legacy behaviour for loose directories).
    """
    if input_dir is not None:
        return Path(input_dir).expanduser().resolve()
    candidate = root_dir / "input"
    if _has_supported_documents(candidate):
        return candidate
    # root_dir 形如 .../benchmark/data/kag
    unified = root_dir.parent / "corpus" / "input"
    if _has_supported_documents(unified):
        return unified
    # An empty per-project input directory should not mask loose documents in
    # root_dir.  This matters for callers that create an ``input/`` directory
    # as part of project setup but place temporary Markdown/TXT fixtures next
    # to it.
    return root_dir


def _has_supported_documents(directory: Path) -> bool:
    return directory.is_dir() and any(
        path.is_file() and path.suffix.casefold() in {".txt", ".md"}
        for path in directory.iterdir()
    )


def _as_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
