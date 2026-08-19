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
    input_dir = root_dir / "input"
    directory = input_dir if input_dir.is_dir() else root_dir
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                records.append(DocumentRecord(path.stem, text, str(path.resolve())))
    return records
