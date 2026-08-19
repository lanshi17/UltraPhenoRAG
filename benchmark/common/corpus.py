"""Corpus preparation helpers shared by all baselines.

PDF extraction, canonical source IDs, and manifest writing used to be
duplicated in every baseline entry point; the canonical mapping and the
page-level extraction rules live here now.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "corpus_manifest.json"

CANONICAL_SOURCE_FILES: dict[str, str] = {
    "ISUOG_2020_fetal-CNS-part1.pdf": "ISUOG-cns-2020",
    "ISUOG_2022_routine-mid-trimester-scan.pdf": "ISUOG-midtrimester-2022",
    "ISUOG_2023_11-14-week-ultrasound-scan.pdf": "ISUOG-11-14w-2023",
    "ISUOG_2023_fetal-cardiac-screening.pdf": "ISUOG-fetal-cardiac-screening-2023",
    "ISUOG-Practice-Guidelines-CNS-part-1-targeted-neurosonography.pdf": (
        "ISUOG-cns-2020"
    ),
    "ISUOG-Practice-Guidelines-Updated-performance-of-11-14-week-ultrasound-scan.pdf": (
        "ISUOG-11-14w-2023"
    ),
    "UOG-2023-Carvalho-ISUOG-Practice-Guidelines-updated-fetal-cardiac-screening.pdf": (
        "ISUOG-fetal-cardiac-screening-2023"
    ),
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def safe_slug(value: str, max_length: int = 120) -> str:
    """ASCII slug usable as a filename fragment for any baseline."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return (slug or "document")[:max_length]


def canonical_source_id(pdf_path: Path) -> str:
    """Map a known corpus PDF to the stable dataset source ID."""
    return CANONICAL_SOURCE_FILES.get(pdf_path.name, safe_slug(pdf_path.stem))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(pdf_path: Path) -> tuple[str, int, list[str]]:
    """Extract page-tagged text from a PDF using pypdf.

    Returns ``(text, page_count, warnings)``.  Per-page failures are collected
    as warnings instead of aborting the whole document.
    """
    from pypdf import PdfReader

    reader: Any
    try:
        reader = PdfReader(str(pdf_path), strict=False)
    except Exception as exc:  # noqa: BLE001
        return "", 0, [f"open: {type(exc).__name__}: {exc}"]

    page_sections: list[str] = []
    warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"page {page_number}: {type(exc).__name__}: {exc}")
            continue
        text = text.replace("\x00", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
        if text:
            page_sections.append(f"## Page {page_number}\n\n{text}")

    return "\n\n".join(page_sections), len(reader.pages), warnings


def source_id_from_text(text: str) -> str | None:
    """Recover the ``SOURCE_ID`` header written during corpus preparation."""
    match = re.search(r"(?m)^SOURCE_ID:\s*(\S+)\s*$", text)
    return match.group(1) if match else None


def write_manifest(project_dir: Path, manifest: dict) -> Path:
    """Persist the corpus manifest atomically enough for benchmark use."""
    manifest_path = project_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path
