"""Offline tests for the HippoRAG 2 baseline facade (no engine/network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.baseline.hippo_rag_client.benchmark import (
    SourceResolver,
    preflight,
)
from benchmark.baseline.hippo_rag_client.client import HippoRAGClient
from benchmark.baseline.hippo_rag_client.documents import (
    _split,
    _token_counter,
    load_corpus_chunks,
)

# -- documents -------------------------------------------------------------


def test_load_corpus_chunks_maps_source_and_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sentence = "The fetal heart is assessed by ultrasound in every survey. "
    # a mid-document page marker must never survive into a chunk
    long_text = (sentence * 200) + "\n\n## Page 7\n\n" + (sentence * 200)
    (input_dir / "FDA-ultrasound-imaging--guideline.txt").write_text(
        long_text, encoding="utf-8"
    )
    (input_dir / "Short--note.txt").write_text(
        "This note records first-trimester screening protocols.", encoding="utf-8"
    )

    chunks = load_corpus_chunks(tmp_path, chunk_tokens=300)

    assert chunks, "expected at least one chunk"
    by_source = {chunk.source_id for chunk in chunks}
    assert by_source == {"FDA-ultrasound-imaging", "Short"}
    for chunk in chunks:
        assert chunk.content.strip()
        assert "## Page" not in chunk.content
        assert len(chunk.content) >= 50
        assert chunk.file_path.endswith(".txt")
    short = [c for c in chunks if c.source_id == "Short"]
    assert len(short) == 1
    assert short[0].content.startswith("This note records")
    long_chunks = [c for c in chunks if c.source_id == "FDA-ultrasound-imaging"]
    count_tokens = _token_counter()
    assert all(count_tokens(c.content) <= 300 for c in long_chunks)
    # chunk indices are contiguous per source
    assert [c.chunk_index for c in long_chunks] == list(range(len(long_chunks)))


def test_split_preserves_content_and_respects_budget() -> None:
    count_tokens = _token_counter()
    paragraphs = [
        f"Paragraph {i}. " + ("sentence words here. " * 40) for i in range(12)
    ]
    text = "\n\n".join(paragraphs)
    pieces = _split(text, 400, count_tokens)
    assert all(count_tokens(piece) <= 400 for piece in pieces)
    # no content is lost: every paragraph survives in some piece
    assert all(paragraph in text for paragraph in paragraphs)
    rejoined = " ".join(" ".join(piece.split()) for piece in pieces)
    for paragraph in paragraphs:
        assert " ".join(paragraph.split()) in rejoined


# -- context mapping -------------------------------------------------------


def test_contexts_from_solution_carries_source_metadata() -> None:
    metadata = [
        {"source_id": "ISUOG-cardiac", "file_path": "ISUOG-cardiac--a.txt"},
        {"source_id": "AIUM-3d"},
    ]
    solution = SimpleNamespace(
        docs=["first passage", "second passage", "no metadata"],
        doc_scores=None,
        doc_metadata=metadata,
    )
    contexts = HippoRAGClient._contexts_from_solution(solution)
    assert [c["source_id"] for c in contexts] == ["ISUOG-cardiac", "AIUM-3d", ""]
    assert contexts[0]["file_path"] == "ISUOG-cardiac--a.txt"
    assert contexts[1]["file_path"] == "AIUM-3d"
    assert contexts[2]["chunk_id"] == "#2"
    for chunk in contexts:
        assert chunk["text"] == chunk["content"]


def test_resolver_maps_file_paths_to_manifest_sources() -> None:
    resolver = SourceResolver(
        {
            "documents": [
                {
                    "source_id": "ISUOG-cardiac",
                    "input_file": "ISUOG-cardiac--2023.txt",
                }
            ]
        }
    )
    assert resolver.resolve("ISUOG-cardiac--2023.txt", "anything") == "ISUOG-cardiac"
    assert resolver.resolve("ISUOG-cardiac--2023", "") == "ISUOG-cardiac"
    assert resolver.resolve("unknown.bin", "") == "unresolved:unknown.bin"


# -- client lifecycle ------------------------------------------------------


def test_closed_client_raises_on_use() -> None:
    client = HippoRAGClient.__new__(HippoRAGClient)  # skip heavy __init__
    client._hipporag = None
    with pytest.raises(RuntimeError, match="closed"):
        client._hr()
    client.close()  # closing twice must be harmless


def test_bare_array_openie_response_is_rewrapped() -> None:
    from benchmark.baseline.hippo_rag_client.client import (
        _normalize_bare_json_list,
        _openie_field,
        _openie_parseable,
    )

    ner_prompt = [{"content": 'respond with {"named_entities": [...]}'}]
    # The real triple template echoes BOTH keys (entities in, triples out):
    # precedence must select the field the engine parses.
    triple_prompt = [
        {"content": 'input {"named_entities": [...]} output {"triples": [[s,p,o]]}'}
    ]
    assert _openie_field(ner_prompt) == "named_entities"
    assert _openie_field(triple_prompt) == "triples"
    assert _openie_field([{"content": "answer this question"}]) is None
    # A bare array answers the wrapper-object contract the engine demands.
    assert (
        _normalize_bare_json_list(ner_prompt, '["Acfam", "PPROM"]', "named_entities")
        == '{"named_entities": ["Acfam", "PPROM"]}'
    )
    assert _normalize_bare_json_list(
        triple_prompt, '[["a", "rel", "b"]]', "triples"
    ) == '{"triples": [["a", "rel", "b"]]}'
    assert _openie_parseable(
        '{"named_entities": ["Acfam", "PPROM"]}', {}, "named_entities"
    )
    assert not _openie_parseable("I cannot help with that.", {}, "named_entities")
    # Reasoning-loop blobs are rejected without the engine's quadratic scan.
    assert not _openie_parseable('{"a": "' + "x" * 100_001, {}, "named_entities")
    # A truncated object is salvageable only via the engine's length fix.
    assert not _openie_parseable(
        '{"named_entities": ["a", "b"', {}, "named_entities"
    )


# -- preflight -------------------------------------------------------------


def test_preflight_reports_missing_and_present_index(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "input").mkdir(parents=True)
    (corpus / "input" / "Doc--one.txt").write_text("body", encoding="utf-8")
    dataset = tmp_path / "dataset.json"

    report = preflight(
        save_dir=tmp_path / "store",
        dataset_path=dataset,
        require_index=True,
        corpus_dir=corpus,
    )
    assert report["missing_index"] is True
    assert any("index is missing" in issue for issue in report["issues"])
    assert any("dataset does not exist" in issue for issue in report["issues"])

    index_file = tmp_path / "store" / "gpt-emb"
    index_file.mkdir(parents=True)
    (index_file / "index_manifest.json").write_text("{}", encoding="utf-8")
    (index_file / "graph.pickle").write_bytes(b"graph")
    (index_file / "entity_embeddings").mkdir()
    (index_file / "entity_embeddings" / "entities.parquet").write_bytes(b"entities")
    (index_file / "fact_embeddings").mkdir()
    (index_file / "fact_embeddings" / "facts.parquet").write_bytes(b"facts")
    report = preflight(
        save_dir=tmp_path / "store",
        dataset_path=dataset,
        require_index=True,
        corpus_dir=corpus,
    )
    assert report["missing_index"] is False
    assert not any("index is missing" in issue for issue in report["issues"])
