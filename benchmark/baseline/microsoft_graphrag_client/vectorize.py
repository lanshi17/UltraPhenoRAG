"""知识库语料向量化。

读取 GraphRAG 项目 ``input/`` 目录下的知识库文本，按 ``settings.yaml`` 的
分块配置切分为 token 块，使用配置的 embedding 模型（默认
``text-embedding-3-large``，可指向自定义 OpenAI-compatible endpoint）批量
向量化，并将结果保存到 ``benchmark/data/proceed``。

输出
----
- ``corpus_vectors.parquet``
    每行一个文本块：``source_id`` / ``chunk_index`` / ``token_count`` /
    ``text`` / ``embedding``。
- ``corpus_vectors.jsonl``
    与 parquet 相同内容的增量写入文件，用于中断后续传。
- ``vectorize_manifest.json``
    生成元信息（模型、维度、分块参数、文档/块数量等）。

用法
----
.. code-block:: bash

    uv run python -m benchmark.baseline.microsoft_graphrag_client.vectorize \\
        --project-dir benchmark/data/microsoft_graphrag \\
        --output-dir benchmark/data/proceed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import tiktoken

from benchmark.baseline.microsoft_graphrag_client.client import GraphRAGClient
from benchmark.baseline.microsoft_graphrag_client.config.llm_config import (
    LLMConfigOverrides,
    ModelConfigOverride,
)
from benchmark.baseline.microsoft_graphrag_client.utils.async_runner import (
    AsyncRunner,
)
from benchmark.common.unified_corpus import DEFAULT_CORPUS_DIR, read_corpus_documents
from benchmark.config import load_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "microsoft_graphrag"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "proceed"

MANIFEST_NAME = "vectorize_manifest.json"
VECTORS_NAME = "corpus_vectors.parquet"
JSONL_NAME = "corpus_vectors.jsonl"

_ENCODING_MODEL = "o200k_base"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _chunk_text(
    text: str,
    encoding: tiktoken.Encoding,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int]]:
    """按 token 滑窗切分文本，返回 [(chunk_text, token_count)]。"""
    tokens = encoding.encode(text)
    total = len(tokens)
    if total == 0:
        return []
    if total <= chunk_size:
        decoded = encoding.decode(tokens)
        return [(decoded, total)] if decoded.strip() else []

    step = max(chunk_size - overlap, 1)
    chunks: list[tuple[str, int]] = []
    seen: set[str] = set()
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        decoded = encoding.decode(tokens[start:end])
        if decoded.strip() and decoded not in seen:
            chunks.append((decoded, end - start))
            seen.add(decoded)
        if end >= total:
            break
        start += step
    return chunks


def _redact_error(exc: BaseException) -> str:
    message = str(exc)
    key = load_environment().api_key
    if key:
        message = message.replace(key, "<redacted>")
    return message


def vectorize_corpus(
    *,
    project_dir: Path,
    output_dir: Path,
    corpus_dir: Path | None = None,
    chunk_size: int = 1200,
    overlap: int = 100,
    batch_size: int = 16,
    limit: int | None = None,
    embedding_model: str | None = None,
    model_provider: str | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """向量化统一语料并保存到 output_dir。"""
    project_dir = project_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / JSONL_NAME
    parquet_path = output_dir / VECTORS_NAME
    manifest_path = output_dir / MANIFEST_NAME

    # 项目 .env 需在创建 embedding 前加载（自定义 endpoint / 密钥）。
    load_environment()
    llm_overrides = LLMConfigOverrides(
        embedding=ModelConfigOverride(
            model=embedding_model,
            model_provider=model_provider,
            api_base=api_base,
            api_key_env=api_key_env,
        )
    )
    client = GraphRAGClient(project_dir, llm_overrides=llm_overrides)
    model_config = client.config.embedding_models["default_embedding_model"]
    from graphrag_llm.embedding import create_embedding

    embedding = create_embedding(model_config)
    encoding = tiktoken.get_encoding(_ENCODING_MODEL)

    documents = read_corpus_documents(corpus_dir)
    if limit is not None:
        documents = documents[:limit]

    # 增量续传：跳过已完成的 source_id。
    existing: set[str] = set()
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing.add(str(json.loads(line).get("source_id", "")))

    api_base = getattr(model_config, "api_base", None) or ""
    embedding_model = str(model_config.model)
    dimension: int | None = None
    started = time.monotonic()
    document_stats: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_chunks = 0
    total_tokens = 0

    with jsonl_path.open("a", encoding="utf-8") as jsonl_stream:
        for index, (source_id, path, text) in enumerate(documents, start=1):
            if source_id in existing:
                print(
                    f"[{index}/{len(documents)}] 跳过已完成 {source_id}",
                    flush=True,
                )
                continue
            doc_started = time.monotonic()
            chunks = _chunk_text(
                text,
                encoding,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            print(
                f"[{index}/{len(documents)}] 向量化 {source_id} ({len(chunks)} 块)",
                flush=True,
            )
            try:
                rows: list[dict[str, Any]] = []
                for batch_start in range(0, len(chunks), batch_size):
                    batch = chunks[batch_start : batch_start + batch_size]
                    batch_texts = [item[0] for item in batch]
                    response = AsyncRunner.run(
                        embedding.embedding_async(input=batch_texts)
                    )
                    for offset, item in enumerate(batch):
                        vector = response.data[offset].embedding
                        if dimension is None:
                            dimension = len(vector)
                        rows.append(
                            {
                                "source_id": source_id,
                                "chunk_index": batch_start + offset,
                                "token_count": item[1],
                                "text": item[0],
                                "embedding": vector,
                            }
                        )
                    total_tokens += sum(item[1] for item in batch)
                for row in rows:
                    jsonl_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_stream.flush()
                total_chunks += len(rows)
                document_stats.append(
                    {
                        "source_id": source_id,
                        "input_file": path.name,
                        "char_count": len(text),
                        "chunk_count": len(rows),
                        "elapsed_seconds": round(time.monotonic() - doc_started, 3),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "source_id": source_id,
                        "error_type": type(exc).__name__,
                        "error": _redact_error(exc),
                    }
                )
                print(
                    f"  错误 {source_id}: {type(exc).__name__}: "
                    f"{_redact_error(exc)[:300]}",
                    flush=True,
                )

    # 合并增量 JSONL 写出 parquet。
    records: list[dict[str, Any]] = []
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    frame = pd.DataFrame(records) if records else pd.DataFrame()
    if not frame.empty:
        frame.to_parquet(parquet_path, index=False)

    manifest = {
        "created_at": _now_iso(),
        "project_dir": str(project_dir),
        "corpus_dir": str((corpus_dir or DEFAULT_CORPUS_DIR).resolve()),
        "output_dir": str(output_dir),
        "embedding_model": embedding_model,
        "embedding_api_base": api_base,
        "embedding_dimension": dimension,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": overlap,
        "batch_size": batch_size,
        "encoding_model": _ENCODING_MODEL,
        "document_count": len(documents),
        "skipped_count": len(existing & {s for s, _, _ in documents}),
        "vectorized_document_count": len(document_stats),
        "chunk_count": total_chunks,
        "token_count": total_tokens,
        "documents": document_stats,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "vectors_path": str(parquet_path),
        "jsonl_path": str(jsonl_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将知识库语料切块并向量化，保存到 proceed 目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="GraphRAG 项目目录（含 settings.yaml，用于解析 embedding 配置）",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="统一语料目录（读取 corpus/input 下的文本）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="向量化结果输出目录",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="分块 token 数",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="相邻分块重叠 token 数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="单次 embedding 请求的文本块数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 个文档（调试用）",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="embedding 模型名（覆盖 settings.yaml，如 text-embedding-3-large）",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        help="embedding 供应商（如 openai/azure）",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="embedding API base URL（含 /v1）",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="embedding API key 环境变量名（如 RAG_API_KEY）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.overlap >= args.chunk_size:
        print(
            "错误: --overlap 必须小于 --chunk-size",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = vectorize_corpus(
            project_dir=args.project_dir,
            output_dir=args.output_dir,
            corpus_dir=args.corpus_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            limit=args.limit,
            embedding_model=args.embedding_model,
            model_provider=args.model_provider,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"向量化失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
