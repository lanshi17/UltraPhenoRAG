"""Compatibility shims for KAG 0.8's OpenAI SDK usage.

The vendored KAG release passes ``model=`` to the OpenAI Python SDK's
``AzureOpenAI`` constructors.  Modern SDK releases reject that argument: the
model belongs on each completions/embeddings request, where KAG already sends
it.  This small runtime-only patch also treats a bare Azure resource URL as an
``azure_endpoint`` so the SDK adds its required ``/openai`` path.

KAG's chat client additionally injects a vLLM-specific
``chat_template_kwargs`` entry into every request's ``extra_body``.  Strict
OpenAI-compatible gateways reject the unknown top-level field with HTTP 400,
which silently empties every extraction/solve call (the builder invokes the
LLM with ``with_except=False``).  The second shim drops that entry when it
carries only the default ``enable_thinking=False`` marker; an explicit
``think=True`` opt-in is preserved for vLLM backends that understand it.

The memory-graph writer can persist name-vector ``Chunk`` nodes without a
``content`` attribute (title-only duplicates).  The vector chunk retriever
returns them verbatim, and the hybrid executor then crashes while reporting
them (``chunk.content[:10]`` on ``None``), which silently empties the whole
retrieval result.  The third shim discards content-less chunks at the
retriever boundary so solver pipelines never see them.

The pagerank chunk retriever used by the graph solver builds ``ChunkData``
with ``node["content"].replace(...)`` and raises ``AttributeError`` when a
pagerank-selected chunk node has no ``content`` attribute (the same
title-only nodes).  KAG's surrounding executor catches the exception but then
discards every graph-retrieved chunk for that query.  The fourth shim rebuilds
that step defensively so content-less nodes are skipped instead of aborting
the whole retrieval.

The vendor checkout is intentionally ignored by this repository, so keeping
the shims in the benchmark client makes the supported dependency combination
explicit and reproducible.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit


def _accepts_model_argument(factory: Callable[..., Any]) -> bool:
    """Whether this installed SDK constructor accepts an arbitrary model arg."""

    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        # An uninspectable future SDK is safest left alone.
        return True
    return any(
        parameter.name == "model" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _is_bare_azure_endpoint(value: Any) -> bool:
    """Recognize ``https://<resource>.openai.azure.com`` resource URLs."""

    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    return hostname.endswith(".openai.azure.com") and not path


def _compatible_azure_factory(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Return a factory that adapts KAG's legacy Azure constructor call."""

    def create(*args: Any, **kwargs: Any) -> Any:
        # KAG keeps ``self.model`` and sends it to chat/embeddings calls, so
        # discarding it here loses no model-selection information.
        kwargs.pop("model", None)
        base_url = kwargs.get("base_url")
        if _is_bare_azure_endpoint(base_url):
            kwargs.pop("base_url")
            kwargs["azure_endpoint"] = str(base_url).rstrip("/")
        return factory(*args, **kwargs)

    return create


def _compatible_openai_init(
    init: Callable[..., Any],
) -> Callable[..., Any]:
    """Return an ``__init__`` that drops the vLLM-only template marker."""

    def create(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        extra = getattr(self, "extra_body", None)
        if isinstance(extra, dict) and extra.get("chat_template_kwargs") == {
            "enable_thinking": False
        }:
            del extra["chat_template_kwargs"]

    return create


def ensure_azure_openai_compatibility() -> None:
    """Patch KAG's two Azure SDK import sites once when the SDK needs it."""

    from kag.common.llm import openai_client as llm_module
    from kag.common.vectorize_model import openai_model as vector_module

    for module in (llm_module, vector_module):
        if getattr(module, "_PRENATAL_KAG_AZURE_COMPAT", False):
            continue
        sync_factory = module.AzureOpenAI
        async_factory = module.AsyncAzureOpenAI
        if not _accepts_model_argument(sync_factory):
            module.AzureOpenAI = _compatible_azure_factory(sync_factory)
        if not _accepts_model_argument(async_factory):
            module.AsyncAzureOpenAI = _compatible_azure_factory(async_factory)
        module._PRENATAL_KAG_AZURE_COMPAT = True


def ensure_openai_extra_body_compatibility() -> None:
    """Drop KAG's default vLLM template marker from chat ``extra_body``."""

    from kag.common.llm import openai_client as llm_module

    if getattr(llm_module, "_PRENATAL_KAG_EXTRA_BODY_COMPAT", False):
        return
    client_class = llm_module.OpenAIClient
    client_class.__init__ = _compatible_openai_init(client_class.__init__)  # type: ignore[method-assign]
    llm_module._PRENATAL_KAG_EXTRA_BODY_COMPAT = True


def ensure_chunk_content_compatibility() -> None:
    """Drop retrieved chunks that carry no usable text content."""

    from kag.common.tools.algorithm_tool.chunk_retriever import (
        vector_chunk_retriever as retriever_module,
    )

    if getattr(retriever_module, "_PRENATAL_KAG_CHUNK_COMPAT", False):
        return
    retriever_class = retriever_module.VectorChunkRetriever
    original_invoke = retriever_class.invoke

    def _invoke_without_empty_content(self: Any, task: Any, **kwargs: Any) -> Any:
        output = original_invoke(self, task, **kwargs)
        chunks = getattr(output, "chunks", None)
        if chunks:
            output.chunks = [
                chunk
                for chunk in chunks
                if (getattr(chunk, "content", None) or "").strip()
            ]
        return output

    retriever_class.invoke = _invoke_without_empty_content  # type: ignore[method-assign]
    retriever_module._PRENATAL_KAG_CHUNK_COMPAT = True


def ensure_ppr_chunk_content_compatibility() -> None:
    """Skip pagerank-selected chunk nodes that carry no text content."""

    from kag.common.tools.algorithm_tool.chunk_retriever import (
        ppr_chunk_retriever as ppr_module,
    )

    if getattr(ppr_module, "_PRENATAL_KAG_PPR_COMPAT", False):
        return
    retriever_class = ppr_module.PprChunkRetriever
    chunk_data = ppr_module.ChunkData
    retriever_output = ppr_module.RetrieverOutput

    def _invoke_without_none_content(self: Any, task: Any, **kwargs: Any) -> Any:
        top_k = kwargs.get("top_k", self.top_k)
        query = task.arguments["query"]
        matched_entities = self.linking_matched_entities(query, **kwargs)
        pagerank_res = (
            self.calculate_pagerank_scores(matched_entities, top_k=top_k)
            if matched_entities
            else {}
        )
        pagerank_scores: dict[Any, Any] = {}
        is_need_get_doc = False
        for key, value in pagerank_res.items():
            if isinstance(value, float):
                pagerank_scores[key] = value
                is_need_get_doc = True
            else:
                pagerank_scores[key] = value["score"]
        sorted_scores = sorted(
            pagerank_scores.items(), key=lambda item: item[1], reverse=True
        )
        if is_need_get_doc:
            matched_docs = self.get_all_docs_by_id([query], sorted_scores, top_k)
        else:
            matched_docs = []
            for doc_id, score in sorted_scores:
                node = pagerank_res[doc_id].get("node") or {}
                content = node.get("content")
                if content is None or not str(content).strip():
                    # A chunk without text cannot serve as evidence, and the
                    # vendor's ``content.replace`` would raise on it.
                    continue
                matched_docs.append(
                    chunk_data(
                        content=content.replace("_split_0", ""),
                        title=(node.get("name") or "").replace("_split_0", ""),
                        chunk_id=doc_id,
                        score=score,
                        properties=node,
                    )
                )

        return retriever_output(retriever_method=self.name, chunks=matched_docs)

    retriever_class.invoke = _invoke_without_none_content  # type: ignore[method-assign]
    ppr_module._PRENATAL_KAG_PPR_COMPAT = True


def ensure_logic_form_parse_compatibility() -> None:
    """Tolerate markdown backticks in the static planner's plan output.

    The static planner prompt asks for ``Step``/``Action`` lines "in code
    style", and gpt-5-mini complies literally: it either wraps whole plan
    lines in backticks or keeps the action inside backticks.  The vendor's
    ``parse_steps`` only recognises lines starting with a bare
    ``Step``/``Action`` prefix, and ``parse_logic_form`` requires the operator
    at the very first character.  Backticked output therefore either yields
    zero tasks (the whole retrieval is silently skipped) or raises
    "parse logic form error".  The shim strips whole-line code fences, splits
    step lines that carry the action itself, and strips action-level
    backticks before parsing.
    """

    import re as _re

    from kag.common.parser import logic_node_parser as parser_module
    from kag.solver.prompt import lf_static_planning_prompt as prompt_module

    if getattr(prompt_module, "_PRENATAL_KAG_LOGIC_FORM_COMPAT", False):
        return

    original_parse_steps = prompt_module.RetrieverLFStaticPlanningPrompt.parse_steps
    original_parse_logic_form = parser_module.ParseLogicForm.parse_logic_form
    operator_prefix = _re.compile(
        r"^(Retrieval|Retriever|GetSPO|Get|Deduce|Math|Output|Search_S|Search)\s*\(",
        _re.IGNORECASE,
    )

    def parse_steps(self: Any, response: str) -> Any:
        lines: list[str] = []
        for raw_line in str(response).split("\n"):
            line = raw_line.strip()
            # A whole line wrapped as inline code (``Step1: ...``) is only
            # unwrapped when the fence is at both ends of the line.
            if line.startswith("`"):
                line = line.strip("`").strip()
            if not line:
                continue
            step = _re.match(r"Step(\d+)\s*[:：]\s*(.+)$", line)
            if step and operator_prefix.match(step.group(2).strip()):
                # The step line carries the action itself; emit it as both
                # the step description and the action so the counts align.
                lines.append(f"Step{step.group(1)}: {step.group(2).strip()}")
                lines.append(f"Action{step.group(1)}: {step.group(2).strip()}")
                continue
            lines.append(line)
        return original_parse_steps(self, "\n".join(lines))

    def parse_logic_form(
        self: Any,
        input_str: str,
        parsed_entity_set: dict | None = None,
        sub_query: str | None = None,
        query: str | None = None,
    ) -> Any:
        cleaned = str(input_str).strip()
        if cleaned.startswith("`") or cleaned.endswith("`"):
            cleaned = cleaned.strip("`").strip()
        return original_parse_logic_form(
            self,
            cleaned,
            parsed_entity_set if parsed_entity_set is not None else {},
            sub_query=sub_query,
            query=query,
        )

    prompt_module.RetrieverLFStaticPlanningPrompt.parse_steps = parse_steps  # type: ignore[method-assign]
    parser_module.ParseLogicForm.parse_logic_form = parse_logic_form  # type: ignore[method-assign]
    prompt_module._PRENATAL_KAG_LOGIC_FORM_COMPAT = True


__all__ = [
    "ensure_azure_openai_compatibility",
    "ensure_chunk_content_compatibility",
    "ensure_logic_form_parse_compatibility",
    "ensure_openai_extra_body_compatibility",
    "ensure_ppr_chunk_content_compatibility",
]
