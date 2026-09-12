"""Bound KAG provider calls and stop a query after a swallowed provider failure.

Only KAG's SDK import sites are patched. SDK retries (two) retain Retry-After
handling; separate sync/async gates prevent internal chat/embedding bursts. No model or
index configuration is changed.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math
import threading

_GATE = threading.Lock()
_ASYNC_GATE = threading.Lock()
_QUERY_GATE = threading.Lock()


class ProviderFailure(RuntimeError):
    """Provider exhausted its retries or returned unusable embeddings."""


@dataclass
class QueryState:
    error: Exception | None = None


_STATE: ContextVar[QueryState | None] = ContextVar('kag_provider_state', default=None)
# KAG queries already serialize access to global configuration. This fallback
# also reaches vendor ThreadPoolExecutor workers that do not copy ContextVars.
_ACTIVE_STATE: QueryState | None = None


def _failed(exc):
    state = _STATE.get() or _ACTIVE_STATE
    if state is not None and state.error is None:
        state.error = exc


def _check():
    state = _STATE.get() or _ACTIVE_STATE
    if state is not None and state.error is not None:
        raise ProviderFailure('KAG provider failed; query aborted') from state.error


def _validate(response, inputs):
    count = 1 if isinstance(inputs, str) else len(inputs)
    data = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in data] != list(range(count)):
        raise ProviderFailure('Embedding response count/index mismatch')
    dimension = None
    for item in data:
        vector = item.embedding
        if not vector or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector):
            raise ProviderFailure('Embedding response contains invalid vector')
        if dimension is not None and len(vector) != dimension:
            raise ProviderFailure('Embedding response dimensions differ')
        dimension = len(vector)
    response.data = data
    return response


def _wrap_create(create, asynchronous, embedding):
    def arguments(kwargs):
        kwargs = dict(kwargs)
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = 120.0
        return kwargs

    if asynchronous:
        @wraps(create)
        async def invoke(**kwargs):
            while not _ASYNC_GATE.acquire(blocking=False):
                _check()
                await asyncio.sleep(0.05)
            try:
                _check()
                result = await create(**arguments(kwargs))
                return _validate(result, kwargs['input']) if embedding else result
            except Exception as exc:
                _failed(exc)
                raise
            finally:
                _ASYNC_GATE.release()
    else:
        @wraps(create)
        def invoke(**kwargs):
            with _GATE:
                try:
                    _check()
                    result = create(**arguments(kwargs))
                    return _validate(result, kwargs['input']) if embedding else result
                except Exception as exc:
                    _failed(exc)
                    raise
    return invoke


def _factory(factory, asynchronous):
    @wraps(factory)
    def create(*args, **kwargs):
        kwargs['max_retries'] = 2
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = 120.0
        client = factory(*args, **kwargs)
        for resource, embedding in ((client.embeddings, True), (client.chat.completions, False)):
            resource.create = _wrap_create(resource.create, asynchronous, embedding)
        return client
    return create


async def guarded_query(operation):
    global _ACTIVE_STATE
    try:
        while not _QUERY_GATE.acquire(blocking=False):
            await asyncio.sleep(0.05)
    except BaseException:
        operation.close()
        raise
    state = QueryState()
    _ACTIVE_STATE = state
    token = _STATE.set(state)
    task = asyncio.create_task(operation)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            _check()
        _check()
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        _STATE.reset(token)
        _ACTIVE_STATE = None
        _QUERY_GATE.release()


def ensure_provider_guard():
    from kag.common.llm import openai_client
    from kag.common.vectorize_model import openai_model
    for module in (openai_client, openai_model):
        if getattr(module, '_PRENATAL_PROVIDER_GUARD', False):
            continue
        for name in ('OpenAI', 'AsyncOpenAI', 'AzureOpenAI', 'AsyncAzureOpenAI'):
            setattr(module, name, _factory(getattr(module, name), name.startswith('Async')))
        module._PRENATAL_PROVIDER_GUARD = True

    # Replace the vendor methods that swallow errors and split async strings
    # into characters. Invalid input is rejected, never replaced with evidence.
    def prepare(texts):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError('Embedding inputs must be nonempty strings')
        return single, values

    def vectorize(self, texts):
        single, values = prepare(texts)
        if not values:
            return []
        response = self.client.embeddings.create(input=values, model=self.model)
        vectors = [item.embedding for item in response.data]
        return vectors[0] if single else vectors

    async def avectorize(self, texts):
        single, values = prepare(texts)
        if not values:
            return []
        async with self.limiter:
            response = await self.aclient.embeddings.create(input=values, model=self.model)
        vectors = [item.embedding for item in response.data]
        return vectors[0] if single else vectors

    for cls in (openai_model.OpenAIVectorizeModel, openai_model.AzureOpenAIVectorizeModel):
        cls.vectorize = vectorize
        cls.avectorize = avectorize

    from kag.common.tools.algorithm_tool.ner import Ner
    if not getattr(Ner, '_PRENATAL_SHAPE_GUARD', False):
        original = Ner._parse_ner_list

        @wraps(original)
        def parse(self, query):
            import logging
            items = original(self, query)
            if not isinstance(items, list):
                logging.getLogger(__name__).warning('Ignoring malformed NER result: expected list')
                return []
            valid = [item for item in items if isinstance(item, dict)
                     and isinstance(item.get('name'), str)
                     and isinstance(item.get('category', ''), str)
                     and isinstance(item.get('official_name', item.get('name')), str)]
            if len(valid) != len(items):
                logging.getLogger(__name__).warning('Ignoring %d malformed NER entries', len(items) - len(valid))
            return valid

        Ner._parse_ner_list = parse
        Ner._PRENATAL_SHAPE_GUARD = True
