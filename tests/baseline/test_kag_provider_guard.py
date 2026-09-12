"""Real SDK transport tests; no provider credentials or network required."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import openai
import pytest

from benchmark.baseline.kag_client.provider_guard import (
    ProviderFailure, _factory, _validate, _wrap_create, guarded_query,
)


def test_retry_after_then_success():
    calls = []
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after-ms': '1'}, json={'error': {'message': 'limited'}})
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [0.2, 0.3]}], 'model': 'test', 'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
    client = _factory(openai.OpenAI, False)(api_key='test', http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    try:
        assert client.embeddings.create(model='test', input=['hello']).data[0].embedding == [0.2, 0.3]
        assert len(calls) == 2
        assert client.timeout == 120
    finally:
        client.close()


@pytest.mark.parametrize('status,expected', [(429, 3), (401, 1)])
def test_exhaustion_stops_even_when_vendor_swallows(status, expected):
    calls = []
    async def run():
        def handle(request):
            calls.append(request)
            return httpx.Response(status, headers={'retry-after-ms': '1'}, json={'error': {'message': 'failure'}})
        client = _factory(openai.AsyncOpenAI, True)(api_key='test', http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        async def operation():
            try:
                await client.embeddings.create(model='test', input=['hello'])
            except Exception:
                pass  # vendor retriever catches exceptions and continues
            await asyncio.sleep(10)
        try:
            with pytest.raises(ProviderFailure):
                await asyncio.wait_for(guarded_query(operation()), 2)
        finally:
            await client.close()
    asyncio.run(run())
    assert len(calls) == expected


@pytest.mark.parametrize('vector', [None, [], [None], [float('nan')], [float('inf')], [True]])
def test_invalid_embeddings_rejected(vector):
    with pytest.raises(ProviderFailure):
        _validate(SimpleNamespace(data=[SimpleNamespace(index=0, embedding=vector)]), ['hello'])


def test_embedding_indices_and_dimensions():
    response = SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1]), SimpleNamespace(index=0, embedding=[2])])
    assert _validate(response, ['a', 'b']).data[0].embedding == [2]
    with pytest.raises(ProviderFailure):
        _validate(response, ['a'])
    response.data[1].embedding = [1, 2]
    with pytest.raises(ProviderFailure):
        _validate(response, ['a', 'b'])


def test_async_string_is_one_input_and_errors_propagate():
    from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel
    seen = []
    async def create(**kwargs):
        seen.append(kwargs['input'])
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1, 2])])
    model = SimpleNamespace(model='test', limiter=nullcontext(), aclient=SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    assert asyncio.run(OpenAIVectorizeModel.avectorize(model, 'hello')) == [1, 2]
    assert seen == [['hello']]
    def fail(**kwargs):
        raise RuntimeError('provider failed')
    model.client = SimpleNamespace(embeddings=SimpleNamespace(create=fail))
    with pytest.raises(RuntimeError, match='provider failed'):
        OpenAIVectorizeModel.vectorize(model, 'hello')


def test_ner_malformed_entries(monkeypatch):
    from kag.common.tools.algorithm_tool.ner import Ner
    from kag.common.tools.algorithm_tool import ner
    monkeypatch.setattr(ner.ner_tool_cache, 'get', lambda _: ['bad', {'name': 'heart', 'category': 'Medical'}, {'name': 'x', 'category': None}])
    obj = object.__new__(Ner)
    assert Ner._parse_ner_list(obj, 'query') == [{'name': 'heart', 'category': 'Medical'}]


def test_executor_failure_reaches_query_guard():
    from concurrent.futures import ThreadPoolExecutor
    def fail():
        def create(**kwargs):
            raise RuntimeError('worker provider failure')
        try:
            _wrap_create(create, False, False)()
        except RuntimeError:
            pass
    async def operation():
        with ThreadPoolExecutor(max_workers=1) as pool:
            await asyncio.get_running_loop().run_in_executor(pool, fail)
        return 'must not accept this answer'
    with pytest.raises(ProviderFailure):
        asyncio.run(guarded_query(operation()))


def test_async_concurrency_and_cancellation_release_gate():
    active = 0
    peak = 0
    async def create(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return 'ok'
        finally:
            active -= 1
    wrapped = _wrap_create(create, True, False)
    async def run():
        assert await asyncio.gather(*(wrapped() for _ in range(3))) == ['ok'] * 3
        task = asyncio.create_task(wrapped())
        await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.wait_for(wrapped(), 1) == 'ok'
    asyncio.run(run())
    assert peak == 1
