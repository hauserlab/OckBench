"""Server-reported `timings` capture (draft-acceptance producer side).

llama.cpp/llama-server attaches a non-standard `timings` object to its
completion response — carrying, among other things, `draft_n`/
`draft_n_accepted` for speculative decoding. The OpenAI SDK has no schema slot
for it, so unknown fields land in pydantic's `model_extra`, and different
llama.cpp builds have nested it in different places (top-level on the final
chunk, or under `usage`). `_dispatch` probes both locations, in order, and
falls through to `None` — never `{}`, never zero-filled counters — because a
backend that never sends `timings` at all (LM Studio's `/v1` endpoint strips
it) is a distinct, legitimate "unknown", not a zero.

This mirrors the `ttft`/`decode_time`/`decode_tps` seam in
`test_dispatch_timing.py` exactly, one file over.
"""
import asyncio

from src.models.registry import create_provider
from tests.transport_fakes import _Choice, _Chunk, _Delta, _FakeStream, _Usage


def _client(**overrides):
    return create_provider(
        "chat_completion", model="m", api_key="k", base_url="https://x/v1", **overrides)


def test_timings_on_the_chunk_itself_survive_onto_server_timings():
    client = _client()
    timings = {"draft_n": 100, "draft_n_accepted": 87, "predicted_ms": 1234.5}
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="hi"), finish_reason="stop")]),
        _Chunk(
            usage=_Usage(prompt_tokens=5, completion_tokens=2, reasoning_tokens=0, total_tokens=7),
            model_extra={"timings": timings},
        ),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    resp = asyncio.run(client._dispatch({}))
    assert resp.server_timings == timings
    # Nothing else about the response is disturbed by this addition.
    assert resp.text == "hi"
    assert resp.error is None


def test_timings_nested_under_usage_survive_onto_server_timings():
    # Some llama.cpp builds attach `timings` under `usage` rather than at the
    # top level of the final chunk — the second probe location.
    client = _client()
    timings = {"draft_n": 40, "draft_n_accepted": 30}
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="hi"), finish_reason="stop")]),
        _Chunk(
            usage=_Usage(
                prompt_tokens=5, completion_tokens=2, reasoning_tokens=0, total_tokens=7,
                model_extra={"timings": timings},
            ),
        ),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    resp = asyncio.run(client._dispatch({}))
    assert resp.server_timings == timings


def test_chunk_level_timings_take_priority_over_usage_level():
    # When both locations somehow carry a `timings` key, the chunk-level probe
    # runs first and wins — pin the documented probe ORDER, not just "one of
    # them is picked".
    client = _client()
    chunk_timings = {"draft_n": 10, "draft_n_accepted": 9}
    usage_timings = {"draft_n": 999, "draft_n_accepted": 1}
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="hi"), finish_reason="stop")]),
        _Chunk(
            usage=_Usage(
                prompt_tokens=5, completion_tokens=2, reasoning_tokens=0, total_tokens=7,
                model_extra={"timings": usage_timings},
            ),
            model_extra={"timings": chunk_timings},
        ),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    resp = asyncio.run(client._dispatch({}))
    assert resp.server_timings == chunk_timings


def test_no_timings_anywhere_leaves_server_timings_none_lm_studio_case():
    # LM Studio's /v1 endpoint strips server timings entirely — no `timings`
    # key on either the chunk or the usage object's model_extra. This is the
    # ordinary case for that backend and must not raise, and must leave
    # server_timings as None (never {} and never zero-filled counters).
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="hi"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=2, reasoning_tokens=0, total_tokens=7)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    resp = asyncio.run(client._dispatch({}))
    assert resp.server_timings is None
    assert resp.error is None


def test_usage_with_model_extra_but_no_timings_key_leaves_server_timings_none():
    # model_extra present (other unknown fields) but no `timings` key at all —
    # must fall through to None, not KeyError, not {}.
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="hi"), finish_reason="stop")]),
        _Chunk(
            usage=_Usage(
                prompt_tokens=5, completion_tokens=2, reasoning_tokens=0, total_tokens=7,
                model_extra={"some_other_field": 1},
            ),
        ),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    resp = asyncio.run(client._dispatch({}))
    assert resp.server_timings is None
