"""Client-side timing parity fix for the chat_completion (OpenAI-compatible)
streaming client.

Bug: `_dispatch` streamed the completion but only ever persisted a coarse
`latency=0` (overwritten later, outside `_dispatch`, by the base retry
wrapper's OWN outer wall-clock measurement) — it never captured `ttft` /
`decode_time` / `decode_tps`, even though the separate `bench_utils.py`
benchmarking path already computes exactly this from the same kind of stream
(see `ockbench_harness/tools/bench_utils.py::timed_call`). These tests pin
`_dispatch` computing all four timing quantities CLIENT-SIDE (from wall-clock
around the actual stream, never from server-reported stats — some backends
strip those), mirroring `timed_call`'s formulas exactly:

    ttft           = wall-clock from request start to the FIRST content/
                     reasoning token received
    total_latency  = request start to stream end (persisted as `latency`,
                     the pre-existing field name — back-compat)
    decode_time     = total_latency - ttft
    decode_tps      = output_tokens / decode_time (0.0 guard on divide-by-zero)

Clock is monkeypatched deterministically (rather than real `asyncio.sleep`)
so the exact numbers are pinned, not just "roughly right".
"""
import asyncio

import pytest

import src.models.openai_api as openai_api_mod
from src.models.registry import create_provider
from tests.transport_fakes import _Choice, _Chunk, _Delta, _FakeStream, _Usage


def _client(**overrides):
    return create_provider(
        "chat_completion", model="m", api_key="k", base_url="https://x/v1", **overrides)


def _fake_clock(monkeypatch, values):
    """Deterministic `time.time()` sequence for `src.models.openai_api`."""
    it = iter(values)

    def _next():
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - guard against under-provisioning
            raise AssertionError("time.time() called more times than the test provisioned for")
    monkeypatch.setattr(openai_api_mod.time, "time", _next)


def test_dispatch_computes_ttft_decode_time_and_decode_tps(monkeypatch):
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="Hello"))]),
        _Chunk(choices=[_Choice(_Delta(content=" world"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=10, reasoning_tokens=0, total_tokens=15)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    # t0=100.0 (request start); first content delta at 102.0 (ttft=2.0); stream
    # ends at 105.0 (total_latency=5.0). Exactly 3 time.time() calls: t0, the
    # ONE ttft-setting call (only the first content/reasoning delta sets it —
    # the second content chunk must NOT call time.time() again), total_latency.
    _fake_clock(monkeypatch, [100.0, 102.0, 105.0])

    resp = asyncio.run(client._dispatch({}))

    assert resp.ttft == pytest.approx(2.0)
    assert resp.latency == pytest.approx(5.0)  # total_latency, back-compat field name
    assert resp.decode_time == pytest.approx(3.0)
    assert resp.tokens.output_tokens == 10
    assert resp.decode_tps == pytest.approx(10 / 3.0, abs=0.005)  # decode_tps is round(., 2)
    assert resp.latency >= resp.ttft
    assert resp.error is None


def test_dispatch_ttft_set_on_first_reasoning_delta_too(monkeypatch):
    # ttft is "first CONTENT OR REASONING token" — a reasoning model's <think>
    # channel starts the clock just as much as a content delta would.
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(reasoning_content="thinking..."))]),
        _Chunk(choices=[_Choice(_Delta(content="answer"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=8, reasoning_tokens=4, total_tokens=17)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    _fake_clock(monkeypatch, [100.0, 101.5, 104.0])

    resp = asyncio.run(client._dispatch({}))
    assert resp.ttft == pytest.approx(1.5)
    assert resp.decode_time == pytest.approx(2.5)


def test_dispatch_decode_tps_zero_when_decode_time_not_positive(monkeypatch):
    # ttft == total_latency (whole answer arrived "instantly") must not divide
    # by zero — decode_tps is guarded to 0.0.
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(content="Hi"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=3, reasoning_tokens=0, total_tokens=8)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    _fake_clock(monkeypatch, [100.0, 100.0, 100.0])

    resp = asyncio.run(client._dispatch({}))
    assert resp.decode_time == 0.0
    assert resp.decode_tps == 0.0


def test_dispatch_no_content_ever_leaves_ttft_none_and_decode_time_zero(monkeypatch):
    # No content/reasoning delta ever arrives (e.g. finish_reason=length with an
    # empty stream) -> ttft stays None; decode_time falls back to 0 (no decode
    # phase was ever observed), matching bench_utils's `ttft or total_latency`
    # formula. This is always an error response (see empty_response_* below),
    # excluded from the completeness gate's strict scope.
    client = _client()
    chunks = [
        _Chunk(choices=[_Choice(_Delta(), finish_reason="length")]),
        _Chunk(usage=_Usage(prompt_tokens=5, completion_tokens=0, reasoning_tokens=0, total_tokens=5)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)
    client.client.chat.completions.create = fake_create

    _fake_clock(monkeypatch, [100.0, 103.0])  # only t0 + total_latency (no ttft call)

    resp = asyncio.run(client._dispatch({}))
    assert resp.ttft is None
    assert resp.decode_time == 0.0
    assert resp.decode_tps == 0.0
    assert resp.error is not None  # empty content -> flagged, not silently blank
