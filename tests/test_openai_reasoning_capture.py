"""Reasoning-capture regression tests for the chat_completion (OpenAI-compatible)
client.

Bug: a reasoning model (enable_thinking=True) streams its <think> channel as
`reasoning_content` deltas alongside the answer's `content` deltas. `_dispatch`
used to count reasoning chars for the empty-response heuristic but never kept
the text, so the persisted record could never show WHY a model answered the
way it did. These tests pin `_dispatch` accumulating the full reasoning stream
into `ModelResponse.reasoning_text`, without disturbing `text`/token
accounting (AC already covered elsewhere).
"""
import asyncio

from src.models.registry import create_provider
from tests.transport_fakes import _Chunk, _Choice, _Delta, _Usage, drive_chat


def _client(**overrides):
    return create_provider(
        "chat_completion", model="m", api_key="k", base_url="https://x/v1", **overrides)


def _stream_chunks(deltas, finish="stop", prompt_tokens=5, completion_tokens=7,
                   reasoning_tokens=3, total_tokens=15):
    """``deltas``: list of (content, reasoning_content) pairs, one per streamed
    chunk, in order. The last content/reasoning chunk carries ``finish_reason``;
    a trailing usage-only chunk (canonical's ``stream_options.include_usage``
    shape) follows, matching what `openai_chunks` already does for plain text.
    """
    chunks = []
    for i, (content, reasoning) in enumerate(deltas):
        fr = finish if i == len(deltas) - 1 else None
        chunks.append(_Chunk(choices=[_Choice(_Delta(content=content, reasoning_content=reasoning), finish_reason=fr)]))
    chunks.append(_Chunk(usage=_Usage(prompt_tokens, completion_tokens, reasoning_tokens, total_tokens)))
    return chunks


def test_dispatch_accumulates_reasoning_text_separately_from_answer_text():
    client = _client()
    deltas = [
        (None, "Let me think"),
        (None, " about this problem."),
        ("The answer", None),
        (" is 42.", None),
    ]
    _, resp = asyncio.run(drive_chat(client, chunks=_stream_chunks(deltas)))
    assert resp.reasoning_text == "Let me think about this problem."
    assert resp.text == "The answer is 42."
    assert resp.error is None


def test_dispatch_interleaved_reasoning_and_content_deltas_accumulate_independently():
    client = _client()
    # A relay may interleave content and reasoning_content deltas within the
    # same stream rather than emitting all of one kind first.
    deltas = [
        (None, "Step 1. "),
        ("Partial ", None),
        (None, "Step 2."),
        ("answer.", None),
    ]
    _, resp = asyncio.run(drive_chat(client, chunks=_stream_chunks(deltas)))
    assert resp.reasoning_text == "Step 1. Step 2."
    assert resp.text == "Partial answer."


def test_dispatch_no_reasoning_content_leaves_reasoning_text_empty():
    client = _client()
    _, resp = asyncio.run(drive_chat(client))  # default openai_chunks: content only
    assert resp.reasoning_text == ""
    assert resp.text == "ok"


def test_dispatch_reasoning_via_model_extra_fallback_accumulates():
    # Some SDK/relay combos surface reasoning_content only through
    # `delta.model_extra` (the attribute is absent, not None) rather than a
    # first-class field — _dispatch already falls back to that; pin it still
    # accumulates into reasoning_text.
    client = _client()

    class _ExtraDelta:
        def __init__(self, content=None, extra=None):
            self.content = content
            self.model_extra = extra
            # No `reasoning_content` attribute at all (simulates an SDK that
            # doesn't declare the field, only exposes it via model_extra).

    chunks = [
        _Chunk(choices=[_Choice(_ExtraDelta(extra={"reasoning_content": "hmm, "}))]),
        _Chunk(choices=[_Choice(_ExtraDelta(extra={"reasoning_content": "carry the one"}), finish_reason="stop")]),
        _Chunk(usage=_Usage(5, 7, 3, 15)),
    ]
    _, resp = asyncio.run(drive_chat(client, chunks=chunks))
    assert resp.reasoning_text == "hmm, carry the one"
    assert resp.text == ""
