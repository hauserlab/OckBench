"""OpenAI-compatible chat completions client (OpenAI, vLLM, SGLang, OpenRouter)."""
import logging
import time
from typing import Any, Dict

import httpx
from openai import AsyncOpenAI

from ..core.schemas import ModelResponse, TokenUsage
from ..utils.usage_normalizer import extract_openai_usage, to_token_usage
from .base import BaseModelClient
from .registry import register_provider

logger = logging.getLogger(__name__)


@register_provider("chat_completion")
class OpenAIClient(BaseModelClient):
    """Client for any OpenAI-compatible chat completions endpoint."""

    protected_paths = ("model", "messages", "stream", "stream_options")
    provider_name = "chat_completion"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # max_retries=0: BaseModelClient.generate owns retry; the SDK's own loop
        # would stack with ours. read timeout = per-chunk gap during streaming.
        client_kwargs = {
            'timeout': httpx.Timeout(connect=30.0, read=float(self.timeout), write=60.0, pool=30.0),
            'max_retries': 0,
        }
        if self.api_key:
            client_kwargs['api_key'] = self.api_key
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
            if 'api_key' not in client_kwargs:
                client_kwargs['api_key'] = 'dummy-key'

        self.client = AsyncOpenAI(**client_kwargs)

    async def aclose(self) -> None:
        await self.client.close()

    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        request["max_tokens"] = max_output_tokens
        if self.top_p is not None:
            request["top_p"] = self.top_p
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        return request

    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        try:
            text = ""
            reasoning_text = ""
            finish_reason = None
            model_name = self.model
            usage_chunk = None

            # Client-side, backend-agnostic timing (mirrors
            # ockbench_harness/tools/bench_utils.py::timed_call's formulas
            # exactly): some backends strip server-reported stats, so ttft/
            # decode timing is measured here off the actual stream, never
            # trusted from a server-reported field. `ttft` is the wall-clock to
            # the FIRST content OR reasoning delta (a reasoning model's <think>
            # channel starts the clock just as much as a content delta would;
            # bench_utils.py only counted content, understating a reasoning
            # model's true first-token latency).
            t0 = time.time()
            ttft: float | None = None

            stream = await self.client.chat.completions.create(**request)
            async for chunk in stream:
                if chunk.model:
                    model_name = chunk.model
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        if ttft is None:
                            ttft = time.time() - t0
                        text += delta.content
                    if delta:
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if reasoning_delta is None and getattr(delta, "model_extra", None):
                            reasoning_delta = delta.model_extra.get("reasoning_content")
                        if reasoning_delta:
                            if ttft is None:
                                ttft = time.time() - t0
                            reasoning_text += reasoning_delta
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    usage_chunk = chunk

            total_latency = time.time() - t0

            if usage_chunk:
                tokens = self._extract_tokens(usage_chunk, text)
            else:
                tokens = TokenUsage(
                    prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                    output_tokens=0, total_tokens=0,
                )

            # decode_time/decode_tps, mirroring bench_utils.py::timed_call. ttft
            # is None only when no content/reasoning delta ever arrived (always
            # an error response below — see empty_error), in which case there
            # was no observed decode phase: decode_time falls back to 0 rather
            # than the whole total_latency.
            decode_time = total_latency - (ttft if ttft is not None else total_latency)
            decode_tps = tokens.output_tokens / decode_time if decode_time > 0 else 0.0

            # Server-reported channel (the OTHER half, distinct from the
            # client-side ttft/decode_time/decode_tps above): llama.cpp/
            # llama-server attaches a non-standard `timings` object to the
            # final streamed chunk carrying speculative-decoding counters
            # (`draft_n`/`draft_n_accepted`, among others). The OpenAI SDK has
            # no schema slot for it, so it lands in pydantic's `model_extra` —
            # and different llama.cpp builds have put it in different places,
            # so probe defensively, in order: the chunk itself, then the
            # chunk's `usage` object. `getattr(..., "model_extra", None) or {}`
            # guards both "attribute doesn't exist on this object" (a plain
            # OpenAI/vLLM/SGLang response) and "model_extra is None" (SDK
            # default when nothing extra was sent). Falls through to `None` —
            # never `{}`, never a zero-filled counters dict — a backend that
            # never sends `timings` at all (LM Studio's `/v1` endpoint strips
            # it) is legitimately unknown, not zero.
            server_timings: dict[str, Any] | None = None
            if usage_chunk is not None:
                chunk_extra = getattr(usage_chunk, "model_extra", None) or {}
                server_timings = chunk_extra.get("timings")
                if server_timings is None:
                    usage_obj = getattr(usage_chunk, "usage", None)
                    usage_extra = getattr(usage_obj, "model_extra", None) or {}
                    server_timings = usage_extra.get("timings")

            # Surface empty-text outcomes as errors so --cache resume will retry
            # them. Common on reasoning models that spend the whole budget on
            # reasoning_content and end the stream without a content delta.
            empty_error = None
            if not text:
                if finish_reason == "length":
                    suffix = " after reasoning_content stream" if reasoning_text else ""
                    empty_error = (
                        "empty_response_length_finish: finish_reason=length with no content "
                        f"emitted{suffix} (likely reasoning consumed entire output budget)"
                    )
                elif tokens.reasoning_tokens > 0 or reasoning_text:
                    empty_error = (
                        "empty_response_reasoning_only: model emitted reasoning tokens but "
                        f"no content (finish_reason={finish_reason or 'unknown'})"
                    )
                elif usage_chunk is None:
                    empty_error = "empty_response_no_stream: no usage and no content received"
                else:
                    empty_error = (
                        "empty_response_no_content: stream completed with usage but no content "
                        f"(finish_reason={finish_reason or 'unknown'})"
                    )

            return ModelResponse(
                text=text,
                reasoning_text=reasoning_text,
                tokens=tokens,
                latency=total_latency,
                ttft=round(ttft, 3) if ttft is not None else None,
                decode_time=round(decode_time, 3),
                decode_tps=round(decode_tps, 2),
                server_timings=server_timings,
                model=model_name,
                finish_reason=finish_reason or "stop",
                error=empty_error,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _extract_tokens(self, response, final_text: str = "") -> TokenUsage:
        return to_token_usage(extract_openai_usage(response.usage, final_text=final_text))
