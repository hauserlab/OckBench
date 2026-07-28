"""Timing-parity threading (Task: ttft/decode_time/decode_tps persistence).

`ModelResponse` and `EvaluationResult` now carry `ttft`/`decode_time`/
`decode_tps` (siblings of `latency`) — but the schema fields alone don't get a
value from the model call into the persisted row; `BenchmarkRunner` has to
thread them through, the same way it already threads `latency`/`tokens`/
`reasoning_text` (see `test_reasoning_capture_gate.py` for the parallel
pattern this file follows). These tests drive a full
`BenchmarkRunner(...).run()` end to end (offline, fixed-answer test double)
and assert the new fields survive into both `exp.results` AND the on-disk
cache line — the write path the completeness gate will read.
"""
import json
import tempfile
from pathlib import Path

import src.evaluators.math_eval as math_eval
import src.models.registry as registry
from src.core.cache import aggregate_cache_file
from src.core.runner import BenchmarkRunner
from src.core.schemas import BenchmarkConfig, JudgeConfig, ModelResponse, TokenUsage
from src.evaluators.judge import JudgeVerdict
from src.models.base import BaseModelClient


class _CountingJudge:
    async def score(self, *, question, ground_truth, candidate):
        return JudgeVerdict(correct=(str(candidate) == str(ground_truth)),
                            extracted_answer=str(candidate), reasoning="")


class _ErroringJudge:
    async def score(self, *, question, ground_truth, candidate):
        return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error="judge timeout")


def _register_timed_provider(name, *, ttft=1.5, decode_time=3.5, decode_tps=5.71, error=None):
    @registry.register_provider(name)
    class _TimedClient(BaseModelClient):
        protected_paths = ("model",)
        provider_name = name

        def build_request(self, prompt, max_output_tokens):
            return {"model": self.model, "prompt": prompt}

        async def _dispatch(self, request):
            return ModelResponse(
                text="" if error else "<answer>6</answer>",
                tokens=TokenUsage(
                    prompt_tokens=10, answer_tokens=5, reasoning_tokens=15,
                    output_tokens=20, total_tokens=30,
                ) if not error else TokenUsage(
                    prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                    output_tokens=0, total_tokens=0,
                ),
                latency=(ttft + decode_time) if not error else 0.0,
                ttft=None if error else ttft,
                decode_time=0.0 if error else decode_time,
                decode_tps=0.0 if error else decode_tps,
                model=self.model, finish_reason="stop",
                error=error,
            )

    return name


def _dataset(tmp, n=1):
    path = Path(tmp) / "ds.jsonl"
    lines = [json.dumps({"problem": f"q{i}", "answer": "6", "id": f"p{i}"}) for i in range(1, n + 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _config(provider, dataset_path, **extra):
    base = dict(dataset_path=dataset_path, provider=provider, model="m",
                max_output_tokens=100, evaluator_type="math",
                judge=JudgeConfig(model="judge-m", base_url="https://judge/v1", api_key="jk"))
    base.update(extra)
    return BenchmarkConfig(**base)


def test_success_threads_timing_into_results_and_cache(monkeypatch):
    provider = _register_timed_provider("fake-timed-ok", ttft=1.5, decode_time=3.5, decode_tps=5.71)
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            result = exp.results[0]
            assert result.ttft == 1.5
            assert result.decode_time == 3.5
            assert result.decode_tps == 5.71
            # NOTE: `latency` itself is NOT asserted here — `BaseModelClient.
            # generate()` always overwrites it with its OWN outer wall-clock
            # measurement (real base.py behavior, unrelated to this fix); this
            # test double's fake `_dispatch` return value for `latency` is not
            # what ends up persisted. ttft/decode_time/decode_tps are NOT
            # touched by that override (base.py only reassigns `.latency`),
            # which is exactly the parity gap this test is pinning.

            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            body = [json.loads(ln) for ln in lines if "__ockbench_cache__" not in ln]
            assert body[0]["ttft"] == 1.5
            assert body[0]["decode_time"] == 3.5
            assert body[0]["decode_tps"] == 5.71

            results, _ = aggregate_cache_file(cache_path)
            assert results[0].ttft == 1.5
            assert results[0].decode_tps == 5.71
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_error_response_threads_none_timing_without_crashing(monkeypatch):
    provider = _register_timed_provider("fake-timed-err", error="empty_response_no_content: boom")
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            result = exp.results[0]
            assert result.error is not None
            assert result.ttft is None
            assert result.decode_time == 0.0
            assert result.decode_tps == 0.0
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_rejudge_carries_forward_cached_timing_without_regenerating(monkeypatch):
    # A judge-outage resume re-scores the cached model_response without calling
    # the model again (see test_cache_identity.py's parallel rejudge test) — the
    # ttft/decode_time/decode_tps already banked on the first pass must survive
    # untouched, the same way reasoning_text does.
    provider = _register_timed_provider("fake-timed-rejudge", ttft=2.25, decode_time=4.0, decode_tps=5.0)
    judge = _ErroringJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)

            exp1 = BenchmarkRunner(cfg, cache_path=cache_path).run()
            assert exp1.summary.error_count == 1  # judge failed, generation succeeded

            monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
            exp2 = BenchmarkRunner(cfg, cache_path=cache_path).run()
            assert exp2.summary.error_count == 0
            assert exp2.results[0].ttft == 2.25
            assert exp2.results[0].decode_time == 4.0
            assert exp2.results[0].decode_tps == 5.0
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)
