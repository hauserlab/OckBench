"""Server-timings threading (Task: draft-acceptance producer side).

`ModelResponse.server_timings` and `EvaluationResult.server_timings` are
additive/Optional siblings of `ttft`/`decode_time`/`decode_tps` — but the
schema field alone doesn't get a value from the model call into the persisted
row; `BenchmarkRunner` has to thread it through, the same way it already
threads those three (see `test_timing_persistence.py`, the file this one
mirrors). These tests drive a full `BenchmarkRunner(...).run()` end to end
(offline, fixed-answer test double) and assert `server_timings` survives into
`exp.results`, the on-disk cache line, a rejudge, AND that a pre-fix cache row
with no `server_timings` key at all still deserializes (additive-only).
"""
import json
import tempfile
from pathlib import Path

import src.evaluators.math_eval as math_eval
import src.models.registry as registry
from src.core.cache import aggregate_cache_file
from src.core.runner import BenchmarkRunner
from src.core.schemas import BenchmarkConfig, EvaluationResult, JudgeConfig, ModelResponse, TokenUsage
from src.evaluators.judge import JudgeVerdict
from src.models.base import BaseModelClient


class _CountingJudge:
    async def score(self, *, question, ground_truth, candidate):
        return JudgeVerdict(correct=(str(candidate) == str(ground_truth)),
                            extracted_answer=str(candidate), reasoning="")


class _ErroringJudge:
    async def score(self, *, question, ground_truth, candidate):
        return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error="judge timeout")


def _register_timed_provider(name, *, server_timings=None, error=None):
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
                latency=5.0 if not error else 0.0,
                ttft=None if error else 1.5,
                decode_time=0.0 if error else 3.5,
                decode_tps=0.0 if error else 5.71,
                server_timings=None if error else server_timings,
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


def test_success_threads_server_timings_into_results_and_cache(monkeypatch):
    timings = {"draft_n": 100, "draft_n_accepted": 87}
    provider = _register_timed_provider("fake-server-timings-ok", server_timings=timings)
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            result = exp.results[0]
            assert result.server_timings == timings

            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            body = [json.loads(ln) for ln in lines if "__ockbench_cache__" not in ln]
            assert body[0]["server_timings"] == timings

            results, _ = aggregate_cache_file(cache_path)
            assert results[0].server_timings == timings
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_lm_studio_backend_persists_server_timings_as_none_not_crashing(monkeypatch):
    # LM Studio's /v1 endpoint strips server timings entirely — this must
    # persist as None, never {} and never a zero-filled counters dict, and
    # must not raise anywhere in the thread-through.
    provider = _register_timed_provider("fake-server-timings-lmstudio", server_timings=None)
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            result = exp.results[0]
            assert result.server_timings is None

            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            body = [json.loads(ln) for ln in lines if "__ockbench_cache__" not in ln]
            assert body[0].get("server_timings") is None
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_error_response_threads_none_server_timings_without_crashing(monkeypatch):
    provider = _register_timed_provider("fake-server-timings-err", error="empty_response_no_content: boom")
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            result = exp.results[0]
            assert result.error is not None
            assert result.server_timings is None
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_rejudge_carries_forward_cached_server_timings_without_regenerating(monkeypatch):
    timings = {"draft_n": 40, "draft_n_accepted": 30}
    provider = _register_timed_provider("fake-server-timings-rejudge", server_timings=timings)
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
            assert exp2.results[0].server_timings == timings
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_pre_fix_result_dict_with_no_server_timings_key_still_deserializes():
    # Additive-only: a result dict written before this field existed (no
    # `server_timings` key at all) must still load, defaulting to None.
    pre_fix = dict(
        problem_id="p1", question="q", ground_truth="6",
        model_response="<answer>6</answer>", correct=True,
        tokens=dict(prompt_tokens=10, answer_tokens=5, reasoning_tokens=0,
                    output_tokens=5, total_tokens=15),
        latency=5.0, ttft=1.5, decode_time=3.5, decode_tps=5.71,
    )
    assert "server_timings" not in pre_fix
    result = EvaluationResult(**pre_fix)
    assert result.server_timings is None
