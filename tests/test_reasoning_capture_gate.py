"""Persistence gate for reasoning capture (`BenchmarkConfig.capture_reasoning`).

Accumulating the model's reasoning stream (test_openai_reasoning_capture.py) is
only half the fix: persisting it for EVERY run at scale can balloon result-file/
cache storage (Elliott's explicit design constraint), so whether it reaches the
written record must be config-gated. These tests drive a full
`BenchmarkRunner(...).run()` end to end (offline, via a fixed-answer test
double, following the pattern in test_cache_identity.py) and assert the gate
holds in the persisted cache + results, in both directions.
"""
import json
import tempfile
from pathlib import Path

import src.evaluators.math_eval as math_eval
import src.models.registry as registry
from src.core.cache import aggregate_cache_file
from src.core.identity import compute_run_identity
from src.core.runner import BenchmarkRunner
from src.core.schemas import BenchmarkConfig, JudgeConfig, ModelResponse, TokenUsage
from src.evaluators.judge import JudgeVerdict
from src.models.base import BaseModelClient

_REASONING = "Let's see: 3+3=6, so the answer is 6."


class _CountingJudge:
    async def score(self, *, question, ground_truth, candidate):
        return JudgeVerdict(correct=(str(candidate) == str(ground_truth)),
                            extracted_answer=str(candidate), reasoning="")


def _register_reasoning_provider(name):
    @registry.register_provider(name)
    class _ReasoningClient(BaseModelClient):
        protected_paths = ("model",)
        provider_name = name

        def build_request(self, prompt, max_output_tokens):
            return {"model": self.model, "prompt": prompt}

        async def _dispatch(self, request):
            return ModelResponse(
                text="<answer>6</answer>",
                reasoning_text=_REASONING,
                tokens=TokenUsage(prompt_tokens=10, answer_tokens=5, reasoning_tokens=15,
                                  output_tokens=20, total_tokens=30),
                latency=0, model=self.model, finish_reason="stop",
            )

    return name


def _dataset(tmp):
    path = Path(tmp) / "ds.jsonl"
    path.write_text(json.dumps({"problem": "q1", "answer": "6", "id": "p1"}) + "\n",
                    encoding="utf-8")
    return str(path)


def _config(provider, dataset_path, **extra):
    base = dict(dataset_path=dataset_path, provider=provider, model="m",
                max_output_tokens=100, evaluator_type="math",
                judge=JudgeConfig(model="judge-m", base_url="https://judge/v1", api_key="jk"))
    base.update(extra)
    return BenchmarkConfig(**base)


def test_capture_on_persists_reasoning_text_in_results_and_cache(monkeypatch):
    provider = _register_reasoning_provider("fake-reasoning-on")
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds, capture_reasoning=True)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            assert exp.results[0].reasoning_text == _REASONING

            # And the on-disk cache line (what actually gets banked) carries it too.
            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            body = [json.loads(ln) for ln in lines if "__ockbench_cache__" not in ln]
            assert body[0]["reasoning_text"] == _REASONING

            results, _ = aggregate_cache_file(cache_path)
            assert results[0].reasoning_text == _REASONING
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_capture_off_leaves_reasoning_text_empty_in_results_and_cache(monkeypatch):
    provider = _register_reasoning_provider("fake-reasoning-off")
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: _CountingJudge())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds, capture_reasoning=False)
            exp = BenchmarkRunner(cfg, cache_path=cache_path).run()

            assert exp.results[0].reasoning_text is None
            # The model DID answer correctly — the gate didn't break generation/
            # scoring, it only withheld the reasoning text.
            assert exp.results[0].correct is True
            assert exp.results[0].model_response == "<answer>6</answer>"

            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            body = [json.loads(ln) for ln in lines if "__ockbench_cache__" not in ln]
            assert body[0].get("reasoning_text") is None

            results, _ = aggregate_cache_file(cache_path)
            assert results[0].reasoning_text is None
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_capture_reasoning_defaults_off_at_canonical_config_level():
    # A plain BenchmarkConfig (no driver involved) must default to NOT
    # capturing — the harness driver is the layer that opts dev/canary runs in.
    cfg = BenchmarkConfig(
        dataset_path="x.jsonl", provider="gemini", model="m", max_output_tokens=100,
        evaluator_type="science",
    )
    assert cfg.capture_reasoning is False


def test_capture_reasoning_is_excluded_from_run_identity():
    # Toggling capture must NOT invalidate an existing cache — it changes what
    # gets written, not what was asked of the model or how it was scored.
    cfg_on = BenchmarkConfig(
        dataset_path="x.jsonl", provider="gemini", model="m", max_output_tokens=100,
        evaluator_type="science", capture_reasoning=True,
    )
    cfg_off = BenchmarkConfig(
        dataset_path="x.jsonl", provider="gemini", model="m", max_output_tokens=100,
        evaluator_type="science", capture_reasoning=False,
    )
    assert compute_run_identity(cfg_on) == compute_run_identity(cfg_off)
