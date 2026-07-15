"""
Strategy 3: Zero-Shot Classification for the 16-emotion label set.

Uses a HuggingFace `zero-shot-classification` pipeline (NLI-based, e.g.
`facebook/bart-large-mnli`) to score arbitrary input text against the
project's 16 candidate emotion labels -- no training required.

This is a drop-in third model: it returns the same `ModelResult` shape
used by `emotion_model.py`, so it can sit alongside the fast/deep models
in the UI's comparison view without touching thresholding or logging code.

Requires `transformers` + a backend (`torch` or `tensorflow`), which pull
in a multi-hundred-MB model download from huggingface.co on first use.
That network access isn't available in this sandbox, so `is_available()`
lets calling code fail gracefully and fall back to the fast/deep models.

Usage:
    from zero_shot_model import get_zero_shot_pipeline

    zsp = get_zero_shot_pipeline()
    if zsp.is_available():
        result = zsp.predict("I've read this chapter five times and it still isn't clicking.")
        print(result.top_label, result.scores)
"""

import time
from dataclasses import dataclass

from data.emotion_dataset import LABELS
from emotion_model import ModelResult, MIXED_EMOTION_THRESHOLD

# Model choice: bart-large-mnli is the standard, well-validated zero-shot
# NLI backbone. Swap for a smaller model (e.g. valhalla/distilbart-mnli-12-3)
# if latency/memory is tight.
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

# NLI-style hypothesis template. Zero-shot NLI pipelines classify by testing
# "This example is {label}" as an entailment hypothesis against the input
# text for every candidate label, then softmax-normalizing the entailment
# scores. Phrasing it in first person matches how the source texts read
# (first-person student statements about their study session).
HYPOTHESIS_TEMPLATE = "This person is feeling {}."


class _ZeroShotPipeline:
    """Lazily loads the HF pipeline on first predict() call, not at import time."""

    def __init__(self):
        self._clf = None
        self._load_error = None

    def _ensure_loaded(self):
        if self._clf is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self._clf = hf_pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL_NAME,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via is_available()/last_error
            self._load_error = exc

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._clf is not None

    def last_error(self) -> str:
        return str(self._load_error) if self._load_error else ""

    def predict(self, text: str) -> ModelResult:
        self._ensure_loaded()
        if self._clf is None:
            raise RuntimeError(
                "Zero-shot classifier unavailable "
                f"({self._load_error}). Call is_available() before predict()."
            )

        start = time.perf_counter()
        raw = self._clf(
            text,
            candidate_labels=LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True,  # score each label independently (0-1), not a forced distribution
        )
        latency_ms = (time.perf_counter() - start) * 1000

        scores = {label: 0.0 for label in LABELS}
        for label, score in zip(raw["labels"], raw["scores"]):
            scores[label] = float(score)

        mixed = [l for l, s in scores.items() if s > MIXED_EMOTION_THRESHOLD]
        if not mixed:
            mixed = [max(scores, key=scores.get)]

        return ModelResult(
            scores=scores,
            mixed_labels=mixed,
            top_label=max(scores, key=scores.get),
            latency_ms=latency_ms,
            model_name=f"Zero-Shot ({ZERO_SHOT_MODEL_NAME})",
        )


_instance = None


def get_zero_shot_pipeline() -> _ZeroShotPipeline:
    global _instance
    if _instance is None:
        _instance = _ZeroShotPipeline()
    return _instance
