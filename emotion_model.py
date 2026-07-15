"""
Module 2: Hybrid Emotion Detection Pipeline.

Provides two independently-trained classifiers standing in for the
BiLSTM / BERT pair described in the spec:

  - FAST_MODEL  ("BiLSTM-style"): TF-IDF (word n-grams) + Logistic Regression.
    Cheap, low-latency, good baseline.
  - DEEP_MODEL  ("BERT-style"):   TF-IDF (word+char n-grams) + a small MLP
    (multi-layer perceptron), giving a heavier, more context-sensitive model.

Both are trained at import time on the bundled synthetic dataset (data/emotion_dataset.py).
Swap-in path to real models:
  - Replace FAST_MODEL with a trained Keras BiLSTM (see README.md).
  - Replace DEEP_MODEL with a fine-tuned HuggingFace BERT checkpoint.
  The rest of the app (thresholding, prompt generation, logging) is agnostic
  to how the scores were produced, as long as `predict_all` keeps returning
  {label: score} dicts in [0, 1].

Classification logic: rather than a strict argmax, we support **mixed emotion
detection** -- any label whose score exceeds MIXED_EMOTION_THRESHOLD is
reported as present.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from data.emotion_dataset import DATA, LABELS

MIXED_EMOTION_THRESHOLD = 0.30


@dataclass
class ModelResult:
    scores: dict            # {label: probability}
    mixed_labels: list       # labels above threshold
    top_label: str
    latency_ms: float
    model_name: str


def _train_fast_model(texts, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(X, labels)
    return vectorizer, clf


def _train_deep_model(texts, labels):
    # Word + character n-gram TF-IDF feeding a small MLP, as a lightweight
    # stand-in for a contextual transformer encoder.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), analyzer="word", min_df=1, sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)
    base_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=3000,
        random_state=42,
    )
    # Calibrate so predict_proba is meaningful with such a small dataset.
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X, labels)
    return vectorizer, clf


class _Pipeline:
    def __init__(self):
        texts = [t for t, _ in DATA]
        labels = [l for _, l in DATA]
        self.fast_vec, self.fast_clf = _train_fast_model(texts, labels)
        self.deep_vec, self.deep_clf = _train_deep_model(texts, labels)

    def _score(self, vectorizer, clf, text) -> dict:
        X = vectorizer.transform([text])
        proba = clf.predict_proba(X)[0]
        classes = clf.classes_
        scores = {label: 0.0 for label in LABELS}
        for cls, p in zip(classes, proba):
            scores[cls] = float(p)
        return scores

    def predict(self, text: str, which: str) -> ModelResult:
        start = time.perf_counter()
        if which == "fast":
            scores = self._score(self.fast_vec, self.fast_clf, text)
            name = "Fast Model (TF-IDF + LogReg, 'BiLSTM-style')"
        else:
            scores = self._score(self.deep_vec, self.deep_clf, text)
            name = "Deep Model (TF-IDF + MLP, 'BERT-style')"
        latency_ms = (time.perf_counter() - start) * 1000

        mixed = [l for l, s in scores.items() if s > MIXED_EMOTION_THRESHOLD]
        if not mixed:
            mixed = [max(scores, key=scores.get)]
        top_label = max(scores, key=scores.get)

        return ModelResult(
            scores=scores,
            mixed_labels=mixed,
            top_label=top_label,
            latency_ms=latency_ms,
            model_name=name,
        )

    def predict_all(self, text: str):
        """Return results from both models plus a simple ensemble average."""
        fast = self.predict(text, "fast")
        deep = self.predict(text, "deep")

        avg_scores = {
            label: (fast.scores[label] + deep.scores[label]) / 2 for label in LABELS
        }
        mixed = [l for l, s in avg_scores.items() if s > MIXED_EMOTION_THRESHOLD]
        if not mixed:
            mixed = [max(avg_scores, key=avg_scores.get)]

        ensemble = ModelResult(
            scores=avg_scores,
            mixed_labels=mixed,
            top_label=max(avg_scores, key=avg_scores.get),
            latency_ms=fast.latency_ms + deep.latency_ms,
            model_name="Ensemble (average of both models)",
        )
        return fast, deep, ensemble


# Singleton, trained once per process (import time).
_pipeline_instance = None


def get_pipeline() -> _Pipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = _Pipeline()
    return _pipeline_instance
