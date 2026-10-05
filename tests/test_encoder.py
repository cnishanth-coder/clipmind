"""Tests for :mod:`clipmind.embeddings.encoder` (no model is ever loaded)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from clipmind.config import Config
from clipmind.embeddings.encoder import (
    EmbeddingModelUnavailable,
    SentenceTransformerEmbedder,
    get_embedder,
    reset_embedder_cache,
)


def test_embedder_is_lazy_and_reports_dimension_before_loading() -> None:
    embedder = SentenceTransformerEmbedder()
    assert embedder.is_loaded is False
    assert embedder.dimension == 384


def test_empty_input_does_not_load_the_model() -> None:
    embedder = SentenceTransformerEmbedder()
    matrix = embedder.embed_texts([])
    assert matrix.shape == (0, 384)
    assert matrix.dtype == np.float32
    assert embedder.is_loaded is False


def test_get_embedder_returns_a_cached_singleton(monkeypatch) -> None:
    reset_embedder_cache()
    try:
        config = Config(db_path=":memory:")
        first = get_embedder(config)
        second = get_embedder(config)
        assert first is second
        assert first.is_loaded is False  # creating it must not load the model

        other = get_embedder(config.with_overrides(embedding_device="cpu"))
        assert other is not first  # a different device is a different instance
    finally:
        reset_embedder_cache()


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch) -> None:
    # ``None`` in sys.modules makes `from sentence_transformers import ...`
    # raise ImportError even when the package is installed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder()
    with pytest.raises(EmbeddingModelUnavailable) as excinfo:
        embedder.embed_texts(["hello"])

    message = str(excinfo.value)
    assert "sentence-transformers is not installed" in message
    assert "pip install -r requirements.txt" in message
    assert "--no-embed" in message
    assert embedder.is_loaded is False


def test_model_load_failure_is_wrapped(monkeypatch) -> None:
    class Exploding:
        def __init__(self, *args, **kwargs):
            raise OSError("no such file or directory: model cache")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", type("M", (), {"SentenceTransformer": Exploding})
    )
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    with pytest.raises(EmbeddingModelUnavailable) as excinfo:
        embedder.embed_texts(["hello"])

    message = str(excinfo.value)
    assert "could not load embedding model 'all-MiniLM-L6-v2'" in message
    assert "no such file or directory" in message
