"""Tests for :mod:`clipmind.embeddings.codec`."""

from __future__ import annotations

import numpy as np
import pytest

from clipmind.embeddings.codec import (
    cosine_similarity,
    decode_vector,
    encode_vector,
    normalize,
    stack_vectors,
)


def test_round_trip_preserves_values_exactly() -> None:
    vector = np.array([0.1, -0.25, 3.5, 0.0], dtype=np.float32)
    decoded = decode_vector(encode_vector(vector))
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, vector)


def test_encoded_size_is_four_bytes_per_float() -> None:
    blob = encode_vector(np.zeros(384, dtype=np.float32))
    assert len(blob) == 384 * 4


def test_decode_returns_independent_writable_copy() -> None:
    blob = encode_vector(np.array([1.0, 2.0], dtype=np.float32))
    first = decode_vector(blob)
    first[0] = 99.0
    assert decode_vector(blob)[0] == 1.0


def test_decode_empty_blob_returns_empty_vector() -> None:
    assert decode_vector(b"").shape == (0,)


def test_stack_vectors_builds_matrix() -> None:
    blobs = [encode_vector(np.array([1.0, 0.0])), encode_vector(np.array([0.0, 1.0]))]
    matrix = stack_vectors(blobs)
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32


def test_stack_vectors_empty_returns_empty_matrix() -> None:
    assert stack_vectors([]).shape == (0, 0)


def test_stack_vectors_rejects_mixed_dimensions() -> None:
    blobs = [encode_vector(np.zeros(3, dtype=np.float32)), encode_vector(np.zeros(4, dtype=np.float32))]
    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        stack_vectors(blobs)


def test_normalize_makes_unit_norm_rows() -> None:
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])


def test_normalize_leaves_zero_vectors_untouched() -> None:
    matrix = normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(matrix).all()
    assert np.linalg.norm(matrix) == 0.0


def test_cosine_similarity_is_bounded_and_ordered() -> None:
    matrix = normalize(np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32))
    scores = cosine_similarity(np.array([1.0, 0.0], dtype=np.float32), matrix)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.7071, abs=1e-4)
    assert scores[2] == pytest.approx(-1.0)


def test_cosine_similarity_handles_empty_matrix() -> None:
    scores = cosine_similarity(np.ones(4, dtype=np.float32), np.zeros((0, 0), dtype=np.float32))
    assert scores.shape == (0,)
