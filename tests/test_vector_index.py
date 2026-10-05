"""Tests for :mod:`clipmind.search.vector_index`."""

from __future__ import annotations

import numpy as np
import pytest

from clipmind.search.vector_index import NumpyVectorIndex, VectorIndex, VectorHit

DOCS = np.array(
    [
        [1.0, 0.0, 0.0],  # key 10
        [0.0, 1.0, 0.0],  # key 11
        [0.9, 0.1, 0.0],  # key 12
    ],
    dtype=np.float32,
)


def _index() -> NumpyVectorIndex:
    return NumpyVectorIndex([10, 11, 12], DOCS)


def test_satisfies_protocol() -> None:
    assert isinstance(_index(), VectorIndex)


def test_size_and_dimension() -> None:
    index = _index()
    assert index.size == 3
    assert index.dimension == 3


def test_empty_index_reports_zero_and_searches_empty() -> None:
    index = NumpyVectorIndex()
    assert index.size == 0
    assert index.search(np.zeros(0, dtype=np.float32), 5) == []


def test_search_orders_by_descending_similarity() -> None:
    hits = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 3)
    assert [hit.key for hit in hits] == [10, 12, 11]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_respects_top_k() -> None:
    hits = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)
    assert len(hits) == 2
    assert [hit.key for hit in hits] == [10, 12]


def test_top_k_zero_or_negative_returns_nothing() -> None:
    index = _index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert index.search(query, 0) == []
    assert index.search(query, -3) == []


def test_top_k_larger_than_index_is_clamped() -> None:
    hits = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 99)
    assert len(hits) == 3


def test_allowed_keys_filters_candidates() -> None:
    hits = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 3, allowed_keys={12})
    assert [hit.key for hit in hits] == [12]


def test_allowed_keys_with_no_match_returns_empty() -> None:
    hits = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 3, allowed_keys={99})
    assert hits == []


def test_hits_are_vector_hit_instances() -> None:
    hit = _index().search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)[0]
    assert isinstance(hit, VectorHit)
    assert isinstance(hit.key, int)
    assert isinstance(hit.score, float)


def test_query_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dims but index holds"):
        _index().search(np.zeros(5, dtype=np.float32), 1)


def test_add_appends_vectors() -> None:
    index = _index()
    index.add([13], np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    assert index.size == 4
    assert index.search(np.array([0.0, 0.0, 1.0], dtype=np.float32), 1)[0].key == 13


def test_add_rejects_key_count_mismatch() -> None:
    with pytest.raises(ValueError, match="keys for"):
        _index().add([1, 2], np.zeros((1, 3), dtype=np.float32))


def test_add_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        _index().add([1], np.zeros((1, 5), dtype=np.float32))


def test_unnormalised_input_still_scores_by_cosine() -> None:
    index = NumpyVectorIndex([1], np.array([[10.0, 0.0]], dtype=np.float32))
    hits = index.search(np.array([0.5, 0.0], dtype=np.float32), 1)
    assert hits[0].score == pytest.approx(1.0)
