"""Serialisation helpers for embedding vectors.

Embeddings are stored in the existing ``items.embedding`` BLOB column as raw
little-endian ``float32`` bytes (384 * 4 = 1536 bytes for ``all-MiniLM-L6-v2``;
768 * 4 = 3072 for ``all-mpnet-base-v2``).
Raw ``float32`` is used rather than ``numpy.save``/``.npy`` because it needed no
header, is trivially portable, and keeps the database readable from any language.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Dtype used on disk and in memory for every vector.
VECTOR_DTYPE = np.float32


def encode_vector(vector: Sequence[float] | np.ndarray) -> bytes:
    """Serialise one embedding into ``float32`` bytes for the BLOB column."""
    array = np.asarray(vector, dtype=VECTOR_DTYPE).reshape(-1)
    return array.tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    """Deserialise a single embedding BLOB into a writable 1-D array.

    Args:
        blob: Bytes produced by :func:`encode_vector`.

    Returns:
        A 1-D ``float32`` array. The result is a copy, so it is safe to mutate
        and to stack with other vectors.
    """
    if not blob:
        return np.zeros(0, dtype=VECTOR_DTYPE)
    return np.frombuffer(blob, dtype=VECTOR_DTYPE).copy()


def stack_vectors(blobs: Sequence[bytes]) -> np.ndarray:
    """Stack many embedding BLOBs into a ``(n, dim)`` matrix.

    Args:
        blobs: Serialised vectors, all of the same dimension.

    Returns:
        A ``(n, dim)`` ``float32`` matrix, or a ``(0, 0)`` array when ``blobs``
        is empty.
    """
    if not blobs:
        return np.zeros((0, 0), dtype=VECTOR_DTYPE)
    vectors = [decode_vector(blob) for blob in blobs]
    dims = {vector.shape[0] for vector in vectors}
    if len(dims) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {sorted(dims)}")
    return np.vstack(vectors).astype(VECTOR_DTYPE, copy=False)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalised copy of ``matrix``.

    Zero vectors are left untouched instead of producing ``NaN``.
    """
    array = np.asarray(matrix, dtype=VECTOR_DTYPE)
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    np.maximum(norms, np.finfo(VECTOR_DTYPE).eps, out=norms)
    return array / norms


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity between one ``query`` vector and each row of ``matrix``.

    Both inputs are normalised first, so the result is the cosine of the angle
    between them regardless of how they were produced.
    """
    query_array = normalize(np.asarray(query, dtype=VECTOR_DTYPE).reshape(1, -1))
    matrix_array = normalize(matrix)
    if matrix_array.size == 0 or query_array.size == 0:
        return np.zeros(matrix_array.shape[0], dtype=VECTOR_DTYPE)
    return (matrix_array @ query_array[0]).astype(VECTOR_DTYPE, copy=False)
