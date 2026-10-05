"""Local text embeddings.

``all-MiniLM-L6-v2`` runs entirely on this machine. The model is downloaded once
by ``sentence-transformers`` and cached on disk; there are no API calls and no
data ever leaves the machine. Importing this package does **not** import torch -
the heavy work happens on the first :meth:`~clipmind.embeddings.encoder.SentenceTransformerEmbedder.embed_texts`.
"""

from __future__ import annotations

from .codec import (
    VECTOR_DTYPE,
    cosine_similarity,
    decode_vector,
    encode_vector,
    normalize,
    stack_vectors,
)
from .encoder import (
    DEFAULT_EMBEDDING_DIM,
    Embedder,
    EmbeddingModelUnavailable,
    SentenceTransformerEmbedder,
    get_embedder,
    is_embeddable,
    reset_embedder_cache,
)
from .service import (
    EmbeddingService,
    EmbedItem,
    ReindexReport,
    embed_batch,
    pending_items,
    reindex_pending,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "VECTOR_DTYPE",
    "EmbedItem",
    "Embedder",
    "EmbeddingModelUnavailable",
    "EmbeddingService",
    "ReindexReport",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
    "decode_vector",
    "embed_batch",
    "encode_vector",
    "get_embedder",
    "is_embeddable",
    "normalize",
    "pending_items",
    "reindex_pending",
    "reset_embedder_cache",
    "stack_vectors",
]
