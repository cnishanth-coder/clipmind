"""Local text embedding.

The only backend in Phase 2 is :class:`SentenceTransformerEmbedder`, which runs a
local ``sentence-transformers`` model (``all-MiniLM-L6-v2``, 384 dimensions, by
default; the configured model is selectable, e.g. ``all-mpnet-base-v2``) on this
machine. The model is downloaded **once** from Hugging Face and cached by
``sentence-transformers`` in ``~/.cache/huggingface`` (or ``HF_HOME``); after that
no network access happens.

Torch and ``sentence-transformers`` are imported lazily so that importing the
package, and running the test-suite with a fake embedder, never pays for them.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import Config, model_prefixes
from .codec import VECTOR_DTYPE, normalize

logger = logging.getLogger(__name__)

#: Dimension of ``all-MiniLM-L6-v2`` output vectors (the default model). A hint
#: used before the model is materialised; corrected from the loaded model.
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the local embedding model cannot be loaded.

    Covers a missing ``sentence-transformers`` install and a failed first-time
    model download, so callers can report one actionable message instead of a
    raw import or network traceback.
    """


class EmbeddingModelMismatch(EmbeddingModelUnavailable):
    """Raised when stored embeddings were produced by a different model.

    Embeddings from two models are not comparable (and usually not even the same
    dimension), so mixing them silently would produce nonsense rankings. The
    database records the model name that produced its vectors; when the
    configured model differs, the caller must re-embed with
    ``clipmind reindex --force``.
    """


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into L2-normalised vectors.

    Implemented by :class:`SentenceTransformerEmbedder` in production and by a
    tiny deterministic fake in the tests, which keeps the suite fast and offline.
    """

    @property
    def dimension(self) -> int:
        """Dimensionality of the vectors this embedder produces."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into a ``(len(texts), dimension)`` float32 matrix.

        Rows must be L2-normalised so cosine similarity is a plain dot product.
        """
        ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around a local ``sentence-transformers`` model.

    The heavy import and the model load both happen on the first call to
    :meth:`embed_texts`, which is where the one-time model download occurs.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        """Configure the embedder without loading anything.

        Args:
            model_name: sentence-transformers model identifier.
            device: Torch device string, or ``None`` to let the library choose.
            batch_size: Texts per forward pass.
            dimension: Expected output dimension (used before the model loads).
            query_prefix: Text prepended to queries (e.g. BGE's instruction).
            document_prefix: Text prepended to documents (e.g. E5's ``passage:``).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._dimension = dimension
        self._model = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ loading

    @property
    def is_loaded(self) -> bool:
        """True once the underlying model has been materialised in memory."""
        return self._model is not None

    def _load(self):
        """Import sentence-transformers and load the model (thread-safe, once).

        Raises:
            EmbeddingModelUnavailable: If the library is not installed or the
                model cannot be loaded (for example the one-time ~90 MB weight
                download fails because the machine is offline).
        """
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                logger.info("loading embedding model %s (first use)", self.model_name)
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingModelUnavailable(
                        "sentence-transformers is not installed. Install the "
                        "embedding dependencies with `pip install -r requirements.txt`, "
                        "or use `clipmind watch --no-embed` and the other commands, "
                        "which do not need an embedding model."
                    ) from exc

                kwargs = {"device": self.device} if self.device else {}
                try:
                    model = SentenceTransformer(self.model_name, **kwargs)
                except Exception as exc:
                    raise EmbeddingModelUnavailable(
                        f"could not load embedding model {self.model_name!r}: {exc}. "
                        "The first use downloads ~90 MB, so this needs network access "
                        "once; afterwards the model is read from the local cache."
                    ) from exc
                dimension = model.get_sentence_embedding_dimension()
                if dimension:
                    self._dimension = int(dimension)
                self._model = model
                logger.info("embedding model ready (%d dims)", self._dimension)
        return self._model

    # ------------------------------------------------------------------ embedding

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents into an L2-normalised ``float32`` matrix.

        The optional document prefix is applied here; queries go through
        :meth:`embed_queries` so a model that wants different text on each side
        (BGE, E5) is handled correctly.

        Args:
            texts: Strings to embed. An empty sequence returns a ``(0, dim)``
                matrix without touching the model.

        Returns:
            A ``(len(texts), dimension)`` ``float32`` array with unit-norm rows.
        """
        return self._encode(texts, self.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed query strings, applying the model's query prefix when set."""
        return self._encode(texts, self.query_prefix)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        """Embed ``texts`` (optionally prefixed) into a unit-norm matrix."""
        if not texts:
            return np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)
        if prefix:
            texts = [prefix + text for text in texts]

        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=VECTOR_DTYPE)
        if matrix.ndim == 1:  # single input -> single row
            matrix = matrix.reshape(1, -1)
        # Re-normalise defensively: the contract callers rely on is unit norm.
        return normalize(matrix)


def embed_queries(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Embed queries, using :meth:`embed_queries` when the embedder has it.

    Test doubles only implement :meth:`embed_texts`; real models implement the
    query-specific method so their retrieval prefix is applied correctly.
    """
    query_method = getattr(embedder, "embed_queries", None)
    if query_method is not None:
        return query_method(texts)
    return embedder.embed_texts(texts)


# Model instances are expensive, so one is shared per (model, device) pair.
_EMBEDDER_CACHE: dict[tuple[str, Optional[str]], SentenceTransformerEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_embedder(config: Optional[Config] = None) -> SentenceTransformerEmbedder:
    """Return the process-wide (lazily created) embedder for ``config``.

    Calling this never loads the model; loading happens on first
    :meth:`SentenceTransformerEmbedder.embed_texts`.

    Query/document prefixes come from the config when set, otherwise from the
    built-in preset for the model (see :data:`clipmind.config.MODEL_PREFIXES`).

    Args:
        config: Configuration supplying the model name, device and batch size.

    Returns:
        The cached :class:`SentenceTransformerEmbedder`.
    """
    config = config or Config()
    default_query, default_document = model_prefixes(config.embedding_model)
    query_prefix = config.embedding_query_prefix or default_query
    document_prefix = config.embedding_document_prefix or default_document
    key = (config.embedding_model, config.embedding_device)
    with _CACHE_LOCK:
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                config.embedding_model,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
            _EMBEDDER_CACHE[key] = embedder
        return embedder


def reset_embedder_cache() -> None:
    """Drop the cached embedders (used by tests)."""
    with _CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def is_embeddable(
    text: Optional[str],
    content_type: str = "text",
    config: Optional[Config] = None,
) -> bool:
    """Return True when ``text`` should be embedded at all.

    Short strings (``< config.embedding_min_chars``) and types listed in
    ``config.embedding_skip_types`` (``number`` by default) carry no useful
    semantics and are skipped; keyword search still finds them in Phase 3.

    Args:
        text: The stored content.
        content_type: The entry's detected content type.
        config: Configuration supplying the length/type policy.
    """
    if text is None or content_type in (config or Config()).embedding_skip_types:
        return False
    return len(text.strip()) >= (config or Config()).embedding_min_chars
