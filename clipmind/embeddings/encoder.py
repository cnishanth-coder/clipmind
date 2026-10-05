"""Local text embedding.

The only backend in Phase 2 is :class:`SentenceTransformerEmbedder`, which runs
``all-MiniLM-L6-v2`` (384 dimensions) on the local machine. The model is
downloaded **once** from Hugging Face and cached by ``sentence-transformers`` in
``~/.cache/huggingface`` (or ``HF_HOME``); after that no network access happens.

Torch and ``sentence-transformers`` are imported lazily so that importing the
package, and running the test-suite with a fake embedder, never pays for them.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..config import Config
from .codec import VECTOR_DTYPE, normalize

logger = logging.getLogger(__name__)

#: Dimension of ``all-MiniLM-L6-v2`` output vectors.
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the local embedding model cannot be loaded.

    Covers a missing ``sentence-transformers`` install and a failed first-time
    model download, so callers can report one actionable message instead of a
    raw import or network traceback.
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
    ) -> None:
        """Configure the embedder without loading anything.

        Args:
            model_name: sentence-transformers model identifier.
            device: Torch device string, or ``None`` to let the library choose.
            batch_size: Texts per forward pass.
            dimension: Expected output dimension (used before the model loads).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
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
        """Embed ``texts`` into an L2-normalised ``float32`` matrix.

        Args:
            texts: Strings to embed. An empty sequence returns a ``(0, dim)``
                matrix without touching the model.

        Returns:
            A ``(len(texts), dimension)`` ``float32`` array with unit-norm rows.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)

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


# Model instances are expensive, so one is shared per (model, device) pair.
_EMBEDDER_CACHE: dict[tuple[str, Optional[str]], SentenceTransformerEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_embedder(config: Optional[Config] = None) -> SentenceTransformerEmbedder:
    """Return the process-wide (lazily created) embedder for ``config``.

    Calling this never loads the model; loading happens on first
    :meth:`SentenceTransformerEmbedder.embed_texts`.

    Args:
        config: Configuration supplying the model name, device and batch size.

    Returns:
        The cached :class:`SentenceTransformerEmbedder`.
    """
    config = config or Config()
    key = (config.embedding_model, config.embedding_device)
    with _CACHE_LOCK:
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                config.embedding_model,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
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
