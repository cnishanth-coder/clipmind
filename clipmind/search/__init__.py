"""Search: semantic, keyword and hybrid retrieval with time-aware queries.

Phase 2 added semantic search; Phase 3 adds the FTS5 keyword retriever, the
weighted fusion of the two, and a deterministic time-phrase parser.
"""

from __future__ import annotations

from .embedding_index import EmbeddingIndex
from .hybrid import HybridOutcome, HybridResult, HybridSearcher, hybrid_search
from .keyword import KeywordHit, KeywordSearcher, keyword_search, sanitize_fts_query
from .semantic import SearchResult, SemanticSearcher, search_semantic
from .timeparse import TimeQuery, parse_time_query
from .vector_index import NumpyVectorIndex, VectorHit, VectorIndex

__all__ = [
    "HybridOutcome",
    "HybridResult",
    "HybridSearcher",
    "EmbeddingIndex",
    "KeywordHit",
    "KeywordSearcher",
    "NumpyVectorIndex",
    "SearchResult",
    "SemanticSearcher",
    "TimeQuery",
    "VectorHit",
    "VectorIndex",
    "hybrid_search",
    "keyword_search",
    "parse_time_query",
    "sanitize_fts_query",
    "search_semantic",
]
