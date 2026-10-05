"""Central configuration for ClipMind.

Configuration currently lives in code with environment-variable overrides. A
``config.toml`` file (hotkey, search weights, retention, ignore list) is added
in Phase 4; :class:`Config` is the single place every module reads from, so
adding TOML support later will not ripple through the codebase.

Environment overrides
---------------------
``CLIPMIND_DATA_DIR``      Directory holding the SQLite database.
``CLIPMIND_DB``            Full path to the SQLite database file.
``CLIPMIND_POLL_INTERVAL`` Clipboard poll interval in seconds.
``CLIPMIND_MAX_CHARS``     Maximum number of characters stored per entry.
``CLIPMIND_EMBED_MODEL``   sentence-transformers model name.
``CLIPMIND_EMBED_DEVICE``  Torch device for the embedding model (``cpu``/``mps``).
``CLIPMIND_MIN_SCORE``     Minimum cosine similarity for a search hit.
``CLIPMIND_TOP_K``         Default number of search results.
``CLIPMIND_SEMANTIC_WEIGHT`` Weight of the semantic score in hybrid ranking.
``CLIPMIND_KEYWORD_WEIGHT``  Weight of the keyword score in hybrid ranking.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

APP_NAME = "clipmind"

#: Entries longer than this are truncated (or skipped) before storage.
DEFAULT_MAX_CHARS = 10_000

#: How many seconds between clipboard reads.
DEFAULT_POLL_INTERVAL = 0.5

#: Default number of rows returned by ``clipmind list``.
DEFAULT_LIST_LIMIT = 20

#: Policy for oversized entries: ``"truncate"`` or ``"skip"``.
DEFAULT_OVERSIZE_POLICY = "truncate"

#: Local sentence-transformers model used for embeddings (384-dim).
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: Texts per forward pass when embedding.
DEFAULT_EMBEDDING_BATCH_SIZE = 32

#: Entries shorter than this are never embedded (nothing to say semantically).
DEFAULT_EMBEDDING_MIN_CHARS = 3

#: Content types that are never embedded: a bare number carries no semantics
#: and would add noise to the vector space. Keyword search still finds them.
DEFAULT_EMBEDDING_SKIP_TYPES: frozenset[str] = frozenset({"number"})

#: Minimum cosine similarity for a result to count as a real match.
DEFAULT_MIN_SCORE = 0.25

#: Default number of results returned by ``clipmind search``.
DEFAULT_TOP_K = 5

#: Weight of the semantic score in the Phase 3 hybrid ranking.
DEFAULT_SEMANTIC_WEIGHT = 0.7

#: Weight of the keyword (FTS5 BM25) score in the Phase 3 hybrid ranking.
DEFAULT_KEYWORD_WEIGHT = 0.3

#: How many candidates each retriever contributes before fusion. The merged
#: list is then trimmed to ``top_k``, so fetching a few extra candidates keeps
#: a strong keyword-only hit from being crowded out by weak semantic hits.
DEFAULT_CANDIDATE_POOL = 20

#: Local-time hour boundaries for the day-part phrases. Ranges are half-open
#: ``[start, end)``. "night" wraps past midnight, which is why it is 21 -> 5.
DEFAULT_MORNING = (5, 12)
DEFAULT_AFTERNOON = (12, 17)
DEFAULT_EVENING = (17, 21)
DEFAULT_NIGHT = (21, 5)

#: Week start for "this week"/"last week"; Monday by Python's convention.
DEFAULT_WEEK_START = 0


def default_data_dir() -> Path:
    """Return the per-user data directory used for the ClipMind database.

    Respects ``CLIPMIND_DATA_DIR`` when set, otherwise falls back to the
    platform-native application data location (``~/Library/Application Support``
    on macOS, ``%APPDATA%`` on Windows, ``$XDG_DATA_HOME`` elsewhere).
    """
    override = os.environ.get("CLIPMIND_DATA_DIR")
    if override:
        return Path(override).expanduser()

    home = Path.home()
    if os.name == "nt":  # pragma: no cover - exercised only on Windows
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return base / APP_NAME
    if sys.platform == "darwin":  # pragma: no cover - macOS only
        return home / "Library" / "Application Support" / APP_NAME
    base = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    return base / APP_NAME


def default_db_path() -> Path:
    """Return the default SQLite database path (honours ``CLIPMIND_DB``)."""
    override = os.environ.get("CLIPMIND_DB")
    if override:
        return Path(override).expanduser()
    return default_data_dir() / "clipmind.db"


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime configuration for ClipMind.

    Attributes:
        db_path: Location of the SQLite database file.
        poll_interval: Seconds between clipboard polls.
        max_chars: Maximum stored characters per entry.
        oversize_policy: ``"truncate"`` to shorten long entries, ``"skip"`` to
            ignore them entirely.
        list_limit: Default row count for the ``list`` CLI command.
        dedup: When True, entries whose content hash already exists are skipped.
        embedding_model: Name of the sentence-transformers model used locally.
        embedding_batch_size: Number of texts per embedding forward pass.
        embedding_min_chars: Minimum length for an entry to be embedded.
        embedding_skip_types: Content types that are never embedded.
        embedding_device: Torch device (``"cpu"``, ``"mps"``, ``"cuda"``) or
            ``None`` to let sentence-transformers choose.
        min_score: Minimum combined (hybrid) score for a result to be kept.
        search_top_k: Default number of search results.
        semantic_weight: Weight of the semantic score in ``final_score``.
        keyword_weight: Weight of the keyword score in ``final_score``.
        candidate_pool: Candidates each retriever fetches before fusion.
        day_part_morning: Local ``(start_hour, end_hour)`` for "this morning".
        day_part_afternoon: Local ``(start_hour, end_hour)`` for "this afternoon".
        day_part_evening: Local ``(start_hour, end_hour)`` for "this evening".
        day_part_night: Local ``(start_hour, end_hour)`` for "last night"; the
            end may be earlier than the start (wrap past midnight).
        week_starts_on: Weekday index treated as the first day of the week
            (``0`` = Monday, ``6`` = Sunday).
    """

    db_path: Path = None  # type: ignore[assignment]
    poll_interval: float = DEFAULT_POLL_INTERVAL
    max_chars: int = DEFAULT_MAX_CHARS
    oversize_policy: str = DEFAULT_OVERSIZE_POLICY
    list_limit: int = DEFAULT_LIST_LIMIT
    dedup: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_min_chars: int = DEFAULT_EMBEDDING_MIN_CHARS
    embedding_skip_types: frozenset[str] = DEFAULT_EMBEDDING_SKIP_TYPES
    embedding_device: str | None = None
    min_score: float = DEFAULT_MIN_SCORE
    search_top_k: int = DEFAULT_TOP_K
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    candidate_pool: int = DEFAULT_CANDIDATE_POOL
    day_part_morning: tuple[int, int] = DEFAULT_MORNING
    day_part_afternoon: tuple[int, int] = DEFAULT_AFTERNOON
    day_part_evening: tuple[int, int] = DEFAULT_EVENING
    day_part_night: tuple[int, int] = DEFAULT_NIGHT
    week_starts_on: int = DEFAULT_WEEK_START

    def __post_init__(self) -> None:
        if self.db_path is None:
            object.__setattr__(self, "db_path", default_db_path())
        if self.oversize_policy not in {"truncate", "skip"}:
            raise ValueError(
                f"oversize_policy must be 'truncate' or 'skip', got {self.oversize_policy!r}"
            )

    def with_overrides(self, **changes: object) -> "Config":
        """Return a copy of this config with selected fields replaced.

        Used by the CLI to apply command-line flags without having to re-list
        every field (which silently drops new options whenever one is added).
        """
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables."""
        cfg = cls()
        env_map: dict[str, tuple[str, type]] = {
            "CLIPMIND_POLL_INTERVAL": ("poll_interval", float),
            "CLIPMIND_MAX_CHARS": ("max_chars", int),
            "CLIPMIND_EMBED_MODEL": ("embedding_model", str),
            "CLIPMIND_EMBED_DEVICE": ("embedding_device", str),
            "CLIPMIND_MIN_SCORE": ("min_score", float),
            "CLIPMIND_TOP_K": ("search_top_k", int),
            "CLIPMIND_SEMANTIC_WEIGHT": ("semantic_weight", float),
            "CLIPMIND_KEYWORD_WEIGHT": ("keyword_weight", float),
            "CLIPMIND_CANDIDATE_POOL": ("candidate_pool", int),
        }
        changes: dict[str, object] = {}
        for env_var, (field_name, cast) in env_map.items():
            raw = os.environ.get(env_var)
            if raw is not None:
                changes[field_name] = cast(raw)
        return replace(cfg, **changes) if changes else cfg  # type: ignore[arg-type]

    def ensure_data_dir(self) -> Path:
        """Create the data directory if needed and return it."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return self.db_path.parent
