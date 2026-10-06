"""Central configuration for ClipMind.

Configuration is resolved in three layers, later layers winning:

1. **Code defaults** - the constants and dataclass defaults in this module.
2. **``config.toml``** - an optional user file (see :func:`config_file_path`),
   validated against :data:`_FIELD_VALIDATORS` with clear error messages.
3. **Environment variables** - ``CLIPMIND_*`` overrides, kept working exactly as
   they did before ``config.toml`` existed.

:class:`Config` is the single place every module reads from. Use
:meth:`Config.load` to build one from all three layers; :meth:`Config.from_env`
still exists for callers that only want defaults plus environment.

``config.toml`` keys map one-to-one onto :class:`Config` field names, so
everything documented below is settable. For example::

    poll_interval = 0.5
    retention_days = 30
    ignore_source_apps = ["1Password", "Bitwarden", "KeePassXC"]
    [secret_actions]
    aws_access_key = "mask"
    credit_card = "skip"

Environment overrides
---------------------
``CLIPMIND_DATA_DIR``      Directory holding the SQLite database.
``CLIPMIND_DB``            Full path to the SQLite database file.
``CLIPMIND_CONFIG``        Path to ``config.toml`` (overrides the default).
``CLIPMIND_POLL_INTERVAL`` Clipboard poll interval in seconds.
``CLIPMIND_MAX_CHARS``     Maximum number of characters stored per entry.
``CLIPMIND_EMBED_MODEL``   sentence-transformers model name.
``CLIPMIND_EMBED_DEVICE``  Torch device for the embedding model (``cpu``/``mps``).
``CLIPMIND_MIN_SCORE``     Minimum cosine similarity for a search hit.
``CLIPMIND_TOP_K``         Default number of search results.
``CLIPMIND_SEMANTIC_WEIGHT`` Weight of the semantic score in hybrid ranking.
``CLIPMIND_KEYWORD_WEIGHT``  Weight of the keyword score in hybrid ranking.
``CLIPMIND_CANDIDATE_POOL``  Candidates each retriever contributes before fusion.
``CLIPMIND_RETENTION_DAYS``  Auto-delete entries older than this many days (0 = off).
``CLIPMIND_PRIVACY``         ``0`` to disable secret detection, ``1`` to enable.
``CLIPMIND_IGNORE_APPS``     Comma-separated source-app ignore list.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

APP_NAME = "clipmind"

#: Name of the optional user configuration file.
CONFIG_FILENAME = "config.toml"

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

#: Whether secret detection/masking is on by default.
DEFAULT_PRIVACY_ENABLED = True

#: Per-category secret policy. ``"skip"`` never stores the entry; ``"mask"``
#: stores it with the secret replaced by ``[REDACTED:<kind>]``. Private keys,
#: card numbers and password-like assignments are dropped; API keys and JWTs
#: are masked (Phase 4 defaults).
DEFAULT_SECRET_ACTIONS: dict[str, str] = {
    "private_key": "skip",
    "credit_card": "skip",
    "password": "skip",
    "aws_access_key": "mask",
    "github_token": "mask",
    "slack_token": "mask",
    "openai_key": "mask",
    "google_api_key": "mask",
    "jwt": "mask",
}

#: Applications ClipMind never captures from. Matching is case-insensitive and
#: substring-based ("1password" matches "1Password 8"). Detection is best-effort
#: on platforms where the frontmost app cannot be read (see ``capture/source_app.py``).
DEFAULT_IGNORE_SOURCE_APPS: tuple[str, ...] = (
    "1Password",
    "Bitwarden",
    "KeePassXC",
    "Keychain Access",
    "LastPass",
)

#: Action values accepted for ``secret_actions``.
SECRET_ACTIONS = frozenset({"skip", "mask"})


class ConfigError(ValueError):
    """Raised when ``config.toml`` contains an unknown key or a bad value."""


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


def config_file_path() -> Path:
    """Return the location of ``config.toml``.

    Resolution order:

    1. ``CLIPMIND_CONFIG`` (explicit file path), when set.
    2. ``$XDG_CONFIG_HOME/clipmind/config.toml``, when ``XDG_CONFIG_HOME`` is set.
    3. ``~/.config/clipmind/config.toml``.

    The file is not required to exist; a missing file simply means "use the
    defaults". This function only computes the path.
    """
    override = os.environ.get("CLIPMIND_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base).expanduser() / APP_NAME / CONFIG_FILENAME
    return Path.home() / ".config" / APP_NAME / CONFIG_FILENAME


# ------------------------------------------------------------------- validation


def _expect(key: str, raw: object, types: tuple[type, ...], expected: str) -> object:
    """Return ``raw`` when it has one of ``types``, else raise ConfigError."""
    # bool is a subclass of int; guard against TOML ``true`` for int fields.
    if isinstance(raw, bool) and bool not in types:
        raise ConfigError(f"{key!r} must be {expected} (got a boolean)")
    if not isinstance(raw, types):
        raise ConfigError(
            f"{key!r} must be {expected} (got {type(raw).__name__})"
        )
    return raw


def _positive_int(key: str, raw: object, *, minimum: int = 1) -> int:
    value = _expect(key, raw, (int,), "an integer")
    value = int(value)
    if value < minimum:
        raise ConfigError(f"{key!r} must be >= {minimum} (got {value})")
    return value


def _positive_float(key: str, raw: object, *, minimum: float = 0.0) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{key!r} must be a number")
    value = float(raw)
    if value < minimum:
        raise ConfigError(f"{key!r} must be >= {minimum} (got {value})")
    return value


def _unit_float(key: str, raw: object) -> float:
    value = _positive_float(key, raw, minimum=0.0)
    if value > 1.0:
        raise ConfigError(f"{key!r} must be between 0 and 1 (got {value})")
    return value


def _string(key: str, raw: object) -> str:
    value = _expect(key, raw, (str,), "a string")
    return str(value)


def _bool(key: str, raw: object) -> bool:
    if not isinstance(raw, bool):
        raise ConfigError(f"{key!r} must be true or false")
    return raw


def _string_list(key: str, raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigError(f"{key!r} must be an array of strings")
    return tuple(raw)


def _string_frozenset(key: str, raw: object) -> frozenset[str]:
    return frozenset(_string_list(key, raw))


def _hour_window(key: str, raw: object) -> tuple[int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw)
    ):
        raise ConfigError(f"{key!r} must be a two-element array of hours, e.g. [5, 12]")
    start, end = int(raw[0]), int(raw[1])
    if not (0 <= start <= 23 and 0 <= end <= 23):
        raise ConfigError(f"{key!r} hours must be between 0 and 23 (got {raw})")
    return start, end


def _secret_actions(key: str, raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{key!r} must be a table of category = 'skip'|'mask'")
    actions: dict[str, str] = {}
    for name, value in raw.items():
        if value not in SECRET_ACTIONS:
            raise ConfigError(
                f"{key}.{name} must be 'skip' or 'mask' (got {value!r})"
            )
        actions[str(name)] = str(value)
    return actions


def _oversize_policy(key: str, raw: object) -> str:
    value = _string(key, raw)
    if value not in {"truncate", "skip"}:
        raise ConfigError(f"{key!r} must be 'truncate' or 'skip' (got {value!r})")
    return value


def _retention_days(key: str, raw: object) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{key!r} must be an integer number of days or omitted")
    if raw < 0:
        raise ConfigError(f"{key!r} must be >= 0 (got {raw})")
    return None if raw == 0 else int(raw)


def _week_start(key: str, raw: object) -> int:
    value = _expect(key, raw, (int,), "an integer 0-6")
    value = int(value)
    if not 0 <= value <= 6:
        raise ConfigError(f"{key!r} must be between 0 (Monday) and 6 (Sunday)")
    return value


def _optional_string(key: str, raw: object) -> Optional[str]:
    if raw is None:
        return None
    return _string(key, raw)


def _db_path(key: str, raw: object) -> Path:
    return Path(_string(key, raw)).expanduser()


#: Maps a ``config.toml`` key to a converter that validates and normalises it.
_FIELD_VALIDATORS: dict[str, Callable[[str, object], object]] = {
    "db_path": _db_path,
    "poll_interval": lambda k, v: _positive_float(k, v, minimum=0.01),
    "max_chars": lambda k, v: _positive_int(k, v),
    "oversize_policy": _oversize_policy,
    "list_limit": lambda k, v: _positive_int(k, v, minimum=0),
    "dedup": _bool,
    "embedding_model": _string,
    "embedding_batch_size": lambda k, v: _positive_int(k, v),
    "embedding_min_chars": lambda k, v: _positive_int(k, v, minimum=0),
    "embedding_skip_types": _string_frozenset,
    "embedding_device": _optional_string,
    "min_score": _unit_float,
    "search_top_k": lambda k, v: _positive_int(k, v),
    "semantic_weight": lambda k, v: _positive_float(k, v),
    "keyword_weight": lambda k, v: _positive_float(k, v),
    "candidate_pool": lambda k, v: _positive_int(k, v),
    "day_part_morning": _hour_window,
    "day_part_afternoon": _hour_window,
    "day_part_evening": _hour_window,
    "day_part_night": _hour_window,
    "week_starts_on": _week_start,
    "privacy_enabled": _bool,
    "secret_actions": _secret_actions,
    "ignore_source_apps": _string_list,
    "retention_days": _retention_days,
}


def _validate_config_data(data: Mapping[str, object]) -> dict[str, object]:
    """Validate a parsed ``config.toml`` mapping into ``Config`` overrides.

    Args:
        data: The table parsed by :mod:`tomllib`.

    Returns:
        A dict of validated field values suitable for :func:`dataclasses.replace`.

    Raises:
        ConfigError: On an unknown key or an invalid value. The message names the
            offending key so the user can fix it without reading the source.
    """
    unknown = sorted(set(data) - set(_FIELD_VALIDATORS))
    if unknown:
        known = ", ".join(sorted(_FIELD_VALIDATORS))
        raise ConfigError(
            f"unknown key(s) in config.toml: {', '.join(unknown)}. "
            f"Valid keys are: {known}"
        )
    changes: dict[str, object] = {}
    for key, raw in data.items():
        changes[key] = _FIELD_VALIDATORS[key](key, raw)
    return changes


def default_config_toml() -> str:
    """Return a documented ``config.toml`` template with the defaults.

    Written to stdout by ``clipmind config show --template`` and embedded in the
    README, so the defaults are documented in exactly one place.
    """
    return (
        "# ClipMind configuration. Every key is optional; omitted keys use the\n"
        "# defaults shown here. Environment variables (CLIPMIND_*) override this\n"
        "# file.\n"
        "\n"
        "# --- capture ---\n"
        f"poll_interval = {DEFAULT_POLL_INTERVAL}\n"
        f"max_chars = {DEFAULT_MAX_CHARS}\n"
        f'oversize_policy = "{DEFAULT_OVERSIZE_POLICY}"\n'
        f"dedup = true\n"
        f"list_limit = {DEFAULT_LIST_LIMIT}\n"
        "\n"
        "# --- search ---\n"
        f"min_score = {DEFAULT_MIN_SCORE}\n"
        f"search_top_k = {DEFAULT_TOP_K}\n"
        f"semantic_weight = {DEFAULT_SEMANTIC_WEIGHT}\n"
        f"keyword_weight = {DEFAULT_KEYWORD_WEIGHT}\n"
        f"candidate_pool = {DEFAULT_CANDIDATE_POOL}\n"
        "\n"
        "# --- embeddings ---\n"
        f'embedding_model = "{DEFAULT_EMBEDDING_MODEL}"\n'
        f"embedding_batch_size = {DEFAULT_EMBEDDING_BATCH_SIZE}\n"
        f"embedding_min_chars = {DEFAULT_EMBEDDING_MIN_CHARS}\n"
        'embedding_skip_types = ["number"]\n'
        "# embedding_device = \"cpu\"\n"
        "\n"
        "# --- privacy (Phase 4) ---\n"
        f"privacy_enabled = {'true' if DEFAULT_PRIVACY_ENABLED else 'false'}\n"
        "# Auto-delete entries older than N days (0 or omitted disables it).\n"
        "# retention_days = 30\n"
        "# Source applications never captured from (case-insensitive substring).\n"
        'ignore_source_apps = ["1Password", "Bitwarden", "KeePassXC", '
        '"Keychain Access", "LastPass"]\n'
        "\n"
        "# Per-category policy for detected secrets: \"skip\" (never store the\n"
        "# entry) or \"mask\" (store with the secret replaced by [REDACTED:kind]).\n"
        "[secret_actions]\n"
        'private_key = "skip"\n'
        'credit_card = "skip"\n'
        'password = "skip"\n'
        'aws_access_key = "mask"\n'
        'github_token = "mask"\n'
        'slack_token = "mask"\n'
        'openai_key = "mask"\n'
        'google_api_key = "mask"\n'
        'jwt = "mask"\n'
    )


def load_config_file(path: Optional[Path | str] = None) -> dict[str, object]:
    """Read and validate ``config.toml``.

    Args:
        path: Explicit file to read. Defaults to :func:`config_file_path`.

    Returns:
        The validated overrides, or ``{}`` when the file does not exist.

    Raises:
        ConfigError: If the file exists but is not valid TOML, has an unknown
            key, or has an invalid value.
    """
    config_path = Path(path) if path is not None else config_file_path()
    if not config_path.exists():
        return {}
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {config_path}: {exc}") from exc
    return _validate_config_data(raw)


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
        privacy_enabled: When False, secret detection is skipped entirely.
        secret_actions: Per-category ``"skip"``/``"mask"`` policy.
        ignore_source_apps: Source applications never captured from
            (case-insensitive substring match).
        retention_days: Auto-delete entries older than this many days. ``None``
            disables automatic purging.
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
    privacy_enabled: bool = DEFAULT_PRIVACY_ENABLED
    secret_actions: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SECRET_ACTIONS)
    )
    ignore_source_apps: tuple[str, ...] = DEFAULT_IGNORE_SOURCE_APPS
    retention_days: Optional[int] = None

    def __post_init__(self) -> None:
        if self.db_path is None:
            object.__setattr__(self, "db_path", default_db_path())
        if self.oversize_policy not in {"truncate", "skip"}:
            raise ValueError(
                f"oversize_policy must be 'truncate' or 'skip', got {self.oversize_policy!r}"
            )
        bad = {name: action for name, action in self.secret_actions.items()
               if action not in SECRET_ACTIONS}
        if bad:
            raise ValueError(f"secret_actions must be 'skip' or 'mask': {bad}")

    def with_overrides(self, **changes: object) -> "Config":
        """Return a copy of this config with selected fields replaced.

        Used by the CLI to apply command-line flags without having to re-list
        every field (which silently drops new options whenever one is added).
        """
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, base: Optional["Config"] = None) -> "Config":
        """Build a :class:`Config` from environment variables.

        Args:
            base: Starting configuration. Defaults to a fresh :class:`Config`
                (code defaults); pass the result of loading ``config.toml`` to
                layer the environment on top of the file.
        """
        cfg = base if base is not None else cls()
        env_map: dict[str, tuple[str, Callable[[str], object]]] = {
            "CLIPMIND_POLL_INTERVAL": ("poll_interval", float),
            "CLIPMIND_MAX_CHARS": ("max_chars", int),
            "CLIPMIND_EMBED_MODEL": ("embedding_model", str),
            "CLIPMIND_EMBED_DEVICE": ("embedding_device", str),
            "CLIPMIND_MIN_SCORE": ("min_score", float),
            "CLIPMIND_TOP_K": ("search_top_k", int),
            "CLIPMIND_SEMANTIC_WEIGHT": ("semantic_weight", float),
            "CLIPMIND_KEYWORD_WEIGHT": ("keyword_weight", float),
            "CLIPMIND_CANDIDATE_POOL": ("candidate_pool", int),
            "CLIPMIND_RETENTION_DAYS": ("retention_days", _parse_retention_env),
            "CLIPMIND_PRIVACY": ("privacy_enabled", _parse_bool_env),
            "CLIPMIND_IGNORE_APPS": ("ignore_source_apps", _parse_app_list_env),
        }
        changes: dict[str, object] = {}
        for env_var, (field_name, cast) in env_map.items():
            raw = os.environ.get(env_var)
            if raw is not None:
                changes[field_name] = cast(raw)
        return replace(cfg, **changes) if changes else cfg  # type: ignore[arg-type]

    @classmethod
    def load(
        cls,
        path: Optional[Path | str] = None,
        *,
        use_env: bool = True,
    ) -> "Config":
        """Resolve the full configuration: defaults, then file, then env.

        Args:
            path: Explicit ``config.toml`` path; defaults to
                :func:`config_file_path`. A missing file is not an error.
            use_env: When False, skip ``CLIPMIND_*`` environment overrides.

        Returns:
            A fully resolved :class:`Config`.

        Raises:
            ConfigError: When the file exists but is invalid.
        """
        cfg = cls()
        data = load_config_file(path)
        if data:
            cfg = replace(cfg, **data)  # type: ignore[arg-type]
        if use_env:
            cfg = cls.from_env(cfg)
        return cfg

    def ensure_data_dir(self) -> Path:
        """Create the data directory if needed and return it."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return self.db_path.parent


def _parse_bool_env(raw: str) -> bool:
    """Parse a boolean environment variable (``1/0/true/false/yes/no``)."""
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"expected a boolean (1/0/true/false), got {raw!r}")


def _parse_retention_env(raw: str) -> Optional[int]:
    """Parse ``CLIPMIND_RETENTION_DAYS`` (0 or empty disables retention)."""
    text = raw.strip()
    if text == "":
        return None
    try:
        days = int(text)
    except ValueError as exc:
        raise ConfigError(
            f"CLIPMIND_RETENTION_DAYS must be an integer, got {raw!r}"
        ) from exc
    if days < 0:
        raise ConfigError(f"CLIPMIND_RETENTION_DAYS must be >= 0 (got {days})")
    return None if days == 0 else days


def _parse_app_list_env(raw: str) -> Sequence[str]:
    """Parse a comma-separated ignore-list environment variable."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())
