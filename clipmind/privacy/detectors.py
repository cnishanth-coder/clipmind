"""Best-effort secret detection for clipboard text.

Each detector is a small regex plus, where the raw pattern would be too eager, a
*validator* (Luhn for card numbers, a decoded header for JWTs). The goal is to
catch the well-known credential shapes while keeping ordinary text - UUIDs, git
hashes, phone numbers, order IDs, long tickets - unflagged.

Two hard rules this module enforces:

* **Findings never carry the secret.** A :class:`Finding` records only the kind
  and the character span, so nothing here can accidentally log a value.
* **Detection is best-effort, not a guarantee.** These are heuristics; a secret
  that does not match one of the shapes below is still stored. See the README.

Usage::

    findings = detect_secrets("password=hunter2")
    # [Finding(kind='password', start=9, end=16)]
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Callable

#: Every category this module can report. The keys match the defaults in
#: :data:`clipmind.config.DEFAULT_SECRET_ACTIONS`.
ALL_KINDS: tuple[str, ...] = (
    "private_key",
    "credit_card",
    "jwt",
    "google_api_key",
    "aws_access_key",
    "github_token",
    "slack_token",
    "openai_key",
    "password",
)

#: Human-readable labels (used by the README and tests, not by the DB).
KIND_LABELS: dict[str, str] = {
    "private_key": "private key block",
    "credit_card": "credit-card number",
    "jwt": "JSON Web Token",
    "google_api_key": "Google API key",
    "aws_access_key": "AWS access key id",
    "github_token": "GitHub token",
    "slack_token": "Slack token",
    "openai_key": "OpenAI/Anthropic API key",
    "password": "password-like assignment",
}

#: Minimum length of a value on the right-hand side of ``password=`` for it to
#: count. Short values ("password: it") are usually prose, not a credential.
_MIN_ASSIGNMENT_VALUE = 6


@dataclass(frozen=True, slots=True)
class Finding:
    """One detected secret, identified by kind and span only.

    Attributes:
        kind: One of :data:`ALL_KINDS`.
        start: Inclusive start offset in the scanned text.
        end: Exclusive end offset in the scanned text.
    """

    kind: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of characters covered by the finding."""
        return self.end - self.start

    @property
    def span(self) -> tuple[int, int]:
        """Return ``(start, end)``."""
        return self.start, self.end


# ------------------------------------------------------------------- validators


def luhn_valid(digits: str) -> bool:
    """Return True when ``digits`` passes the Luhn checksum.

    Args:
        digits: A string of decimal digits (no separators).
    """
    if not digits or not digits.isdigit():
        return False
    total = 0
    for index, character in enumerate(reversed(digits)):
        value = ord(character) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _jwt_valid(token: str) -> bool:
    """Return True when ``token``'s header decodes to a JSON object with ``alg``.

    This rejects ``eyJ...``-looking ordinary text that is not a real JWT.
    """
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        return False
    header = parts[0]
    padding = "=" * (-len(header) % 4)
    try:
        decoded = base64.urlsafe_b64decode(header + padding)
        payload = json.loads(decoded)
    except Exception:  # noqa: BLE001 - any decode/parse failure means "not a JWT"
        return False
    return isinstance(payload, dict) and "alg" in payload


# --------------------------------------------------------------------- patterns

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
    r"[\s\S]+?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----",
    re.IGNORECASE,
)

_AWS_ACCESS_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"
)

_GITHUB_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"
    r"(?![A-Za-z0-9_])"
)

_SLACK_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9-])"
)

_OPENAI_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
)

_GOOGLE_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_\-]{35}(?![A-Za-z0-9_])"
)

_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r"(?![A-Za-z0-9_-])"
)

# A run of 13-19 digits, optionally grouped with spaces/hyphens. The lookarounds
# stop it matching a slice of a longer word or number.
_CARD_CANDIDATE_RE = re.compile(r"(?<![\w-])(\d[0-9 \-]{10,25}\d)(?![\w-])")

_PASSWORD_KEYWORD = (
    r"(?:password|passwd|pwd|secret|api[ _\-]?key|apikey|access[ _\-]?key"
    r"|secret[ _\-]?key|client[ _\-]?secret|auth[ _\-]?token|private[ _\-]?key)"
)
_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])" + _PASSWORD_KEYWORD + r"\b\s*[:=]\s*"
    r"(?:\"([^\"\n]+)\"|'([^'\n]+)'|([^\s,;\"']+))",
    re.IGNORECASE,
)

#: Values that look like placeholders rather than real secrets.
_PLACEHOLDER_VALUES = frozenset({"none", "null", "true", "false", "empty", "password"})


# ------------------------------------------------------------------- detectors


def _find_private_keys(text: str) -> list[Finding]:
    """Find PEM ``-----BEGIN ... PRIVATE KEY-----`` blocks."""
    return [Finding("private_key", m.start(), m.end()) for m in _PRIVATE_KEY_RE.finditer(text)]


def _find_aws_access_keys(text: str) -> list[Finding]:
    """Find AWS access key ids (``AKIA``/``ASIA`` + 16 chars)."""
    return [Finding("aws_access_key", m.start(), m.end()) for m in _AWS_ACCESS_KEY_RE.finditer(text)]


def _find_github_tokens(text: str) -> list[Finding]:
    """Find GitHub personal/installation tokens."""
    return [Finding("github_token", m.start(), m.end()) for m in _GITHUB_TOKEN_RE.finditer(text)]


def _find_slack_tokens(text: str) -> list[Finding]:
    """Find Slack tokens (``xoxb-``/``xoxp-``/...)."""
    return [Finding("slack_token", m.start(), m.end()) for m in _SLACK_TOKEN_RE.finditer(text)]


def _find_openai_keys(text: str) -> list[Finding]:
    """Find OpenAI/Anthropic style ``sk-...`` keys."""
    return [Finding("openai_key", m.start(), m.end()) for m in _OPENAI_KEY_RE.finditer(text)]


def _find_google_api_keys(text: str) -> list[Finding]:
    """Find Google API keys (``AIza`` + 35 characters)."""
    return [
        Finding("google_api_key", m.start(), m.end())
        for m in _GOOGLE_API_KEY_RE.finditer(text)
    ]


def _find_jwts(text: str) -> list[Finding]:
    """Find three-segment JWTs whose header decodes to a signed JWT header."""
    findings: list[Finding] = []
    for match in _JWT_RE.finditer(text):
        if _jwt_valid(match.group(0)):
            findings.append(Finding("jwt", match.start(), match.end()))
    return findings


def _find_credit_cards(text: str) -> list[Finding]:
    """Find 13-19 digit runs that pass the Luhn checksum."""
    findings: list[Finding] = []
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ \-]", "", match.group(1))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            start, end = match.span(1)
            findings.append(Finding("credit_card", start, end))
    return findings


def _find_passwords(text: str) -> list[Finding]:
    """Find ``password=...``/``secret: ...`` style assignments."""
    findings: list[Finding] = []
    for match in _ASSIGNMENT_RE.finditer(text):
        for group in (1, 2, 3):
            if match.group(group) is None:
                continue
            value = match.group(group)
            if len(value) < _MIN_ASSIGNMENT_VALUE:
                break
            if value.lower() in _PLACEHOLDER_VALUES:
                break
            if set(value) <= {".", "*", "-", "x", "X"}:
                break
            start, end = match.span(group)
            findings.append(Finding("password", start, end))
            break
    return findings


#: Detectors in *priority* order: an earlier detector wins when two findings
#: overlap (for example a JWT that is also the value of a ``token:`` assignment).
_DETECTORS: tuple[Callable[[str], list[Finding]], ...] = (
    _find_private_keys,
    _find_credit_cards,
    _find_jwts,
    _find_google_api_keys,
    _find_aws_access_keys,
    _find_github_tokens,
    _find_slack_tokens,
    _find_openai_keys,
    _find_passwords,
)

_KIND_PRIORITY = {kind: index for index, kind in enumerate(
    (
        "private_key",
        "credit_card",
        "jwt",
        "google_api_key",
        "aws_access_key",
        "github_token",
        "slack_token",
        "openai_key",
        "password",
    )
)}


def detect_secrets(text: str) -> list[Finding]:
    """Return the secrets found in ``text``.

    Overlapping findings are resolved by keeping the earliest one and, among
    equal starts, the higher-priority kind (see :data:`_DETECTORS`). The result
    is sorted by start offset and never contains the secret itself.

    Args:
        text: The text to scan. ``None``/empty returns ``[]``.

    Returns:
        A list of :class:`Finding`, possibly empty.
    """
    if not text:
        return []
    candidates: list[Finding] = []
    for detector in _DETECTORS:
        candidates.extend(detector(text))
    if not candidates:
        return []
    # Sort by start; for a tie prefer the higher-priority (more specific) kind.
    candidates.sort(key=lambda f: (f.start, _KIND_PRIORITY.get(f.kind, 99)))
    resolved: list[Finding] = []
    last_end = -1
    for finding in candidates:
        if finding.start >= last_end:
            resolved.append(finding)
            last_end = finding.end
    return resolved
