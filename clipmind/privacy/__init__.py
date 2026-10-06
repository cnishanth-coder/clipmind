"""Privacy filters (Phase 4).

Detects well-known credential shapes (API keys, JWTs, private keys, card
numbers, password-like assignments), applies a per-category ``skip``/``mask``
policy, and enforces a source-application ignore list. Everything runs locally;
a detected secret is never logged or stored when the policy says otherwise.

The public surface is intentionally small:

* :func:`detect_secrets` - scan text and return :class:`Finding` (kind + span).
* :class:`PrivacyFilter` - the monitor's entry point (ignore list + policy).
* :class:`PrivacyPolicy` / :class:`PrivacyOutcome` - detection → action.
"""

from __future__ import annotations

from .detectors import ALL_KINDS, KIND_LABELS, Finding, detect_secrets, luhn_valid
from .policy import (
    MASK_TEMPLATE,
    PrivacyFilter,
    PrivacyOutcome,
    PrivacyPolicy,
    mask_text,
    matches_ignore_list,
    should_ignore_source_app,
)

__all__ = [
    "ALL_KINDS",
    "KIND_LABELS",
    "MASK_TEMPLATE",
    "Finding",
    "PrivacyFilter",
    "PrivacyOutcome",
    "PrivacyPolicy",
    "detect_secrets",
    "luhn_valid",
    "mask_text",
    "matches_ignore_list",
    "should_ignore_source_app",
]
