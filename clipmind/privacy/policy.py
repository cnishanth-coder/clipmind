"""Privacy policy: what to do with a detected secret, and who to ignore.

The policy turns :func:`~clipmind.privacy.detectors.detect_secrets` findings into
one of three outcomes:

``allow``  No secret (or privacy disabled). Store the text as-is.
``mask``   At least one secret is configured as ``"mask"`` and none as
           ``"skip"``. Store the text with every secret replaced by
           ``[REDACTED:<kind>]``.
``skip``   At least one secret is configured as ``"skip"``. Do not store the
           entry at all.

The masked text is what the rest of ClipMind sees, so the raw secret never
reaches the database, the FTS index, the embedding queue, or any log line.
Nothing in this module ever includes a matched value in an exception or message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..config import DEFAULT_SECRET_ACTIONS, Config
from .detectors import Finding, detect_secrets

#: Template used to replace a secret in masked text.
MASK_TEMPLATE = "[REDACTED:{kind}]"


@dataclass(frozen=True, slots=True)
class PrivacyOutcome:
    """Result of applying the privacy policy to one piece of text.

    Attributes:
        action: ``"allow"``, ``"mask"`` or ``"skip"``.
        content: The text to store. Unchanged for ``allow``, masked for
            ``mask``; for ``skip`` it is the original text and the caller must
            not store it.
        findings: The detected secrets (kind + span only - never the value).
    """

    action: str
    content: str
    findings: tuple[Finding, ...] = ()

    @property
    def allowed(self) -> bool:
        """True when the text may be stored unchanged."""
        return self.action == "allow"

    @property
    def masked(self) -> bool:
        """True when a secret was replaced in :attr:`content`."""
        return self.action == "mask"

    @property
    def skipped(self) -> bool:
        """True when the entry must not be stored."""
        return self.action == "skip"

    @property
    def kinds(self) -> tuple[str, ...]:
        """Distinct secret kinds found, in detection order."""
        seen: list[str] = []
        for finding in self.findings:
            if finding.kind not in seen:
                seen.append(finding.kind)
        return tuple(seen)

    def describe(self) -> str:
        """Return a short, secret-free summary suitable for logs or the CLI."""
        if self.action == "allow":
            return "no secrets detected"
        kinds = ", ".join(self.kinds)
        if self.action == "skip":
            return f"skipped: {kinds}"
        return f"masked: {kinds}"


def mask_text(text: str, findings: tuple[Finding, ...] | list[Finding]) -> str:
    """Replace each finding's span in ``text`` with ``[REDACTED:<kind>]``.

    Replacements run right-to-left so earlier spans stay valid.
    """
    masked = text
    for finding in sorted(findings, key=lambda f: f.start, reverse=True):
        masked = masked[: finding.start] + MASK_TEMPLATE.format(kind=finding.kind) + masked[finding.end :]
    return masked


class PrivacyPolicy:
    """Apply the configured secret policy (detect, then skip or mask)."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Create a policy from ``config`` (defaults when omitted)."""
        self.config = config or Config()

    def action_for(self, kind: str) -> str:
        """Return the configured action for ``kind``, falling back to defaults.

        A partial ``[secret_actions]`` table therefore never leaves a category
        unhandled: anything it omits keeps its built-in default.
        """
        return self.config.secret_actions.get(
            kind, DEFAULT_SECRET_ACTIONS.get(kind, "mask")
        )

    def evaluate(self, text: str) -> PrivacyOutcome:
        """Detect secrets in ``text`` and decide how to handle it.

        Args:
            text: Normalised clipboard text.

        Returns:
            A :class:`PrivacyOutcome`. When privacy is disabled, or nothing is
            detected, the outcome is ``allow`` and :attr:`PrivacyOutcome.content`
            is ``text`` unchanged.
        """
        if not self.config.privacy_enabled:
            return PrivacyOutcome("allow", text)
        findings = detect_secrets(text)
        if not findings:
            return PrivacyOutcome("allow", text)

        actions = {self.action_for(finding.kind) for finding in findings}
        as_tuple = tuple(findings)
        if "skip" in actions:
            return PrivacyOutcome("skip", text, as_tuple)
        return PrivacyOutcome("mask", mask_text(text, as_tuple), as_tuple)


def matches_ignore_list(source_app: Optional[str], apps: Iterable[str]) -> bool:
    """Return True when ``source_app`` matches an entry of the ignore list.

    Matching is case-insensitive and substring-based, so ``"1Password 8"``
    matches the default ``"1Password"``. An unknown source app (``None`` or an
    empty string) never matches - ClipMind degrades gracefully rather than
    failing closed.
    """
    if not source_app:
        return False
    needle = source_app.strip().lower()
    if not needle:
        return False
    for app in apps:
        candidate = str(app).strip().lower()
        if candidate and candidate in needle:
            return True
    return False


def should_ignore_source_app(source_app: Optional[str], config: Optional[Config] = None) -> bool:
    """Return True when capture from ``source_app`` should be skipped."""
    config = config or Config()
    return matches_ignore_list(source_app, config.ignore_source_apps)


class PrivacyFilter:
    """Convenience wrapper combining detection and the source-app ignore list.

    The monitor holds one of these so capture code has a single dependency:
    ask the filter whether an entry is ignored, then whether its text should be
    stored, masked, or skipped.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Create a filter from ``config`` (defaults when omitted)."""
        self.config = config or Config()
        self.policy = PrivacyPolicy(self.config)

    def ignored(self, source_app: Optional[str]) -> bool:
        """Return True when ``source_app`` is on the ignore list."""
        return should_ignore_source_app(source_app, self.config)

    def evaluate(self, text: str) -> PrivacyOutcome:
        """Detect and classify secrets in ``text``."""
        return self.policy.evaluate(text)
