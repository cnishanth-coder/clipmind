"""Detector tests: true positives, false positives, and validators.

The false-positive table matters as much as the true-positive one: ordinary
text (UUIDs, git hashes, phone numbers, order IDs, timestamps) must never be
flagged, which is why card numbers are Luhn-checked and JWTs are decoded.
"""

from __future__ import annotations

import base64
import json

import pytest

from clipmind.privacy.detectors import Finding, detect_secrets, luhn_valid


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_jwt(header: dict | None = None, payload: dict | None = None) -> str:
    """Build a syntactically valid (unsigned) JWT for detector tests."""
    header = header or {"alg": "HS256", "typ": "JWT"}
    payload = payload or {"sub": "1234567890", "name": "Test"}
    head = _b64url(json.dumps(header).encode("utf-8"))
    body = _b64url(json.dumps(payload).encode("utf-8"))
    return f"{head}.{body}.{_b64url(b'signature')}"


PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz\n"
    "-----END RSA PRIVATE KEY-----"
)

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "A" * 36
GITHUB_PAT = "github_pat_" + "B" * 30
SLACK_TOKEN = "xoxb-123456789012-abcdefghijkl"
OPENAI_KEY = "sk-" + "a" * 32
ANTHROPIC_KEY = "sk-ant-api03-" + "c" * 28
GOOGLE_KEY = "AIza" + "C" * 35


TRUE_POSITIVES = [
    ("aws_access_key", f"my key is {AWS_KEY}"),
    ("github_token", f"token {GITHUB_TOKEN}"),
    ("github_token", f"token {GITHUB_PAT}"),
    ("slack_token", f"token {SLACK_TOKEN}"),
    ("openai_key", f"token {OPENAI_KEY}"),
    ("openai_key", f"token {ANTHROPIC_KEY}"),
    ("google_api_key", f"key {GOOGLE_KEY}"),
    ("jwt", f"Authorization: Bearer {make_jwt()}"),
    ("private_key", PRIVATE_KEY),
    ("credit_card", "card 4242 4242 4242 4242"),
    ("credit_card", "card 4111111111111111"),
    ("password", "password=hunter2"),
    ("password", "secret: hunter2secret"),
    ("password", 'api_key = "abcd1234efgh"'),
]


@pytest.mark.parametrize("kind, text", TRUE_POSITIVES)
def test_true_positives(kind: str, text: str) -> None:
    findings = detect_secrets(text)
    assert any(f.kind == kind for f in findings), (kind, text)


@pytest.mark.parametrize("kind, text", TRUE_POSITIVES)
def test_finding_reports_the_kind_span_only(kind: str, text: str) -> None:
    finding = next(f for f in detect_secrets(text) if f.kind == kind)
    # A Finding carries no secret text - only a kind and offsets.
    assert set(Finding.__slots__) == {"kind", "start", "end"}
    assert 0 <= finding.start < finding.end <= len(text)
    assert finding.span == (finding.start, finding.end)
    assert finding.length == finding.end - finding.start


FALSE_POSITIVES = [
    "550e8400-e29b-41d4-a716-446655440000",  # UUID
    "e4d909c290d0fb1ca068ffaddf22cbd0a1b2c3d4",  # git hash
    "+1 (415) 555-2671",  # phone number
    "Order ID: 1234567890123",  # 13 digits, invalid Luhn
    "1234567812345678",  # 16 digits, invalid Luhn
    "2026-10-05T14:33:00Z",  # ISO timestamp
    "Meeting at 4pm in room 12B",
    "The quick brown fox jumps over the lazy dog.",
    "https://example.com/orders/123456789012345678",
    "0000000000000000000000000000000000000000",
    "color: #1a2b3c",
    "sha256: deadbeefcafe",
]


@pytest.mark.parametrize("text", FALSE_POSITIVES)
def test_false_positives(text: str) -> None:
    assert detect_secrets(text) == []


def test_multiple_secrets_are_all_reported() -> None:
    text = f"key {AWS_KEY} and card 4242 4242 4242 4242"
    kinds = {f.kind for f in detect_secrets(text)}
    assert kinds == {"aws_access_key", "credit_card"}


def test_findings_are_sorted_by_start_and_do_not_overlap() -> None:
    text = f"first {AWS_KEY} then {GITHUB_TOKEN}"
    findings = detect_secrets(text)
    assert len(findings) == 2
    assert findings[0].start < findings[1].start
    assert findings[0].end <= findings[1].start


def test_empty_and_none_like_input() -> None:
    assert detect_secrets("") == []
    assert detect_secrets("no secrets here") == []


def test_short_password_value_is_not_flagged() -> None:
    # Prose such as "password: it" is not a credential.
    assert detect_secrets("password: it") == []


def test_placeholder_assignment_is_not_flagged() -> None:
    assert detect_secrets("password=none") == []
    assert detect_secrets("password=********") == []


def test_fake_jwt_header_is_rejected() -> None:
    # Three dot-separated eyJ... segments whose header is not JSON with "alg".
    fake = _b64url(b"not json") + "." + _b64url(b"body") + "." + _b64url(b"sig")
    assert detect_secrets(fake) == []


# ------------------------------------------------------------------ Luhn


@pytest.mark.parametrize(
    "digits, expected",
    [
        ("4242424242424242", True),
        ("4111111111111111", True),
        ("5555555555554444", True),  # Mastercard test number
        ("4242424242424241", False),
        ("1234567890123", False),
        ("1234567812345678", False),
        ("", False),
        ("abcd", False),
        ("4242-4242", False),
    ],
)
def test_luhn(digits: str, expected: bool) -> None:
    assert luhn_valid(digits) is expected
