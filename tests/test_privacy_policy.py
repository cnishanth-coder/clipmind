"""Policy tests: detection -> skip/mask, masking spans, and the ignore list."""

from __future__ import annotations

import pytest

from clipmind.config import Config
from clipmind.privacy.policy import (
    PrivacyPolicy,
    mask_text,
    matches_ignore_list,
    should_ignore_source_app,
)

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _config(**kwargs) -> Config:
    return Config(db_path=":memory:", **kwargs)


# ------------------------------------------------------------- default policy


def test_default_actions_match_the_spec() -> None:
    policy = PrivacyPolicy(_config())
    assert policy.action_for("private_key") == "skip"
    assert policy.action_for("credit_card") == "skip"
    assert policy.action_for("password") == "skip"
    assert policy.action_for("aws_access_key") == "mask"
    assert policy.action_for("github_token") == "mask"
    assert policy.action_for("slack_token") == "mask"
    assert policy.action_for("openai_key") == "mask"
    assert policy.action_for("google_api_key") == "mask"
    assert policy.action_for("jwt") == "mask"


def test_normal_text_is_allowed_unchanged() -> None:
    text = "just an ordinary note"
    outcome = PrivacyPolicy(_config()).evaluate(text)
    assert outcome.allowed
    assert outcome.content == text
    assert outcome.kinds == ()
    assert outcome.describe() == "no secrets detected"


def test_password_is_skipped() -> None:
    outcome = PrivacyPolicy(_config()).evaluate("password=hunter2")
    assert outcome.skipped
    assert outcome.action == "skip"
    assert outcome.kinds == ("password",)


def test_api_key_is_masked_and_secret_removed() -> None:
    text = f"my key is {AWS_KEY} ok"
    outcome = PrivacyPolicy(_config()).evaluate(text)
    assert outcome.masked
    assert AWS_KEY not in outcome.content
    assert "[REDACTED:aws_access_key]" in outcome.content
    assert outcome.content == "my key is [REDACTED:aws_access_key] ok"


def test_mask_keeps_surrounding_text_intact() -> None:
    text = f"before {AWS_KEY} after"
    outcome = PrivacyPolicy(_config()).evaluate(text)
    assert outcome.content.startswith("before ")
    assert outcome.content.endswith(" after")


def test_describe_never_contains_the_secret() -> None:
    outcome = PrivacyPolicy(_config()).evaluate(f"key {AWS_KEY}")
    assert AWS_KEY not in outcome.describe()
    assert "aws_access_key" in outcome.describe()


def test_skip_wins_over_mask_when_both_are_present() -> None:
    text = f"key {AWS_KEY} and password=hunter2"
    outcome = PrivacyPolicy(_config()).evaluate(text)
    assert outcome.skipped  # password defaults to skip


# ------------------------------------------------------ configurability


def test_partial_action_table_falls_back_to_defaults() -> None:
    # Only password is overridden to mask; everything else keeps its default.
    policy = PrivacyPolicy(_config(secret_actions={"password": "mask"}))
    assert policy.action_for("password") == "mask"
    assert policy.action_for("credit_card") == "skip"
    assert policy.action_for("aws_access_key") == "mask"


def test_can_mask_everything() -> None:
    actions = {kind: "mask" for kind in (
        "private_key", "credit_card", "password", "aws_access_key",
        "github_token", "slack_token", "openai_key", "google_api_key", "jwt",
    )}
    outcome = PrivacyPolicy(_config(secret_actions=actions)).evaluate("password=hunter2")
    assert outcome.masked
    assert "hunter2" not in outcome.content


def test_can_skip_everything() -> None:
    actions = {kind: "skip" for kind in (
        "private_key", "credit_card", "password", "aws_access_key",
        "github_token", "slack_token", "openai_key", "google_api_key", "jwt",
    )}
    outcome = PrivacyPolicy(_config(secret_actions=actions)).evaluate(f"key {AWS_KEY}")
    assert outcome.skipped


def test_privacy_disabled_allows_everything() -> None:
    outcome = PrivacyPolicy(_config(privacy_enabled=False)).evaluate("password=hunter2")
    assert outcome.allowed
    assert outcome.content == "password=hunter2"


# ------------------------------------------------------------------ masking


def test_mask_text_replaces_only_the_given_spans() -> None:
    text = "a secret b secret c"
    from clipmind.privacy.detectors import Finding

    masked = mask_text(text, [Finding("jwt", 2, 8)])
    assert masked == "a [REDACTED:jwt] b secret c"


def test_mask_text_multiple_findings_right_to_left() -> None:
    from clipmind.privacy.detectors import Finding

    text = "xx yy zz"
    masked = mask_text(text, [Finding("a", 0, 2), Finding("b", 6, 8)])
    assert masked == "[REDACTED:a] yy [REDACTED:b]"


# --------------------------------------------------------------- ignore list


DEFAULT_APPS = ("1Password", "Bitwarden", "KeePassXC", "Keychain Access", "LastPass")


@pytest.mark.parametrize(
    "app, expected",
    [
        ("1Password", True),
        ("1Password 8", True),
        ("bitwarden", True),
        ("KeePassXC", True),
        ("Keychain Access", True),
        ("Safari", False),
        ("Terminal", False),
        ("", False),
        (None, False),
    ],
)
def test_matches_ignore_list(app, expected) -> None:
    assert matches_ignore_list(app, DEFAULT_APPS) is expected


def test_should_ignore_source_app_uses_config() -> None:
    assert should_ignore_source_app("1Password 8", _config()) is True
    assert should_ignore_source_app("Safari", _config()) is False


def test_custom_ignore_list() -> None:
    config = _config(ignore_source_apps=("Mail",))
    assert should_ignore_source_app("Apple Mail", config) is True
    assert should_ignore_source_app("1Password", config) is False
