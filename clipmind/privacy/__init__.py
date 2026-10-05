"""Privacy filters (Phase 4).

Will detect and mask or drop secrets (API keys, JWTs, credit-card numbers,
PEM private keys), enforce a source-application ignore list, and apply the
auto-delete-after-N-days retention policy. Everything here runs locally.
"""

from __future__ import annotations

__all__: list[str] = []
