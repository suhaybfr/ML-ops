"""API security dependencies: optional Admin API Key auth & sliding-window rate limiter.

Design:
  - ``require_admin_api_key``: when ``ADMIN_API_KEY`` env var is unset or empty,
    mutating/ops endpoints stay open for zero-friction local dev and offline CI.
    When ``ADMIN_API_KEY`` is set, callers must send ``X-API-Key: <key>`` or
    ``Authorization: Bearer <key>`` (verified with ``hmac.compare_digest``).
  - ``SlidingWindowRateLimiter``: thread-safe in-memory per-client-IP rate limiter
    protecting LLM/upload endpoints (``/tutor/ask``, ``/tutor/documents``).
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException, Request


def get_configured_admin_key() -> str:
    """Return the configured ``ADMIN_API_KEY`` (empty string when auth is disabled)."""
    return os.environ.get("ADMIN_API_KEY", "").strip()


def require_admin_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    """Enforce ``ADMIN_API_KEY`` on mutating/admin endpoints whenever the env var is set."""
    expected = get_configured_admin_key()
    if not expected:
        return

    provided = (x_api_key or "").strip()
    if not provided and authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()

    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Admin API key. Supply X-API-Key or Authorization: Bearer.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SlidingWindowRateLimiter:
    """Thread-safe per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = {}

    def check(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(client_key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])) + 1)
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded ({self.max_requests} requests per "
                        f"{self.window_seconds}s). Retry after {retry_after}s."
                    ),
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def _env_int(name: str, default: int) -> int:
    try:
        val = int(os.environ.get(name, str(default)).strip())
        return max(1, val)
    except (ValueError, TypeError):
        return default


_tutor_ask_limiter: Optional[SlidingWindowRateLimiter] = None
_tutor_ask_limit_cfg: int = -1
_tutor_upload_limiter: Optional[SlidingWindowRateLimiter] = None
_tutor_upload_limit_cfg: int = -1
_limiter_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def enforce_tutor_ask_rate_limit(request: Request) -> None:
    """Rate-limit ``POST /tutor/ask`` (default 30 req / 60s per IP)."""
    global _tutor_ask_limiter, _tutor_ask_limit_cfg
    limit = _env_int("TUTOR_ASK_RATE_LIMIT", 30)
    with _limiter_lock:
        if _tutor_ask_limiter is None or _tutor_ask_limit_cfg != limit:
            _tutor_ask_limiter = SlidingWindowRateLimiter(max_requests=limit, window_seconds=60)
            _tutor_ask_limit_cfg = limit
        limiter = _tutor_ask_limiter
    limiter.check(_client_ip(request))


def enforce_tutor_upload_rate_limit(request: Request) -> None:
    """Rate-limit ``POST /tutor/documents`` (default 10 req / 60s per IP)."""
    global _tutor_upload_limiter, _tutor_upload_limit_cfg
    limit = _env_int("TUTOR_UPLOAD_RATE_LIMIT", 10)
    with _limiter_lock:
        if _tutor_upload_limiter is None or _tutor_upload_limit_cfg != limit:
            _tutor_upload_limiter = SlidingWindowRateLimiter(max_requests=limit, window_seconds=60)
            _tutor_upload_limit_cfg = limit
        limiter = _tutor_upload_limiter
    limiter.check(_client_ip(request))


def reset_rate_limiters() -> None:
    """Reset in-memory rate-limit counters (used by tests)."""
    with _limiter_lock:
        if _tutor_ask_limiter is not None:
            _tutor_ask_limiter.reset()
        if _tutor_upload_limiter is not None:
            _tutor_upload_limiter.reset()
