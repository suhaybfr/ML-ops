"""Live Discord & generic webhook alert notifier for drift, retrain, and error paging.

Gracefully no-ops when neither ``ALERT_WEBHOOK_URL`` nor ``DISCORD_WEBHOOK_URL``
is configured so offline CI and local dev never make outbound network calls.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

_SEVERITY_COLORS = {
    "critical": 0xDC2626,  # red
    "warning": 0xD97706,  # amber
    "info": 0x15803D,  # green
}


def get_alert_webhook_url() -> str:
    return (
        os.environ.get("ALERT_WEBHOOK_URL", "").strip()
        or os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    )


def _is_discord_webhook(url: str) -> bool:
    lower = url.lower()
    return "discord.com/api/webhooks/" in lower or "discordapp.com/api/webhooks/" in lower


def build_alert_payload(
    title: str,
    message: str,
    severity: str = "warning",
    fields: Optional[Dict[str, Any]] = None,
    webhook_url: str = "",
) -> Dict[str, Any]:
    """Build either a Discord Embed payload or a generic JSON webhook payload."""
    sev = severity.strip().lower() if severity else "warning"
    now_iso = datetime.now(timezone.utc).isoformat()
    if _is_discord_webhook(webhook_url):
        embed_fields = [
            {"name": str(k), "value": str(v), "inline": True}
            for k, v in (fields or {}).items()
        ]
        return {
            "username": "MLOps Monitor",
            "embeds": [
                {
                    "title": f"[{sev.upper()}] {title}",
                    "description": message,
                    "color": _SEVERITY_COLORS.get(sev, _SEVERITY_COLORS["warning"]),
                    "fields": embed_fields,
                    "timestamp": now_iso,
                }
            ],
        }

    return {
        "service": "student-performance-predictor",
        "severity": sev,
        "title": title,
        "message": message,
        "fields": {str(k): v for k, v in (fields or {}).items()},
        "timestamp": now_iso,
    }


def send_alert_webhook(
    title: str,
    message: str,
    severity: str = "warning",
    fields: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Send a drift/retrain/error alert if a webhook URL is configured (non-fatal)."""
    url = (webhook_url if webhook_url is not None else get_alert_webhook_url()).strip()
    if not url:
        return {"sent": False, "reason": "no_webhook_configured"}

    payload = build_alert_payload(
        title=title,
        message=message,
        severity=severity,
        fields=fields,
        webhook_url=url,
    )
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return {"sent": True, "status_code": resp.status_code, "discord": _is_discord_webhook(url)}
    except Exception as exc:
        return {"sent": False, "error": f"{type(exc).__name__}: {exc}"}
