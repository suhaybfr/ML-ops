"""Live Discord, generic webhook, and SMTP email alert notifier for drift & retrain paging.

Gracefully no-ops when neither webhook nor SMTP email env vars are configured so
offline CI and local dev never make outbound network calls.
"""

from __future__ import annotations

import html as html_lib
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
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


def get_email_alert_config() -> Optional[Dict[str, Any]]:
    """Return SMTP email alert config if all required env vars are set, else None."""
    to_addr = os.environ.get("ALERT_EMAIL_TO", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    if not (to_addr and user and password):
        return None
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip() or "smtp.gmail.com"
    try:
        port = int(os.environ.get("SMTP_PORT", "465").strip() or "465")
    except ValueError:
        port = 465
    from_addr = os.environ.get("ALERT_EMAIL_FROM", "").strip() or user
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "from_addr": from_addr,
        "to_addr": to_addr,
    }


def is_alerts_configured() -> bool:
    return bool(get_alert_webhook_url() or get_email_alert_config())


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


def send_alert_email(
    title: str,
    message: str,
    severity: str = "warning",
    fields: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Send a formatted alert email via SMTP (Gmail App Password or any SMTP server)."""
    cfg = get_email_alert_config()
    if cfg is None:
        return {"sent": False, "reason": "no_email_configured"}

    sev = (severity or "warning").strip().upper()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [f"[{sev}] {title}", "", message, ""]
    for k, v in (fields or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", f"Timestamp: {now_iso}"])
    plain_body = "\n".join(lines)

    rows_html = "".join(
        f"<tr><td style='padding:4px 10px 4px 0;font-weight:600;color:#18181b;'>"
        f"{html_lib.escape(str(k))}</td>"
        f"<td style='padding:4px 0;color:#27272a;'>{html_lib.escape(str(v))}</td></tr>"
        for k, v in (fields or {}).items()
    )
    html_body = (
        "<div style='font-family:system-ui,-apple-system,sans-serif;max-width:560px;"
        "border:1px solid #e4e4e7;border-radius:8px;padding:16px;'>"
        f"<h2 style='margin:0 0 8px;font-size:16px;color:#09090b;'>"
        f"[{html_lib.escape(sev)}] {html_lib.escape(title)}</h2>"
        f"<p style='margin:0 0 12px;color:#27272a;'>{html_lib.escape(message)}</p>"
        f"<table style='border-collapse:collapse;font-size:13px;'>{rows_html}</table>"
        f"<p style='margin:14px 0 0;font-size:11px;color:#71717a;'>"
        f"Student Performance MLOps Monitor · {html_lib.escape(now_iso)}</p></div>"
    )

    msg = EmailMessage()
    msg["Subject"] = f"[{sev}] {title}"
    msg["From"] = cfg["from_addr"]
    msg["To"] = cfg["to_addr"]
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=timeout) as smtp:
                smtp.login(cfg["user"], cfg["password"])
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=timeout) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(cfg["user"], cfg["password"])
                smtp.send_message(msg)
        return {"sent": True, "to": cfg["to_addr"], "host": cfg["host"]}
    except Exception as exc:
        return {"sent": False, "error": f"{type(exc).__name__}: {exc}"}


def send_alert_webhook(
    title: str,
    message: str,
    severity: str = "warning",
    fields: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Send a drift/retrain/error alert via webhook (Discord) and/or SMTP email (non-fatal)."""
    url = (webhook_url if webhook_url is not None else get_alert_webhook_url()).strip()
    email_res = send_alert_email(
        title=title,
        message=message,
        severity=severity,
        fields=fields,
        timeout=timeout,
    )

    if not url:
        if email_res.get("sent"):
            return {"sent": True, "discord": False, "email_sent": True, "email": email_res}
        return {
            "sent": False,
            "reason": "no_webhook_configured",
            "email_sent": False,
            "email": email_res,
        }

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
        return {
            "sent": True,
            "status_code": resp.status_code,
            "discord": _is_discord_webhook(url),
            "email_sent": bool(email_res.get("sent")),
            "email": email_res,
        }
    except Exception as exc:
        return {
            "sent": bool(email_res.get("sent")),
            "error": f"{type(exc).__name__}: {exc}",
            "email_sent": bool(email_res.get("sent")),
            "email": email_res,
        }
