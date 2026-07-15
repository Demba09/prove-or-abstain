"""webhook.py — notify external systems when the autopilot acts.

When an ASSERT triggers an EXECUTE, send a structured payload to a
configurable webhook URL (Slack, Discord, Teams, or any HTTP endpoint).

Set WEBHOOK_URL in env or .env. Without it, outputs to stdout.
Format auto-detects common platforms from the URL hostname.
"""

from __future__ import annotations
import json
import os
import sys

try:
    import requests
except ImportError:
    requests = None


def _detect_format(url: str) -> str:
    """Detect the payload format from the webhook URL hostname."""
    host = url.split("/")[2] if "://" in url else ""
    if "slack.com" in host or "hooks.slack" in host:
        return "slack"
    if "discord.com" in host or "discordapp.com" in host:
        return "discord"
    if "office.com" in host or "webhook.office" in host:
        return "teams"
    return "generic"


def _build_payload(metric: str, verdict: str, confidence: float,
                   cause: str | None, action: str, detail: str,
                   format: str = "generic") -> dict:
    """Build a platform-specific payload from the investigation result."""
    action_emoji = "✅" if action == "EXECUTE" else "⚠️" if action == "ESCALATE" else "📋"
    title = f"prove-or-abstain: {verdict}"
    cause_text = f"*Root cause:* {cause}" if cause else "*No single cause isolated*"

    if format == "slack":
        color = "#2F7A50" if verdict == "ASSERT" else "#A6472E"
        return {
            "attachments": [{
                "color": color,
                "title": f"{action_emoji} {title}",
                "fields": [
                    {"title": "Metric", "value": metric, "short": True},
                    {"title": "Confidence", "value": f"{confidence:.2f}", "short": True},
                    {"title": "Cause", "value": cause_text, "short": False},
                    {"title": "Action", "value": f"*{action}* — {detail}", "short": False},
                ],
                "footer": "prove-or-abstain · Track 4 Autopilot Agent",
            }]
        }

    if format == "discord":
        color = 0x2F7A50 if verdict == "ASSERT" else 0xA6472E
        return {
            "embeds": [{
                "title": f"{action_emoji} {title}",
                "color": color,
                "fields": [
                    {"name": "Metric", "value": metric, "inline": True},
                    {"name": "Confidence", "value": f"{confidence:.2f}", "inline": True},
                    {"name": "Cause", "value": cause_text, "inline": False},
                    {"name": "Action", "value": f"**{action}** — {detail}", "inline": False},
                ],
                "footer": {"text": "prove-or-abstain · Track 4 Autopilot Agent"},
            }]
        }

    # generic / teams: simple key-value
    return {
        "title": f"{action_emoji} {title}",
        "metric": metric,
        "verdict": verdict,
        "confidence": confidence,
        "cause": cause,
        "action": action,
        "detail": detail,
    }


def notify(metric: str, verdict: str, confidence: float,
           cause: str | None, action: str, detail: str) -> bool:
    """Send a notification about an autopilot action.

    Returns True if a webhook was sent, False if it was logged to stdout.
    Raises only on unrecoverable errors (never on network timeout).
    """
    url = os.environ.get("WEBHOOK_URL", "").strip()

    if not url:
        payload = _build_payload(metric, verdict, confidence, cause, action, detail)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return False

    fmt = _detect_format(url)
    payload = _build_payload(metric, verdict, confidence, cause, action, detail, fmt)

    if requests is None:
        print(f"[webhook] requests not installed — would send to {url}", file=sys.stderr)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return False

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code >= 400:
            print(f"[webhook] {url} returned {resp.status_code}: {resp.text[:200]}",
                  file=sys.stderr)
            return False
        return True
    except requests.RequestException as exc:
        print(f"[webhook] could not reach {url}: {exc}", file=sys.stderr)
        return False
