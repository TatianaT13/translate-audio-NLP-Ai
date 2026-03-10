import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from flash_nlp.analysis.event_extractor import TrafficEvent

_SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def notify_console(event: TrafficEvent) -> None:
    icon = _SEVERITY_ICON.get(event.severity, "⚪")
    routes = ", ".join(event.routes) if event.routes else "—"
    direction = f" {event.direction}" if event.direction else ""
    delay = f" | {event.delay_hint}" if event.delay_hint else ""
    ts = event.timestamp or "?"
    print(
        f"{icon} [{ts}] {event.zone.upper():5s} | {event.type.upper():18s} | "
        f"{routes}{direction}{delay}\n"
        f"   {event.location_hint}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# macOS notification (osascript)
# ---------------------------------------------------------------------------

def notify_macos(event: TrafficEvent) -> None:
    icon = _SEVERITY_ICON.get(event.severity, "")
    title = f"{icon} Flash Trafic — {event.zone.upper()}"
    routes = " ".join(event.routes) if event.routes else ""
    direction = event.direction or ""
    body = f"{event.type.upper()} {routes} {direction}".strip()
    # Échappe les guillemets pour AppleScript
    title_safe = title.replace('"', '\\"')
    body_safe = body.replace('"', '\\"')
    script = f'display notification "{body_safe}" with title "{title_safe}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # osascript non disponible (Linux / CI)


# ---------------------------------------------------------------------------
# Webhook HTTP
# ---------------------------------------------------------------------------

def notify_webhook(event: TrafficEvent, url: str) -> None:
    if not _HAS_REQUESTS:
        print("[notifier] requests non disponible, webhook ignoré.", file=sys.stderr)
        return
    payload = event.as_dict()
    payload["alerted_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[notifier] webhook {url} → erreur : {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Log fichier JSONL
# ---------------------------------------------------------------------------

def log_to_file(event: TrafficEvent, alerts_dir: Path) -> None:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    line = event.as_dict()
    line["alerted_at"] = datetime.now(timezone.utc).isoformat()
    with (alerts_dir / "alerts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dispatch central
# ---------------------------------------------------------------------------

def dispatch(
    event: TrafficEvent,
    alerts_dir: Path,
    macos: bool = False,
    webhook_url: Optional[str] = None,
) -> None:
    notify_console(event)
    log_to_file(event, alerts_dir)
    if macos:
        notify_macos(event)
    if webhook_url:
        notify_webhook(event, webhook_url)
