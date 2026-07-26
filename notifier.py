"""
Phone push notifications — trade OPEN and CLOSE only.

Primary: ntfy.sh — zero-account push. The bot POSTs to a secret topic; the
ntfy app on the phone subscribes to that topic and gets instant notifications.
Optional: Telegram, if TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are ever set.

Topic name is secret-ish (treat like a password) — set via NTFY_TOPIC in .env.
"""
import os
import requests

import config  # noqa: F401  (ensures .env is loaded)
from logger import log

NTFY_TOPIC = os.getenv("NTFY_TOPIC")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")


def send(msg: str, title: str = "Blackbox", priority: str = "default") -> bool:
    ok = False
    if NTFY_TOPIC:
        try:
            safe_title = title.encode("ascii", "replace").decode("ascii")  # headers must be ASCII
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=msg.encode("utf-8"),
                headers={
                    "Title": safe_title,
                    "Priority": priority,
                    "Tags": "chart_with_upwards_trend",
                    # Blackbox brain logo (shown on Android/web; iOS always uses the app icon)
                    "Icon": "https://blackbox.54-246-124-234.sslip.io/icon.png",
                },
                timeout=10,
            )
            ok = True
        except Exception as e:
            log.warning(f"ntfy send failed: {e}")
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": f"{title}\n{msg}"},
                timeout=10,
            )
            ok = True
        except Exception as e:
            log.warning(f"telegram send failed: {e}")
    return ok
