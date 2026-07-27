"""
Shared connector to Groq's chat-completion API (OpenAI-compatible), used by the
historical-analog engine, the AI sentiment scanner and the portfolio narrator.

Fails open: ask() returns None on any failure (missing key, network error,
malformed response) so callers degrade gracefully instead of crashing — same
pattern as sentiment_scanner.read_latest() and the OANDA client.
"""
import json
import os
import secrets
import threading
import time
import urllib.request

import config

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
# Groq's edge (Cloudflare) blocks the default urllib User-Agent as bot traffic.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------------------------------------------------------------------------
# Per-user (and global) rate limiting — conserves the shared free Groq quota
# so one visitor or a bot can't drain it. State is in-memory, which is correct
# here because the app runs a single gunicorn worker. Only REAL Groq calls are
# counted (cache hits never reach ask()), and the background scanner — which has
# no web-request context — is never throttled. All limits are env-overridable.
# ---------------------------------------------------------------------------
PER_USER_MAX = int(os.getenv("AI_PER_USER_MAX", "15"))        # calls per user…
PER_USER_WINDOW = int(os.getenv("AI_PER_USER_WINDOW", "1800"))  # …per 30 minutes
GLOBAL_MAX = int(os.getenv("AI_GLOBAL_MAX", "250"))           # calls total…
GLOBAL_WINDOW = int(os.getenv("AI_GLOBAL_WINDOW", "3600"))     # …per hour (backstop)
_RL_LOCK = threading.Lock()
_RL = {}        # session id -> [timestamps]
_GLOBAL = []    # [timestamps] across all users


def available():
    return bool(config.GROQ_API_KEY)


def _session_id():
    """Stable per-browser id for rate limiting; None outside a web request
    (e.g. the scanner), so background work is never counted or blocked."""
    try:
        from flask import session, has_request_context
        if not has_request_context():
            return None
        sid = session.get("aid")
        if not sid:
            sid = secrets.token_hex(8)
            session["aid"] = sid
            session.permanent = True
        return sid
    except Exception:
        return None


def _rate_ok():
    """True if this caller may make one more Groq call now. Records the call
    when allowed. Non-web callers (scanner) are always allowed."""
    sid = _session_id()
    if sid is None:
        return True
    now = time.time()
    with _RL_LOCK:
        g = [t for t in _GLOBAL if now - t < GLOBAL_WINDOW]
        if len(g) >= GLOBAL_MAX:
            _GLOBAL[:] = g
            return False
        q = [t for t in _RL.get(sid, []) if now - t < PER_USER_WINDOW]
        if len(q) >= PER_USER_MAX:
            _RL[sid] = q
            return False
        q.append(now)
        _RL[sid] = q
        g.append(now)
        _GLOBAL[:] = g
        if len(_RL) > 5000:  # opportunistic cleanup of idle sessions
            for k in [k for k, v in _RL.items() if not v or now - v[-1] > PER_USER_WINDOW]:
                _RL.pop(k, None)
        return True


def _lang_pref():
    """'de' when the current web request asks for German (X-Lang header), else None.
    Guarded so the background scanner (no request context) is unaffected."""
    try:
        from flask import request, has_request_context
        if has_request_context() and request.headers.get("X-Lang", "").lower().startswith("de"):
            return "de"
    except Exception:
        pass
    return None


def ask(prompt, system=None, max_tokens=600, timeout=20):
    """One-shot chat completion. Returns the reply text, or None on any failure."""
    if not config.GROQ_API_KEY:
        return None
    if not _rate_ok():
        # Over the per-user (or global) budget — fail open like any other outage,
        # but tag the request so callers can show a rate-limit message.
        try:
            from flask import g, has_request_context
            if has_request_context():
                g._ai_rate_limited = True
        except Exception:
            pass
        return None
    if _lang_pref() == "de":
        # Make human-readable output German; keep any JSON keys/structure intact
        # (ask_json calls through here, so this must not break structured replies).
        de = ("Write all human-readable text and prose in natural, fluent German. "
              "Keep any JSON structure, keys and numbers exactly as instructed.")
        system = (de + " " + system) if system else de
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "User-Agent": UA,
        "Accept": "application/json",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def ask_json(prompt, system=None, max_tokens=600, timeout=20):
    """Like ask(), but parses the reply as JSON. Returns None if the call or
    the parse fails, so callers never need to handle malformed AI output."""
    text = ask(prompt, system=system, max_tokens=max_tokens, timeout=timeout)
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return None
