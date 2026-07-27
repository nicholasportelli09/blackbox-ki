"""
Shared connector to Groq's chat-completion API (OpenAI-compatible), used by the
historical-analog engine, the AI sentiment scanner and the portfolio narrator.

Fails open: ask() returns None on any failure (missing key, network error,
malformed response) so callers degrade gracefully instead of crashing — same
pattern as sentiment_scanner.read_latest() and the OANDA client.
"""
import json
import urllib.request

import config

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
# Groq's edge (Cloudflare) blocks the default urllib User-Agent as bot traffic.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def available():
    return bool(config.GROQ_API_KEY)


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
