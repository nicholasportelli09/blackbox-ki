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


def ask(prompt, system=None, max_tokens=600, timeout=20):
    """One-shot chat completion. Returns the reply text, or None on any failure."""
    if not config.GROQ_API_KEY:
        return None
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
