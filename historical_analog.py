"""
Historical-analog engine — matches today's real news to the closest match from
a fixed library of real past market events, then shows how the real, already-
backtested strategies in catalog.json actually performed in the months after.

AI's only job is CLASSIFICATION against data/historical_events.json — it picks
an id from that fixed list, it never invents an event, a date or a number.
Every figure shown comes from the real strategy curves already validated for
the rest of the app. Fails open throughout: any missing piece (no AI key, no
sentiment snapshot, no match) returns None rather than raising.
"""
import os, json
from datetime import datetime, timedelta, timezone

import ai_client

HERE = os.path.dirname(__file__)
EVENTS_PATH = os.path.join(HERE, "data", "historical_events.json")
CATALOG_PATH = os.path.join(HERE, "logs", "catalog.json")
SENTIMENT_PATH = os.path.join(HERE, "logs", "sentiment.json")
OUT = os.path.join(HERE, "logs", "historical_analog.json")

HORIZON_DAYS = 90


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _events():
    return _load(EVENTS_PATH, [])


def _nearest(curve, target_date):
    """Nearest curve point to target_date (curve = [{'t': 'YYYY-MM-DD', 'v': float}, ...])."""
    td = datetime.fromisoformat(target_date)
    return min(curve, key=lambda p: abs((datetime.fromisoformat(p["t"]) - td).days))


def _strategy_reactions(event_date, horizon_days=HORIZON_DAYS):
    """Real % change in each strategy's equity curve from event_date to +horizon_days."""
    cat = _load(CATALOG_PATH, {"strategies": []})
    end_date = (datetime.fromisoformat(event_date) + timedelta(days=horizon_days)).date().isoformat()
    out = {}
    for s in cat.get("strategies", []):
        curve = s.get("curve", [])
        if not curve:
            continue
        try:
            p0, p1 = _nearest(curve, event_date), _nearest(curve, end_date)
            if p0["v"]:
                out[s["id"]] = {"name": s["name"], "pct": round((p1["v"] / p0["v"] - 1) * 100, 1)}
        except Exception:
            continue
    return out


def _pick_analog(headlines):
    """Ask AI to choose the closest-matching event from the fixed curated list
    only — constrained to real ids, so it classifies rather than invents."""
    events = _events()
    if not events or not headlines:
        return None
    menu = "\n".join(f"- {e['id']}: {e['title']} ({e['date']}) — {e['summary']}" for e in events)
    heads = "\n".join(f"- {h}" for h in headlines[:15])
    prompt = (
        "Match today's financial news to the ONE closest historical analog from this "
        "fixed list of real past market events. Only pick an id from the list — never "
        "invent a new event.\n\n"
        f"EVENTS:\n{menu}\n\n"
        f"TODAY'S HEADLINES:\n{heads}\n\n"
        'Reply with ONLY this JSON, nothing else: '
        '{"event_id": "<id from the list>", "reason": "<one sentence on why '
        'today\'s headlines resemble it>"}'
    )
    result = ai_client.ask_json(prompt, max_tokens=300)
    if not result or result.get("event_id") not in {e["id"] for e in events}:
        return None
    return result


def analyze():
    """Full pipeline: real headlines -> AI picks a real analog -> real price
    lookup. Returns None (fails open) if AI or sentiment data is unavailable."""
    sentiment = _load(SENTIMENT_PATH, {})
    headlines = [t["title"] for a in sentiment.get("assets", {}).values() for t in a.get("top", [])]
    picked = _pick_analog(headlines)
    if not picked:
        return None
    event = {e["id"]: e for e in _events()}[picked["event_id"]]
    doc = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "reason": picked.get("reason", ""),
        "horizon_days": HORIZON_DAYS,
        "reactions": _strategy_reactions(event["date"]),
    }
    tmp = OUT + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, OUT)
    return doc


def read_latest(max_age_min=180):
    """For the app: cached snapshot if fresh enough, else None. Never raises."""
    try:
        d = json.load(open(OUT, encoding="utf-8"))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(d["generated"])).total_seconds() / 60
        return d if age <= max_age_min else None
    except Exception:
        return None
