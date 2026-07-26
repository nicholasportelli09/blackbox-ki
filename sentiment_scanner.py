"""
AI MARKET SENTIMENT SCANNER — continuously scans online sources (news via Google
News RSS, social via Reddit public JSON, the economic calendar we already pull) and
distils them into per-asset sentiment scores with plain-English explanations.

NLP: finance-tuned lexicon scoring (VADER-style: polarity words + intensifiers,
title-level) — no API keys required. Every score is traceable to the headlines
that produced it (shown in the app; no black box).

Output: logs/sentiment.json   (latest snapshot, served by the app)
        logs/sentiment_history.csv (score history -> trend arrows)
Bots read the snapshot and SHADOW-LOG the advisory next to their own decisions —
sentiment informs and explains, it does not yet override validated rules.
"""
import os, json, re, csv, time, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone

import ai_client

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "logs", "sentiment.json")
HIST = os.path.join(HERE, "logs", "sentiment_history.csv")
UA = {"User-Agent": "Mozilla/5.0 (blackbox-research)"}

ASSETS = ["equities", "gold", "rates", "fx", "crypto", "oil"]

NEWS_QUERIES = {
    "equities": "stock market",
    "gold": "gold price",
    "rates": "federal reserve interest rates",
    "fx": "us dollar currency",
    "crypto": "bitcoin crypto market",
    "oil": "oil prices",
}
REDDIT_SUBS = {
    "wallstreetbets": "equities", "stocks": "equities", "investing": "equities",
    "CryptoCurrency": "crypto", "Bitcoin": "crypto",
}
# keyword re-tagging (a WSB post about gold lands in the gold bucket)
TAGS = {
    "gold": ["gold", "bullion", "xau", "precious metal"],
    "crypto": ["bitcoin", "btc", "ethereum", "crypto", "solana"],
    "rates": ["fed ", "federal reserve", "interest rate", "treasury", "yield", "fomc", "powell", "ecb", "central bank"],
    "oil": ["oil", "crude", "opec", "brent", "wti"],
    "fx": ["dollar", "euro", "yen", "forex", "currency"],
    "equities": ["stock", "s&p", "nasdaq", "dow", "equit", "shares"],
}

POS = set("""surge surges surging rally rallies rallying soar soars soaring jump jumps gains gain boom booms
record-high beat beats strong strength bullish optimism optimistic upgrade upgraded growth recover recovers
recovery rebound rebounds breakout outperform outperforms dovish stimulus soft-landing cools cooling
all-time-high high tops top rise rises rising climb climbs advance advances upbeat resilient hope hopes""".split())
NEG = set("""crash crashes plunge plunges plunging tumble tumbles slump slumps fall falls falling drop drops
fear fears bearish recession downgrade downgraded miss misses weak weakness selloff sell-off panic default
crisis war sanctions hawkish hike hikes layoffs bankruptcy bubble correction liquidation slide slides sink
sinks slips slip warning warns worry worries risk risks losses lose loses turmoil rout jitters dive dives""".split())
INTENS = {"very": 1.5, "sharply": 1.5, "massive": 1.5, "record": 1.4, "historic": 1.4, "huge": 1.4}


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def score_text(text):
    """Lexicon polarity score in [-1, 1] for one headline."""
    words = re.findall(r"[a-z&\-']+", text.lower())
    pos = neg = 0.0
    mult = 1.0
    for w in words:
        if w in INTENS:
            mult = INTENS[w]; continue
        if w in POS: pos += mult; mult = 1.0
        elif w in NEG: neg += mult; mult = 1.0
        else: mult = 1.0
    tot = pos + neg
    return (pos - neg) / tot if tot else 0.0


def tag_asset(title, default):
    t = title.lower()
    for asset, kws in TAGS.items():
        if any(k in t for k in kws):
            return asset
    return default


def fetch_news():
    items = []
    for asset, q in NEWS_QUERIES.items():
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
            root = ET.fromstring(_get(url))
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                src = (it.findtext("source") or "news").strip()
                if title:
                    items.append({"title": title[:180], "src": src[:40],
                                  "asset": tag_asset(title, asset), "via": "news"})
        except Exception:
            continue
    return items


def fetch_reddit():
    items = []
    for sub, default in REDDIT_SUBS.items():
        try:
            d = json.loads(_get(f"https://www.reddit.com/r/{sub}/hot.json?limit=35"))
            for post in d.get("data", {}).get("children", []):
                p = post.get("data", {})
                title = (p.get("title") or "").strip()
                if title and not p.get("stickied"):
                    items.append({"title": title[:180], "src": f"r/{sub}",
                                  "asset": tag_asset(title, default), "via": "social"})
        except Exception:
            continue
        time.sleep(0.4)
    return items


def explain(asset, scored, score):
    n = len(scored)
    if n < 3:
        return f"Too little coverage of {asset} right now to read sentiment reliably."
    npos = sum(1 for s in scored if s["score"] > 0.15)
    nneg = sum(1 for s in scored if s["score"] < -0.15)
    words = {}
    for s in scored:
        for w in re.findall(r"[a-z\-']+", s["title"].lower()):
            if w in POS or w in NEG:
                words[w] = words.get(w, 0) + 1
    drivers = ", ".join(f"'{w}'" for w, _ in sorted(words.items(), key=lambda z: -z[1])[:4])
    mood = ("strongly bullish" if score > 40 else "leaning bullish" if score > 12 else
            "strongly bearish" if score < -40 else "leaning bearish" if score < -12 else "mixed/neutral")
    return (f"{mood.capitalize()}: of {n} scanned items, {npos} read positive and {nneg} negative"
            + (f", driven by {drivers}" if drivers else "") + ".")


AI_SAMPLE = 25  # headlines per bucket sent to the AI — keeps calls small and fast


def ai_score_bucket(asset, items):
    """AI reads up to AI_SAMPLE headlines for one asset bucket and scores each
    in [-1, 1], with a plain-English explanation. Returns (scores, explanation)
    for the sampled items, or None on any failure — caller falls back to the
    keyword lexicon for that bucket, so a bad AI response never breaks a scan.

    Scores are keyed by headline number rather than returned as a bare ordered
    list — models occasionally miscount on ~25-item lists, and a keyed dict lets
    a missing entry default to neutral instead of invalidating the whole batch."""
    sample = items[:AI_SAMPLE]
    if not sample:
        return None
    heads = "\n".join(f"{i}. {it['title']}" for i, it in enumerate(sample))
    prompt = (
        f"Score financial-news sentiment for the '{asset}' asset class. For EACH "
        "numbered headline below, give a sentiment score from -1 (very bearish for "
        "this asset class) to +1 (very bullish), 0 if neutral/irrelevant. Read for "
        "real meaning, not just keywords — handle negation and sarcasm correctly.\n\n"
        f"HEADLINES:\n{heads}\n\n"
        'Reply with ONLY this JSON, nothing else: {"scores": {"<headline number as '
        'a string>": <score>, ... one entry per headline}, "explanation": "<one '
        'plain-English sentence on the mood and what is driving it>"}'
    )
    result = ai_client.ask_json(prompt, max_tokens=900)
    if not result or not isinstance(result.get("scores"), dict):
        return None
    raw = result["scores"]
    scores = []
    for i in range(len(sample)):
        try:
            scores.append(max(-1.0, min(1.0, float(raw.get(str(i), 0.0)))))
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores, str(result.get("explanation", ""))[:300]


def run_scan():
    items = fetch_news() + fetch_reddit()
    now = datetime.now(timezone.utc).isoformat()
    use_ai = ai_client.available()
    any_ai = False
    assets = {}
    for asset in ASSETS:
        sub = [i for i in items if i["asset"] == asset]
        ai_out = ai_score_bucket(asset, sub) if use_ai else None
        if ai_out:
            any_ai = True
            ai_scores, ai_expl = ai_out
            scored = [{**it, "score": round(s, 3)} for it, s in zip(sub, ai_scores)]
            scored += [{**it, "score": round(score_text(it["title"]), 3)} for it in sub[len(ai_scores):]]
        else:
            ai_expl = None
            scored = [{**i, "score": round(score_text(i["title"]), 3)} for i in sub]
        nonzero = [s for s in scored if s["score"] != 0]
        score = round(100 * (sum(s["score"] for s in nonzero) / len(nonzero)), 1) if nonzero else 0.0
        top = sorted(scored, key=lambda s: -abs(s["score"]))[:5]
        assets[asset] = {"score": score, "n_items": len(sub),
                         "n_social": sum(1 for s in sub if s["via"] == "social"),
                         "explanation": ai_expl or explain(asset, scored, score),
                         "method": "ai" if ai_out else "lexicon",
                         "top": [{"title": t["title"], "src": t["src"], "score": t["score"],
                                  "via": t["via"]} for t in top]}
    # trend vs previous snapshots
    hist_rows = []
    if os.path.exists(HIST):
        try: hist_rows = list(csv.DictReader(open(HIST, encoding="utf-8")))
        except Exception: pass
    for a in ASSETS:
        prev = [float(r[a]) for r in hist_rows[-12:] if r.get(a)]
        cur = assets[a]["score"]
        assets[a]["trend"] = ("improving" if prev and cur > np_mean(prev) + 5 else
                              "deteriorating" if prev and cur < np_mean(prev) - 5 else "stable")
    method_desc = ("Headlines from Google News RSS + Reddit public feeds, " +
        ("read and scored by an AI model per headline (falls back to a keyword "
         "lexicon for any bucket where that fails). "
         if any_ai else
         "scored with a finance-tuned NLP lexicon (polarity + intensifiers). ") +
        "Every score is traceable to its headlines. Advisory only — validated rules trade.")
    doc = {"generated": now, "n_scanned": len(items), "assets": assets, "method": method_desc}
    tmp = OUT + ".tmp"
    json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, OUT)
    new = not os.path.exists(HIST)
    with open(HIST, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["ts"] + ASSETS)
        w.writerow([now] + [assets[a]["score"] for a in ASSETS])
    return doc


def np_mean(x):
    return sum(x) / len(x) if x else 0.0


def read_latest(max_age_min=90):
    """For the bots: latest snapshot if fresh, else None. Never raises."""
    try:
        d = json.load(open(OUT, encoding="utf-8"))
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(d["generated"])).total_seconds() / 60
        return d if age <= max_age_min else None
    except Exception:
        return None


if __name__ == "__main__":
    d = run_scan()
    print(f"scanned {d['n_scanned']} items")
    for a, v in d["assets"].items():
        print(f"  {a:<9} {v['score']:>6.1f}  ({v['n_items']} items, {v['trend']})  {v['explanation'][:90]}")
