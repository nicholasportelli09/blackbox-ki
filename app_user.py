"""
BLACKBOX — user-facing strategy platform (ETH KI challenge), v2.

The journey: hero -> "Let's go" -> pick your path (6-question quiz where EVERY
answer visibly constrains the result, or build your own mix from raw stats) ->
"Start my Blackbox" -> a personal paper portfolio marked to LIVE markets by the
same validated rules, with the AI scanner utilised per-strategy (exposure tilt +
explanations). No auth (demo): the browser keeps a local id; the server keeps the
book. Design: premium-site-workflow + emil-design-eng.
"""
import os, json, time, secrets, hashlib
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request, render_template_string, Response, session, g

HERE = os.path.dirname(__file__)
CATALOG = os.path.join(HERE, "logs", "catalog.json")
SENTIMENT = os.path.join(HERE, "logs", "sentiment.json")
USERS = os.path.join(HERE, "logs", "app_users.json")
ACCOUNTS = os.path.join(HERE, "logs", "app_accounts.json")

app = Flask(__name__)


def _app_secret():
    """Stable signing secret across restarts (env wins; else persisted once)."""
    s = os.getenv("APP_SECRET")
    if s:
        return s
    p = os.path.join(HERE, "logs", "app_secret.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    s = secrets.token_hex(32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(s)
    return s


app.secret_key = _app_secret()
app.permanent_session_lifetime = timedelta(days=60)
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)
_cache = {}


def cached(key, ttl, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    v = fn()
    _cache[key] = (now, v)
    return v


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _save_users(u):
    tmp = USERS + ".tmp"
    json.dump(u, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, USERS)


VALID_IDS = {"stack_bal", "stack_agg", "equity_season", "gold_macro",
             "commod_trend", "risk_parity", "crypto_trend", "spx_hold"}


def _notify(msg, title="Blackbox Platform"):
    """Owner push on product events (signups, new portfolios). Never raises."""
    try:
        import notifier
        notifier.send(msg, title=title)
    except Exception:
        pass

TIER_LABELS = ["Capital Preservation", "Balanced Growth", "Growth", "Maximum Growth"]


@app.route("/api/catalog")
def api_catalog():
    return jsonify(_load(CATALOG, {"strategies": []}))


@app.after_request
def _tag_ai_limit(resp):
    """When a Groq call was blocked by the per-user limit, turn the generic
    'available:false' AI reply into a clear, localized 'limit reached' message.
    The single-page client already prefers a server-provided text/answer."""
    try:
        if getattr(g, "_ai_rate_limited", False) and resp.is_json:
            data = resp.get_json(silent=True)
            if isinstance(data, dict) and data.get("available") is False:
                de = request.headers.get("X-Lang", "").lower().startswith("de")
                msg = ("Du hast das KI-Limit vorerst erreicht — bitte versuche es in "
                       "ein paar Minuten erneut." if de else
                       "You've reached the AI limit for now — please try again in a few minutes.")
                data["limited"] = True
                data.setdefault("text", msg)
                data.setdefault("answer", msg)
                resp.set_data(json.dumps(data))
    except Exception:
        pass
    return resp


@app.route("/api/sentiment")
def api_sentiment():
    return jsonify(_load(SENTIMENT, {"assets": {}}))


@app.route("/api/marks")
def api_marks():
    try:
        from user_engine import marks
        return jsonify(marks())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical-analog")
def api_historical_analog():
    try:
        import historical_analog as ha
        doc = ha.read_latest() or ha.analyze()
        return jsonify(doc or {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/regime")
def api_regime():
    """Live predictions from our own trained neural-network suite (Regime Radar).
    Five models, one per asset class — distinct from the LLM features, which use
    a pre-trained model; these we trained ourselves."""
    def build():
        import regime_ai
        preds = regime_ai.predict_all()
        report = _load(os.path.join(HERE, "logs", "regime_report.json"), {})
        if not preds:
            return {"available": False}
        return {"predictions": preds, "report": report}
    try:
        return jsonify(cached("regime", 1200, build))
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


# ---------------- AI content features (all fail open, all grounded in real data) ----
@app.route("/api/ai/briefing")
def api_ai_briefing():
    try:
        import ai_features
        doc = cached("ai_briefing", 1800, ai_features.briefing)
        return jsonify(doc or {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ai/coach", methods=["POST"])
def api_ai_coach():
    p = request.get_json(force=True, silent=True) or {}
    try:
        import ai_features
        text = ai_features.coach(p.get("question", ""), p.get("history"))
        return jsonify({"answer": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ai/glossary", methods=["POST"])
def api_ai_glossary():
    term = (request.get_json(force=True, silent=True) or {}).get("term", "")
    try:
        import ai_features
        text = cached("gloss_" + term.strip().lower()[:60], 86400,
                      lambda: ai_features.glossary(term))
        return jsonify({"text": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ai/strategy/<sid>")
def api_ai_strategy(sid):
    try:
        import ai_features
        text = cached("deepdive_" + sid, 86400, lambda: ai_features.strategy_deep_dive(sid))
        return jsonify({"text": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ai/positioning")
def api_ai_positioning():
    try:
        import ai_features
        doc = cached("ai_positioning", 1200, ai_features.positioning)
        return jsonify(doc or {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/ai/compare", methods=["POST"])
def api_ai_compare():
    p = request.get_json(force=True, silent=True) or {}
    a, b = p.get("a", ""), p.get("b", "")
    try:
        import ai_features
        key = "cmp_" + "_".join(sorted([a, b]))
        text = cached(key, 86400, lambda: ai_features.compare(a, b))
        return jsonify({"text": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Constructs a CUSTOM mix continuously from all six answers — every answer
    moves the weights (caps bind, crypto has eligibility gates, priority tilts).
    Fully transparent: returns the 'why' trail + a 0-100 risk-capacity score."""
    p = request.get_json(force=True, silent=True) or {}
    exp = p.get("experience", "some"); hor = p.get("horizon", "3-10y")
    dd = p.get("max_dd", "-25%"); react = p.get("react", "hold")
    goal = p.get("goal", "steady"); prio = p.get("priority", "beat")
    raw = ({"preserve": 0, "steady": 2, "maximize": 4}.get(goal, 2)
           + {"-10%": 0, "-25%": 1.5, "-50%+": 3}.get(dd, 1.5)
           + {"sell": 0, "hold": 1, "buy": 2}.get(react, 1)
           + {"<1y": 0, "1-3y": 1, "3-10y": 2, "10y+": 2.5}.get(hor, 2)
           + {"new": 0, "some": 1, "experienced": 1.5}.get(exp, 1))
    caps, why = [], []
    for cond, cap, msg in [
        (exp == "new", 5.0, "New to investing — leverage and crypto are capped out of your mix"),
        (hor == "<1y", 1.5, "Money needed within a year gets the most defensive construction"),
        (hor == "1-3y", 5.5, "A 1-3 year horizon can't ride out deep drawdowns — risk capacity capped"),
        (dd == "-10%", 4.5, "A -10% pain limit rules out the higher-drawdown engines"),
        (dd == "-25%", 8.5, "A -25% pain limit keeps the crypto engine to a small satellite"),
        (react == "sell", 4.0, "You'd sell in a -20% month — strategies that CAN fall that far are excluded"),
    ]:
        if cond:
            caps.append(cap)
            if cap < raw:
                why.append(msg)
    score = min([raw] + caps)
    f = max(0.0, min(1.0, score / 13.0))
    capacity = round(f * 100)
    # continuous allocation recipe (weights are functions of risk capacity f)
    crypto_ok = (dd == "-50%+") and (exp != "new") and (hor in ("3-10y", "10y+"))
    w = {"risk_parity": 45 * (1 - f) ** 1.6,
         "gold_macro": 18 + 8 * (1 - f),
         "stack_bal": 24,
         "equity_season": 12 + 10 * f,
         "commod_trend": 5 + 9 * f,
         "stack_agg": 30 * f ** 1.3,
         "crypto_trend": (24 * f ** 2.0) if crypto_ok else 0.0}
    tilt = {"smooth": {"risk_parity": 1.4, "gold_macro": 1.25, "stack_agg": 0.7, "crypto_trend": 0.65},
            "beat": {"stack_bal": 1.3, "equity_season": 1.15},
            "max": {"stack_agg": 1.35, "crypto_trend": 1.35, "risk_parity": 0.55}}[prio]
    for k, m in tilt.items():
        w[k] = w.get(k, 0) * m
    # drop dust (<5%) then normalize to exactly 100 (largest remainder)
    tot = sum(w.values())
    w = {k: v for k, v in w.items() if v / tot >= 0.05}
    tot = sum(w.values())
    fl = {k: int(v * 100 / tot) for k, v in w.items()}
    rem = 100 - sum(fl.values())
    for k in sorted(w, key=lambda k: -(w[k] * 100 / tot - fl[k]))[:rem]:
        fl[k] += 1
    mix = {k: v for k, v in sorted(fl.items(), key=lambda z: -z[1]) if v > 0}
    tier = 0 if f < 0.25 else 1 if f < 0.5 else 2 if f < 0.75 else 3
    if not why:
        why.append({"preserve": "You asked to preserve capital first — defensive engines dominate",
                    "steady": "Steady-growth goal with matching tolerance — a diversified core",
                    "maximize": "Max-growth goal with the tolerance to back it"}.get(goal, ""))
    if crypto_ok and mix.get("crypto_trend"):
        why.append("Deep drawdown tolerance + long horizon unlocked a crypto momentum satellite")
    elif not crypto_ok:
        why.append("Crypto stayed locked out (needs -50% tolerance, experience and a 3y+ horizon)")
    why.append({"smooth": "Priority 'smooth ride' tilted weight into risk parity and gold",
                "beat": "Priority 'beat the market' put the multi-edge engines at the centre",
                "max": "Priority 'max wealth' tilted into the levered engines"}[prio])
    cat = {s["id"]: s for s in _load(CATALOG, {"strategies": []})["strategies"]}

    def _parts_blend(m):
        parts, cagr, vol, ddw = [], 0.0, 0.0, 0.0
        for sid, wt in m.items():
            s = cat.get(sid)
            if not s:
                continue
            st = s["stats"]
            parts.append({"id": sid, "name": s["name"], "weight": wt, "risk": s["risk"],
                          "cagr": st["cagr"], "sharpe": st["sharpe"], "max_dd": st["max_dd"]})
            cagr += wt / 100 * st["cagr"]; vol += wt / 100 * st["vol"]; ddw += wt / 100 * st["max_dd"]
        return parts, {"cagr_est": round(cagr, 1), "vol_est": round(vol, 1), "dd_est": round(ddw, 1)}

    parts, blend = _parts_blend(mix)
    freetext = p.get("freetext")
    if freetext:
        try:
            import quiz_nudge
            nudged = quiz_nudge.nudge(mix, parts, freetext)
        except Exception:
            nudged = None
        if nudged:
            mix, reason = nudged
            parts, blend = _parts_blend(mix)
            why.append(f"Your own words: {reason}")
    return jsonify({"tier": tier, "label": TIER_LABELS[tier], "capacity": capacity,
                    "why": [x for x in why if x], "mix": mix, "parts": parts, "blend": blend})


@app.route("/api/user/create", methods=["POST"])
def user_create():
    p = request.get_json(force=True, silent=True) or {}
    # A Blackbox is now tied to a signed-in account (no anonymous portfolios),
    # so it's saved to a real user and visible on their dashboard on any device.
    if not _me()[0]:
        return jsonify({"need_auth": True}), 401
    name = (p.get("name") or "My").strip()[:24] or "My"
    mix = {k: float(v) for k, v in (p.get("mix") or {}).items() if k in VALID_IDS and float(v) > 0}
    tot = sum(mix.values())
    if not mix or not (95 <= tot <= 105):
        return jsonify({"error": "mix must sum to 100%"}), 400
    mix = {k: round(v * 100 / tot, 1) for k, v in mix.items()}
    users = _load(USERS, {})
    uid = secrets.token_hex(4)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Baseline the mix's intraday move AT creation, so a brand-new book reads
    # exactly 0.00% now and only accrues the move made *after* you started —
    # you never inherit the part of today's move that happened before you began.
    try:
        from user_engine import marks as _marks
        mk0 = _marks()["strategies"]
        base_day_ret = sum((w / 100.0) * mk0.get(sid, {}).get("tilt", 1.0)
                           * mk0.get(sid, {}).get("day_ret", 0.0)
                           for sid, w in mix.items())
    except Exception:
        base_day_ret = 0.0
    users[uid] = {"name": name, "mix": mix, "created": today,
                  "base_nav": 10000.0, "base_date": today, "base_day_ret": base_day_ret,
                  "history": [[today, 10000.0]]}
    # signed in -> this becomes the account's active Blackbox (replacing any old one)
    a, aid = _me()
    if a:
        old = a.get("uid")
        if old and old in users and old != uid:
            users.pop(old, None)
        accounts = _accounts()
        accounts[aid]["uid"] = uid
        if not p.get("name"):
            users[uid]["name"] = a["name"]
        _save_accounts(accounts)
    _save_users(users)
    mixtxt = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(mix.items(), key=lambda z: -z[1]))
    _notify(f"New Blackbox started: \"{users[uid]['name']}\" ({'account' if a else 'anon'}) — {mixtxt}")
    return jsonify({"user_id": uid, "signed_in": bool(a)})


# ---------------- accounts (recovery-key now, Google when GOOGLE_CLIENT_ID set) ----
WORDS = ("ember violet falcon drift nova quartz cedar lumen onyx delta raven pixel "
         "zephyr cobalt aurora slate ridge polar comet marble tigris echo umbra vega "
         "sable orchid krypton fjord saffron helix mesa argon").split()


def _accounts():
    return _load(ACCOUNTS, {})


def _save_accounts(a):
    tmp = ACCOUNTS + ".tmp"
    json.dump(a, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, ACCOUNTS)


def _gen_key():
    r = secrets.SystemRandom()
    return "-".join(r.sample(WORDS, 3)) + "-" + str(r.randrange(1000, 9999))


def _hash_key(key, salt):
    return hashlib.sha256((salt + key.strip().lower()).encode()).hexdigest()


def _me():
    aid = session.get("aid")
    if not aid:
        return None, None
    return _accounts().get(aid), aid


def _login(aid):
    session.permanent = True
    session["aid"] = aid


@app.route("/api/config")
def api_config():
    return jsonify({"google_client_id": os.getenv("GOOGLE_CLIENT_ID") or None})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    p = request.get_json(force=True, silent=True) or {}
    name = (p.get("name") or "").strip()[:24] or "Trader"
    accounts = _accounts()
    aid = secrets.token_hex(8)
    key = _gen_key()
    salt = secrets.token_hex(8)
    acct = {"name": name, "salt": salt, "key_hash": _hash_key(key, salt),
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "uid": None, "google_sub": None, "email": None, "picture": None}
    adopt = p.get("adopt_uid")
    if adopt and adopt in _load(USERS, {}):
        acct["uid"] = adopt
    accounts[aid] = acct
    _save_accounts(accounts)
    _login(aid)
    _notify(f"New platform account: \"{name}\" (recovery-key), total {len(accounts)}")
    return jsonify({"ok": True, "name": name, "recovery_key": key})


@app.route("/api/auth/recover", methods=["POST"])
def auth_recover():
    p = request.get_json(force=True, silent=True) or {}
    key = (p.get("key") or "")
    accounts = _accounts()
    for aid, a in accounts.items():
        if a.get("key_hash") and a["key_hash"] == _hash_key(key, a.get("salt", "")):
            _login(aid)
            return jsonify({"ok": True, "name": a["name"]})
    return jsonify({"error": "key not recognised"}), 401


@app.route("/api/auth/google", methods=["POST"])
def auth_google():
    cid = os.getenv("GOOGLE_CLIENT_ID")
    if not cid:
        return jsonify({"error": "google not configured"}), 400
    p = request.get_json(force=True, silent=True) or {}
    cred = p.get("credential") or ""
    try:
        import requests as rq
        info = rq.get("https://oauth2.googleapis.com/tokeninfo",
                      params={"id_token": cred}, timeout=10).json()
    except Exception:
        return jsonify({"error": "verification unavailable"}), 502
    if (info.get("aud") != cid or "accounts.google.com" not in str(info.get("iss", ""))
            or int(info.get("exp", 0)) < time.time()):
        return jsonify({"error": "invalid token"}), 401
    sub = info.get("sub")
    accounts = _accounts()
    aid = next((k for k, a in accounts.items() if a.get("google_sub") == sub), None)
    if aid is None:
        aid = secrets.token_hex(8)
        accounts[aid] = {"name": (info.get("given_name") or info.get("name") or "Trader")[:24],
                         "salt": None, "key_hash": None,
                         "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                         "uid": None, "google_sub": sub}
        adopt = p.get("adopt_uid")
        if adopt and adopt in _load(USERS, {}):
            accounts[aid]["uid"] = adopt
        _notify(f"New platform account: \"{accounts[aid]['name']}\" (Google), total {len(accounts)}")
    accounts[aid]["email"] = info.get("email")
    accounts[aid]["picture"] = info.get("picture")
    _save_accounts(accounts)
    _login(aid)
    return jsonify({"ok": True, "name": accounts[aid]["name"]})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("aid", None)
    return jsonify({"ok": True})


@app.route("/api/me")
def api_me():
    a, aid = _me()
    if not a:
        return jsonify({"signed_in": False})
    portfolio = _user_payload(a["uid"]) if a.get("uid") else None
    return jsonify({"signed_in": True, "name": a["name"], "email": a.get("email"),
                    "picture": a.get("picture"), "google": bool(a.get("google_sub")),
                    "portfolio": portfolio})


def _user_payload(uid):
    users = _load(USERS, {})
    u = users.get(uid)
    if not u:
        return None
    try:
        from user_engine import marks
        mk = marks()["strategies"]
    except Exception:
        mk = {}
    day = 0.0
    strat_view = []
    for sid, w in u["mix"].items():
        m = mk.get(sid, {})
        contrib = (w / 100) * m.get("tilt", 1.0) * m.get("day_ret", 0.0)
        day += contrib
        strat_view.append({"id": sid, "weight": w, "day_ret": m.get("day_ret", 0.0),
                           "gross": m.get("gross", 0), "note": m.get("note", ""),
                           "tilt": m.get("tilt", 1.0), "tilt_why": m.get("tilt_why"),
                           "sentiment": m.get("sentiment", {}), "holdings": m.get("holdings", [])})
    # display-only: user_engine.settle_users() (scanner daemon, after the 21:00 UTC
    # daily close) is the ONE place books roll — no request-time folding here.
    # On the creation day only, subtract the intraday move captured at creation so
    # the book starts at exactly 0% and grows from the moment you began.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    off = u.get("base_day_ret")
    eff = day - off if (off is not None and u.get("base_date") == today) else day
    nav_now = round(u["base_nav"] * (1 + eff), 2)
    return {"uid": uid, "name": u["name"], "created": u["created"], "mix": u["mix"],
            "nav": nav_now, "day_ret": round(eff * 100, 3),
            "since_start": round((nav_now / 10000 - 1) * 100, 2),
            "history": u["history"], "strategies": strat_view,
            "note": "Paper portfolio marked to live markets by the same validated rules. Estimates, not investment advice."}


@app.route("/api/user/<uid>", methods=["GET"])
def user_get(uid):
    p = _user_payload(uid)
    if not p:
        return jsonify({"error": "not found"}), 404
    return jsonify(p)


@app.route("/api/user/<uid>/narrate")
def user_narrate(uid):
    p = _user_payload(uid)
    if not p:
        return jsonify({"error": "not found"}), 404
    try:
        import portfolio_narrator as pn
        text = cached("narrate_" + uid, 900, lambda: pn.narrate(p))
        return jsonify({"text": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/user/<uid>/ask", methods=["POST"])
def user_ask(uid):
    p = _user_payload(uid)
    if not p:
        return jsonify({"error": "not found"}), 404
    q = (request.get_json(force=True, silent=True) or {}).get("question", "")
    try:
        import portfolio_narrator as pn
        text = pn.answer(p, q)
        return jsonify({"answer": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/user/<uid>/review")
def user_review(uid):
    p = _user_payload(uid)
    if not p:
        return jsonify({"error": "not found"}), 404
    try:
        import ai_features
        mixkey = "review_" + "_".join(f"{k}{v}" for k, v in sorted((p.get("mix") or {}).items()))
        text = cached(mixkey, 3600, lambda: ai_features.portfolio_review(p))
        return jsonify({"text": text} if text else {"available": False})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/api/user/<uid>/delete", methods=["POST"])
def user_delete(uid):
    users = _load(USERS, {})
    users.pop(uid, None)
    _save_users(users)
    a, aid = _me()
    if a and a.get("uid") == uid:
        accounts = _accounts()
        accounts[aid]["uid"] = None
        _save_accounts(accounts)
    return jsonify({"ok": True})


@app.route("/icon.png")
def icon():
    p = os.path.join(HERE, "icon.png")
    if os.path.exists(p):
        return Response(open(p, "rb").read(), mimetype="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    return Response(status=404)


@app.route("/")
@app.route("/start")
@app.route("/strategies")
@app.route("/briefing")
@app.route("/intel")
@app.route("/learn")
@app.route("/ai")
@app.route("/compare")
@app.route("/account")
def index():
    return render_template_string(PAGE)


PAGE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blackbox — Real strategies. Real numbers.</title>
<link rel="icon" href="/icon.png"><link rel="apple-touch-icon" href="/icon.png">
<meta name="theme-color" content="#0B0906">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script src="https://accounts.google.com/gsi/client" async defer></script>
<style>
:root{--bg:#0B0906;--surface:#171310;--surface2:#1F1912;--line:rgba(201,168,106,.16);
--txt:#F3ECDD;--mut:#A79A82;--acc:#C9A961;--acc2:#B08D4F;--deep:#8B6B32;
--green:#5FBF8F;--red:#E2716B;--ease:cubic-bezier(.22,1,.36,1)}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--txt);font:16px/1.6 'Inter',sans-serif;-webkit-font-smoothing:antialiased;overflow-x:hidden}
h1,h2,h3{font-family:'Fraunces','Inter',serif;letter-spacing:-.01em}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
.rv{opacity:0;transform:translateY(24px);transition:opacity .7s var(--ease),transform .7s var(--ease)}
.rv.in{opacity:1;transform:none}
.hero{position:relative;min-height:100svh;display:flex;align-items:center;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(90px);opacity:.5;pointer-events:none}
.o1{width:560px;height:560px;background:radial-gradient(circle,#C9A96155,transparent 65%);top:-120px;left:-120px;animation:d1 22s ease-in-out infinite}
.o2{width:640px;height:640px;background:radial-gradient(circle,#8B6B3245,transparent 65%);bottom:-200px;right:-160px;animation:d2 26s ease-in-out infinite}
@keyframes d1{50%{transform:translate(70px,50px)}}
@keyframes d2{50%{transform:translate(-60px,-40px)}}
.hero::after{content:"";position:absolute;inset:0;background:rgba(11,9,6,.3)}
.hero .wrap{position:relative;z-index:2;padding:90px 24px 60px}
.kicker{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;color:var(--acc);
border:1px solid var(--line);border-radius:999px;padding:7px 14px;background:rgba(201,168,106,.07);margin-bottom:26px}
.kicker .dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pl 2.4s infinite}
@keyframes pl{50%{opacity:.4}}
h1{font-family:'Fraunces','Inter',serif;font-size:clamp(42px,7.2vw,88px);font-weight:600;font-style:italic;line-height:1.05}
h1 .w{display:inline-block;opacity:0;transform:translateY(26px);animation:win .7s var(--ease) forwards}
@keyframes win{to{opacity:1;transform:none}}
.grad{background:linear-gradient(92deg,#E8D5A3,#C9A961);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;font-style:normal}
.sub{max-width:560px;color:var(--mut);font-size:18px;margin:24px 0 36px}
.cta{display:inline-flex;align-items:center;gap:10px;background:linear-gradient(135deg,#DCBB78,#B08D4F);
color:#1A140A;font-weight:700;font-size:16px;padding:16px 30px;border-radius:14px;border:0;cursor:pointer;
box-shadow:0 10px 40px -8px rgba(160,124,58,.55);transition:transform .16s var(--ease),box-shadow .2s var(--ease)}
.cta:hover{box-shadow:0 14px 52px -8px rgba(160,124,58,.75)}
.cta:active{transform:scale(.97)}
.cta.ghost{background:var(--surface2);border:1px solid var(--line);box-shadow:none;color:var(--txt)}
.cta:disabled{opacity:.45;cursor:not-allowed}
.hstats{display:flex;gap:34px;margin-top:56px;flex-wrap:wrap}
.hstat .n{font-family:'Fraunces';font-size:26px;font-weight:700}
.hstat .l{font-size:12.5px;color:var(--mut)}
section{padding:96px 0}
.sec-k{font-size:13px;font-weight:700;color:var(--acc);text-transform:uppercase;letter-spacing:.14em;margin-bottom:14px}
h2{font-size:clamp(30px,4vw,48px);font-weight:800;margin-bottom:14px}
.lead{color:var(--mut);max-width:640px;font-size:17px}
/* journey */
.modes{display:flex;gap:14px;margin-top:36px;flex-wrap:wrap}
.mode{flex:1;min-width:280px;background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:26px;cursor:pointer;
transition:transform .2s var(--ease),border-color .2s}
.mode:active{transform:scale(.985)}
.mode.on{border-color:var(--acc);background:rgba(201,168,106,.08)}
.mode h3{font-size:19px;margin-bottom:6px}
.mode p{color:var(--mut);font-size:14.5px}
.quiz{background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:36px;margin-top:22px;display:none}
.quiz.show,.builderwrap.show{display:block;animation:fup .5s var(--ease)}
@keyframes fup{from{opacity:0;transform:translateY(14px)}to{opacity:1}}
.q{margin-bottom:24px}
.q .ql{font-weight:600;margin-bottom:11px}
.chips{display:flex;gap:10px;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:var(--surface2);color:var(--mut);border-radius:12px;
padding:11px 18px;font-size:14.5px;font-weight:600;cursor:pointer;transition:transform .15s var(--ease),border-color .2s,color .2s,background .2s}
.chip:active{transform:scale(.96)}
.chip.on{border-color:var(--acc);color:var(--txt);background:rgba(201,168,106,.18)}
.reco{display:none;margin-top:30px;border-top:1px solid var(--line);padding-top:28px}
.reco.show{display:block;animation:fup .5s var(--ease)}
.why{margin:14px 0 4px;padding:0 0 0 2px}
.why li{list-style:none;color:var(--mut);font-size:14px;padding:5px 0 5px 26px;position:relative}
.why li::before{content:"→";position:absolute;left:0;color:var(--acc)}
.blend{display:flex;gap:26px;flex-wrap:wrap;margin:18px 0 8px}
.b .n{font-family:'Inter';font-size:22px;font-weight:800}
.b .l{font-size:12px;color:var(--mut)}
.mixbar{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:16px 0 10px;border:1px solid var(--line)}
.mixbar i{display:block;height:100%}
.mixlegend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mut)}
.mixlegend b{color:var(--txt)}
.dotc{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:6px;vertical-align:1px}
.namebox{display:flex;gap:12px;margin-top:22px;flex-wrap:wrap;align-items:center}
.namebox input{background:var(--surface2);border:1px solid var(--line);border-radius:12px;color:var(--txt);
padding:14px 16px;font:600 15px 'Inter';min-width:220px}
.namebox input:focus{outline:none;border-color:var(--acc)}
.qtext{width:100%;background:var(--surface2);border:1px solid var(--line);border-radius:12px;color:var(--txt);
padding:14px 16px;font:400 14.5px/1.6 'Inter';resize:vertical;min-height:80px}
.qtext:focus{outline:none;border-color:var(--acc)}
/* builder */
.builderwrap{display:none;margin-top:22px}
.bbar{position:sticky;bottom:14px;z-index:30;background:rgba(23,19,16,.92);backdrop-filter:blur(12px);
border:1px solid var(--line);border-radius:18px;padding:16px 22px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;
box-shadow:0 18px 50px -20px rgba(0,0,0,.7);margin-top:26px}
.bbar .tot{font-family:'Inter';font-weight:800;font-size:22px}
.bbar .bs{font-size:13px;color:var(--mut)}
.step{display:flex;align-items:center;gap:8px;margin-top:14px}
.step button{width:34px;height:34px;border-radius:10px;border:1px solid var(--line);background:var(--surface2);
color:var(--txt);font-size:17px;font-weight:700;cursor:pointer;transition:transform .13s}
.step button:active{transform:scale(.92)}
.step .v{font-family:'Inter';font-weight:800;min-width:52px;text-align:center;font-size:17px}
/* catalog */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px;margin-top:44px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:24px;cursor:pointer;
transition:transform .22s var(--ease),border-color .22s,box-shadow .22s var(--ease)}
@media(hover:hover){.card:hover{transform:translateY(-4px);border-color:rgba(201,168,106,.5);box-shadow:0 18px 50px -20px rgba(160,124,58,.5)}}
.card .top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px}
.card h3{font-size:19px;line-height:1.25}
.badge{font-size:11px;font-weight:700;padding:4px 10px;border-radius:999px;white-space:nowrap}
.badge.Conservative{background:rgba(95,191,143,.14);color:var(--green)}
.badge.Moderate{background:rgba(201,168,106,.16);color:var(--acc)}
.badge.High{background:rgba(251,191,36,.13);color:#fbbf24}
.badge.VeryHigh{background:rgba(251,113,133,.13);color:var(--red)}
.card .tag{color:var(--mut);font-size:14px;min-height:44px;margin-bottom:12px}
.sentchip{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:700;border-radius:999px;
padding:4px 11px;margin-bottom:12px;background:var(--surface2);border:1px solid var(--line)}
.sentchip .sd{width:7px;height:7px;border-radius:50%}
.spark{width:100%;height:56px;display:block;margin-bottom:14px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.st .n{font-family:'Inter';font-weight:800;font-size:16.5px}
.st .l{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
/* modal */
.mask{position:fixed;inset:0;background:rgba(8,6,4,.72);backdrop-filter:blur(6px);z-index:50;display:none;align-items:center;justify-content:center;padding:20px}
.mask.show{display:flex}
.modal{background:var(--surface);border:1px solid var(--line);border-radius:24px;max-width:960px;width:100%;
max-height:90vh;overflow-y:auto;padding:34px;transform:scale(.96);opacity:0;transition:transform .25s var(--ease),opacity .25s}
.mask.show .modal{transform:scale(1);opacity:1}
.modal .x{float:right;background:var(--surface2);border:1px solid var(--line);color:var(--mut);width:36px;height:36px;
border-radius:10px;font-size:17px;cursor:pointer}
.mrow{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:24px}
@media(max-width:820px){.mrow{grid-template-columns:1fr}}
.mbox{background:var(--surface2);border:1px solid var(--line);border-radius:16px;padding:20px;min-width:0}
.mbox h4{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
/* Chart.js REQUIRES a fixed-height positioned wrapper with responsive mode —
   height on the canvas itself makes it grow unbounded. */
.chartbox{position:relative;height:280px;min-width:0}
.chartbox.short{height:230px}
.chartbox canvas{position:absolute;inset:0}
.modal{overflow-x:hidden}
.modal .stats{grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:14px 10px}
.hchips{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.hchip{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:10px 14px;font-size:12.5px}
.hchip .n{font-family:'Inter';font-weight:800;font-size:15px;display:block}
.method{font-size:14px;color:var(--mut);line-height:1.65;margin-top:20px;border-left:3px solid var(--deep);padding-left:14px}
/* me */
#me{background:var(--surface);border-block:1px solid var(--line)}
.mehead{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
.nav-big{font-family:'Fraunces';font-size:clamp(40px,6vw,64px);font-weight:600;letter-spacing:-.01em}
.pill{display:inline-block;font-size:13px;font-weight:700;border-radius:999px;padding:6px 14px;margin-left:8px;vertical-align:14px}
.pill.up{background:rgba(95,191,143,.15);color:var(--green)}
.pill.dn{background:rgba(226,113,107,.15);color:var(--red)}
.metric-term{cursor:help;border-bottom:1px dotted var(--acc);transition:color .15s}
.metric-term:hover{color:var(--acc)}
.mpop{position:fixed;z-index:70;max-width:300px;background:var(--surface2);border:1px solid var(--acc);
border-radius:14px;padding:14px 16px;box-shadow:0 20px 50px -18px rgba(0,0,0,.8);font-size:13px;line-height:1.6;color:var(--txt)}
.mpop h5{font-size:13px;color:var(--acc);margin-bottom:6px;font-family:'Inter';font-weight:700}
.dstats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:14px 10px}
.dstat .dn{font-family:'Inter';font-weight:800;font-size:18px}
.dstat .dl{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.mehead-btns{display:flex;gap:10px;flex-wrap:wrap}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.badge2{display:flex;align-items:center;gap:8px;background:var(--surface2);border:1px solid var(--line);
border-radius:999px;padding:8px 14px;font-size:12.5px;font-weight:600;cursor:default}
.badge2 .be{font-size:15px}
.badge2 .bt{color:var(--txt)}
.megrid{display:grid;grid-template-columns:1.2fr 1fr;gap:20px;margin-top:32px}
@media(max-width:900px){.megrid{grid-template-columns:1fr}}
.scard2{background:var(--surface2);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:14px}
.scard2 .h2r{display:flex;justify-content:space-between;align-items:center;gap:10px}
.scard2 h3{font-size:16.5px}
.wtag{font-family:'Inter';font-weight:800;color:var(--acc)}
.note{color:var(--mut);font-size:13.5px;margin-top:6px}
.tiltwarn{margin-top:10px;font-size:12.5px;color:#fbbf24;background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);
border-radius:10px;padding:8px 12px}
.hlist{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.htag{font-size:12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:5px 10px;color:var(--mut)}
.htag b{color:var(--txt)}
.skel{color:var(--mut);padding:40px;text-align:center;font-size:15px}
.skel .sp{display:inline-block;width:16px;height:16px;border:2px solid var(--acc);border-top-color:transparent;border-radius:50%;
animation:spin .8s linear infinite;vertical-align:-3px;margin-right:10px}
@keyframes spin{to{transform:rotate(360deg)}}
/* sentiment section */
.sgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px;margin-top:44px}
.scard{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:22px}
.scard .h{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.scard h3{font-size:17px;text-transform:capitalize}
.gauge{height:10px;border-radius:5px;background:linear-gradient(90deg,#E2716B,#2B241A 50%,#5FBF8F);position:relative;margin:10px 0 6px}
.gauge i{position:absolute;top:-4px;width:4px;height:18px;border-radius:2px;background:#fff;box-shadow:0 0 8px rgba(255,255,255,.6);transition:left .8s var(--ease)}
.sscore{font-family:'Inter';font-weight:800;font-size:22px}
.sexp{font-size:13.5px;color:var(--mut);margin:10px 0 12px;min-height:40px}
.head{font-size:12.5px;padding:8px 0;border-top:1px solid rgba(201,168,106,.1);color:var(--mut);display:flex;gap:8px;justify-content:space-between}
.acard{background:var(--surface2);border:1px solid var(--line);border-radius:20px;padding:26px;margin-top:32px}
.acard .ah{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.acard h3{font-size:19px}
.adate{font-size:13px;color:var(--mut);white-space:nowrap}
.areason{font-size:13.5px;color:var(--mut);margin:14px 0 20px;padding:12px 14px;background:rgba(201,168,106,.07);border-radius:12px;border-left:2px solid var(--acc)}
.agrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.arow{display:flex;justify-content:space-between;padding:11px 13px;background:var(--surface);border-radius:10px;font-size:13px}
.askel{color:var(--mut);font-size:14px;display:flex;align-items:center;gap:10px;padding:20px 0}
.qarow{font-size:13.5px;padding:10px 0;border-top:1px solid rgba(201,168,106,.1)}
.qarow .q{color:var(--txt);font-weight:600;margin-bottom:4px}
.qarow .a{color:var(--mut)}
.qform{display:flex;gap:8px;margin-top:12px}
.qform .ain{padding:11px 14px;font-weight:500;font-size:14px}
.head .t{flex:1}
.head .s{white-space:nowrap;font-weight:700}
/* long-form model deep-dive */
.corebar{display:inline-flex;align-items:center;gap:9px;font-size:12px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#1A140A;background:linear-gradient(135deg,#DCBB78,#B08D4F);padding:7px 15px;border-radius:999px;margin-bottom:20px}
.longform{max-width:840px;margin-top:8px}
.longform h3{font-family:'Fraunces';font-weight:600;font-size:23px;margin:38px 0 10px}
.longform h4{font-size:16px;margin:24px 0 6px;color:var(--txt)}
.longform p{color:var(--mut);font-size:15px;line-height:1.85;margin-bottom:15px}
.longform b{color:var(--txt)}
.longform em{color:var(--acc);font-style:normal;font-weight:600}
.longform ul{margin:0 0 15px;padding-left:22px;color:var(--mut);font-size:14.5px;line-height:1.75}
.longform li{padding:4px 0}
.longform li b{color:var(--txt)}
.dtable{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 18px}
.dtable th,.dtable td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
.dtable th{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.dtable td b{color:var(--acc)}
/* AI Intelligence overview + "how it works" expanders */
.aioverview{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:26px}
.aiov{background:var(--surface2);border:1px solid var(--line);border-radius:16px;padding:18px}
.aiov .ati{font-size:22px}
.aiov h4{font-size:15px;margin:8px 0 4px}
.aiov p{font-size:12.5px;color:var(--mut);line-height:1.55}
.aiov .tag2{margin-top:10px;display:inline-block;font-size:10px;padding:3px 9px;border-radius:999px;background:rgba(201,168,106,.12);color:var(--acc);font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.howto{margin-top:20px;background:var(--surface2);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.howto summary{cursor:pointer;padding:14px 18px;font-weight:600;font-size:13.5px;color:var(--acc);list-style:none;display:flex;align-items:center;gap:9px}
.howto summary::-webkit-details-marker{display:none}
.howto summary::before{content:"▸";transition:transform .2s var(--ease);font-size:11px}
.howto[open] summary::before{transform:rotate(90deg)}
.howto .hbody{padding:2px 20px 18px;font-size:13.5px;color:var(--mut);line-height:1.7}
.howto ol{margin:0;padding-left:22px}
.howto ol li{padding:5px 0}
.howto b{color:var(--txt)}
/* Regime Radar (our own trained model) */
.regcard{background:linear-gradient(135deg,rgba(201,168,106,.08),rgba(201,168,106,.02));border:1px solid var(--acc);border-radius:22px;padding:30px;margin-top:30px}
.reghead{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.regverdict{font-family:'Fraunces';font-style:italic;font-weight:600;font-size:clamp(34px,6vw,54px);line-height:1}
.regverdict.calm{color:var(--green)}.regverdict.turbulent{color:var(--red)}
.regmeta{color:var(--mut);font-size:13.5px}
.reggauge{height:12px;border-radius:6px;background:linear-gradient(90deg,#5FBF8F,#DCBB78,#E2716B);position:relative;margin:22px 0 8px}
.reggauge.sm{height:8px;margin:12px 0 6px}
.reggauge i{position:absolute;top:-5px;width:5px;height:22px;border-radius:3px;background:#fff;box-shadow:0 0 8px rgba(255,255,255,.7);transition:left .8s var(--ease)}
.reggauge.sm i{top:-4px;height:16px;width:4px}
.regmgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.regmini{background:var(--surface2);border:1px solid var(--line);border-radius:16px;padding:18px}
.rmname{font-family:'Inter';font-weight:800;font-size:15px;display:flex;flex-direction:column;gap:2px}
.rmasset{font-size:11px;color:var(--mut);font-weight:500;text-transform:uppercase;letter-spacing:.04em}
.regverdict2{font-family:'Fraunces';font-style:italic;font-weight:600;font-size:26px;line-height:1;margin-top:10px}
.regverdict2.calm{color:var(--green)}.regverdict2.turbulent{color:var(--red)}
.rmmeta{font-size:11.5px;color:var(--mut);line-height:1.5}
.regfind{margin-top:18px;font-size:13.5px;color:var(--txt);background:rgba(201,168,106,.08);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.regstats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-top:22px}
.regstat{background:var(--surface2);border:1px solid var(--line);border-radius:14px;padding:14px}
.regstat .rn{font-family:'Inter';font-weight:800;font-size:20px}
.regstat .rl{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-top:3px}
.regnote{font-size:13.5px;color:var(--mut);line-height:1.7;margin-top:20px;border-left:3px solid var(--acc);padding-left:16px}
.regtable{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}
.regtable th,.regtable td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}
.regtable th{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.regtable td.win{color:var(--green);font-weight:700}
/* AI Inside explainer page */
.principle{background:linear-gradient(135deg,rgba(201,168,106,.1),rgba(201,168,106,.03));border:1px solid var(--acc);
border-radius:20px;padding:28px;margin-top:34px}
.principle .pbig{font-family:'Fraunces';font-style:italic;font-size:clamp(22px,3.2vw,30px);font-weight:600;color:var(--txt);margin-bottom:12px}
.principle p{color:var(--mut);font-size:15px;line-height:1.7}
.airole-h{font-family:'Fraunces';font-weight:600;font-size:22px;margin:48px 0 4px}
.capgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:18px}
.capcard{background:var(--surface2);border:1px solid var(--line);border-radius:18px;padding:22px}
.capcard .capi{font-size:26px;margin-bottom:10px}
.capcard h4{font-size:16.5px;margin-bottom:8px}
.capcard p{color:var(--mut);font-size:13.8px;line-height:1.65}
.honesty{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:28px;margin-top:48px}
.honesty h3{font-size:22px;margin-bottom:16px}
.honesty ul{list-style:none;padding:0}
.honesty li{color:var(--mut);font-size:14px;line-height:1.7;padding:12px 0 12px 26px;position:relative;border-top:1px solid var(--line)}
.honesty li:first-child{border-top:0}
.honesty li::before{content:"✓";position:absolute;left:0;color:var(--acc);font-weight:700}
.honesty b{color:var(--txt)}
.aitech{margin-top:34px;border-left:3px solid var(--deep);padding:6px 0 6px 18px}
.aitech h4{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:8px}
.aitech p{color:var(--mut);font-size:14px;line-height:1.7}
/* AI content: briefing / positioning / learn / compare */
.airich{font-size:14.5px;line-height:1.75;color:var(--txt);white-space:pre-wrap}
.briefrow{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}
.bscore{flex:1;min-width:100px;background:var(--surface2);border:1px solid var(--line);border-radius:14px;padding:14px}
.bscore .ba{font-size:11px;color:var(--mut);text-transform:capitalize}
.bscore .bv{font-family:'Inter';font-weight:800;font-size:22px;margin-top:3px}
.posgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:36px}
.poscard{background:var(--surface2);border:1px solid var(--line);border-radius:18px;padding:20px}
.poscard h3{font-size:16.5px;display:flex;justify-content:space-between;align-items:center;gap:10px}
.postag{font-size:11px;font-weight:700;padding:3px 9px;border-radius:999px}
.postag.brake{background:rgba(226,113,107,.14);color:var(--red)}
.postag.clear{background:rgba(95,191,143,.14);color:var(--green)}
.posnote{font-size:13.5px;color:var(--mut);line-height:1.65;margin-top:10px}
.poshold{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.poshold .htag{font-size:11.5px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:4px 9px;color:var(--mut)}
.glchip{border:1px solid var(--line);background:var(--surface2);color:var(--txt);border-radius:999px;padding:9px 15px;font-size:13.5px;font-weight:600;cursor:pointer;transition:border-color .2s,background .2s}
.glchip:hover{border-color:var(--acc)}
.glchip.on{border-color:var(--acc);background:rgba(201,168,106,.12)}
.cmppick{display:flex;gap:16px;align-items:flex-end;margin-top:30px;flex-wrap:wrap}
.cmplabel{display:block;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.cmpsel{min-width:220px;cursor:pointer}
.cmpvs{font-family:'Fraunces';font-style:italic;color:var(--mut);padding-bottom:12px}
.cmptable{width:100%;border-collapse:collapse;font-size:14px}
.cmptable th,.cmptable td{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line)}
.cmptable th{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.cmptable td.metric{color:var(--mut)}
.cmptable td.val{font-family:'Inter';font-weight:700;color:var(--txt)}
/* top nav + auth */
.topnav{position:fixed;top:0;left:0;right:0;z-index:40;background:rgba(11,9,6,.65);
backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}
.tn{display:flex;align-items:center;height:60px;gap:14px}
.brand{font-family:'Fraunces';font-weight:600;font-style:italic;font-size:17px;letter-spacing:.02em;cursor:pointer;flex-shrink:0}
.navlinks{display:flex;gap:2px;flex:1;min-width:0;overflow-x:auto;scrollbar-width:none;-ms-overflow-style:none}
.navlinks::-webkit-scrollbar{display:none}
.navlinks a{color:var(--mut);font-size:13px;font-weight:600;padding:8px 12px;border-radius:999px;cursor:pointer;transition:color .2s,background .2s;white-space:nowrap;flex-shrink:0}
.navlinks a:hover{color:var(--txt)}
.navlinks a.on{color:var(--acc);background:rgba(201,168,106,.1)}
.navlinks a.star{color:var(--acc);font-weight:800}
.navlinks a.star::before{content:"◆";font-size:8px;vertical-align:1px;margin-right:5px;opacity:.85}
.navlinks a.star.on{background:rgba(201,168,106,.16)}
#authbox{flex-shrink:0}
.cta.small{padding:9px 18px;font-size:14px;border-radius:11px}
.uchip{display:flex;gap:10px;align-items:center}
.uchip img{width:30px;height:30px;border-radius:50%;border:1px solid var(--line)}
.uinit{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#DCBB78,#B08D4F);
display:grid;place-items:center;font-weight:800;font-size:14px;color:#1A140A}
.uchip .un{font-weight:600;font-size:14px;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.amodal{max-width:440px}
.ain{width:100%;background:var(--surface2);border:1px solid var(--line);border-radius:12px;color:var(--txt);
padding:14px 16px;font:600 15px 'Inter'}
.ain:focus{outline:none;border-color:var(--acc)}
.aerr{color:var(--red);font-size:13px;margin-top:10px;min-height:16px}
.keybox{font-family:'Inter';font-weight:800;font-size:19px;letter-spacing:.04em;text-align:center;
background:var(--surface2);border:1px dashed rgba(201,168,106,.55);border-radius:14px;padding:18px 14px;margin-top:14px;
user-select:all;word-break:break-all}
.ordiv{display:flex;align-items:center;gap:12px;color:var(--mut);font-size:12px;margin:16px 0}
.ordiv::before,.ordiv::after{content:"";flex:1;height:1px;background:var(--line)}
@media(max-width:640px){.uchip .un{display:none}}
footer{border-top:1px solid var(--line);padding:44px 0 60px;color:var(--mut);font-size:13px}
footer .d{max-width:820px;line-height:1.7}
@media(prefers-reduced-motion:reduce){.rv,.orb,h1 .w,.gauge i{animation:none!important;transition:none!important;opacity:1!important;transform:none!important}}
@media(max-width:640px){section{padding:64px 0}.quiz{padding:24px}.stats{grid-template-columns:repeat(2,1fr);gap:12px}}
</style></head><body>

<nav class="topnav"><div class="wrap tn">
  <div class="brand" onclick="go('home')">BLACKBOX</div>
  <div class="navlinks">
    <a data-page="home" onclick="go('home')">Home</a>
    <a data-page="start" onclick="go('start')">Start</a>
    <a data-page="strategies" onclick="go('strategies')">Strategies</a>
    <a data-page="briefing" onclick="go('briefing')">Briefing</a>
    <a data-page="intel" class="star" onclick="go('intel')">Intelligence</a>
    <a data-page="learn" onclick="go('learn')">Learn</a>
    <a data-page="ai" onclick="go('ai')">AI Inside</a>
    <a data-page="compare" onclick="go('compare')">Compare</a>
    <a data-page="account" onclick="go('account')">Your Blackbox</a>
  </div>
  <button id="langtog" onclick="toggleLang()" title="Sprache umschalten / Switch language"
    style="background:transparent;border:1px solid var(--line);color:var(--mut);font-size:12px;font-weight:700;padding:6px 10px;border-radius:999px;cursor:pointer;margin-right:8px;letter-spacing:.04em;flex-shrink:0">DE</button>
  <div id="authbox"><button class="cta ghost small" onclick="openAuth()">Sign in</button></div>
</div></nav>

<div id="page-home">
<section class="hero">
  <div class="orb o1"></div><div class="orb o2"></div>
  <div class="wrap">
    <div class="kicker"><span class="dot"></span>Live engine · 5 asset classes · AI market scanner</div>
    <h1 id="hl"></h1>
    <p class="sub">Choose an institutional-grade systematic strategy — or let six questions choose it for you — then press go. Your own Blackbox starts trading on paper, marked to live markets, guided by validated rules and an AI that reads the news.</p>
    <button class="cta" id="heroCta" onclick="heroGo()">Let's go →</button>
    <div class="hstats">
      <div class="hstat"><div class="n" id="hs1">—</div><div class="l">strategies, fully backtested</div></div>
      <div class="hstat"><div class="n">20 yrs</div><div class="l">of real market data</div></div>
      <div class="hstat"><div class="n" id="hs2">—</div><div class="l">sources scanned by the AI</div></div>
    </div>
  </div>
</section>
</div>

<div id="page-start" style="display:none">
<section id="journey" style="padding-top:150px">
  <div class="wrap">
    <div class="rv"><div class="sec-k">01 — Start your journey</div>
    <h2>Two ways in. One press of go.</h2>
    <p class="lead">Answer six questions and every answer visibly shapes the outcome — or skip the quiz and build your own mix straight from the numbers.</p></div>
    <div class="modes rv">
      <div class="mode" id="modeQuiz" onclick="setMode('quiz')"><h3>🧭 Guide me</h3><p>A six-question profile. We show exactly which answers capped or shaped your mix — no black box.</p></div>
      <div class="mode" id="modeManual" onclick="setMode('manual')"><h3>🎛 I'll choose myself</h3><p>Browse the strategies below and set your own weights by CAGR, volatility and max drawdown.</p></div>
    </div>

    <div class="quiz" id="quiz">
      <div class="q"><div class="ql">1 · How experienced are you with investing?</div>
        <div class="chips" data-k="experience"><button class="chip" data-v="new">I'm new</button><button class="chip on" data-v="some">Some experience</button><button class="chip" data-v="experienced">Experienced</button></div></div>
      <div class="q"><div class="ql">2 · How long can this money stay invested?</div>
        <div class="chips" data-k="horizon"><button class="chip" data-v="<1y">&lt;1 year</button><button class="chip" data-v="1-3y">1–3 years</button><button class="chip on" data-v="3-10y">3–10 years</button><button class="chip" data-v="10y+">10+ years</button></div></div>
      <div class="q"><div class="ql">3 · The most you could watch it fall without selling?</div>
        <div class="chips" data-k="max_dd"><button class="chip" data-v="-10%">−10%</button><button class="chip on" data-v="-25%">−25%</button><button class="chip" data-v="-50%+">−50% or more</button></div></div>
      <div class="q"><div class="ql">4 · Your portfolio drops 20% in a month. You…</div>
        <div class="chips" data-k="react"><button class="chip" data-v="sell">Sell everything</button><button class="chip on" data-v="hold">Hold, but worry</button><button class="chip" data-v="buy">Buy more</button></div></div>
      <div class="q"><div class="ql">5 · What's the goal?</div>
        <div class="chips" data-k="goal"><button class="chip" data-v="preserve">Preserve capital</button><button class="chip on" data-v="steady">Grow steadily</button><button class="chip" data-v="maximize">Maximize growth</button></div></div>
      <div class="q"><div class="ql">6 · What matters most to you?</div>
        <div class="chips" data-k="priority"><button class="chip" data-v="smooth">A smooth ride</button><button class="chip on" data-v="beat">Beating the market</button><button class="chip" data-v="max">Max long-term wealth</button></div></div>
      <div class="q">
        <div class="ql">In your own words — anything about your situation, goals or concerns the six questions didn't capture? (optional)</div>
        <textarea class="qtext" id="freetext" rows="3" placeholder="e.g. I just had a baby and want to be a bit more careful than usual, even though my risk tolerance on paper is high…"></textarea>
        <div class="note" style="margin-top:6px">If you write something here, AI reads it and may apply a small, bounded adjustment to your mix — it can never override the validated math above, only nudge it, and it always explains why.</div>
      </div>
      <button class="cta" onclick="recommend()">Build my mix →</button>
      <div class="reco" id="reco"></div>
    </div>

    <div class="reco show" id="namecard" style="display:none;border:0;padding-top:6px">
      <div class="namebox">
        <input id="uname" placeholder="Name your Blackbox (e.g. Nico's)" maxlength="24">
        <button class="cta" id="startBtn" onclick="startBlackbox()">🚀 Start my Blackbox</button>
      </div>
      <div class="note" style="margin-top:10px">Starts a personal paper portfolio of 10,000 marked to live markets. Demo capital — no real money.</div>
    </div>
  </div>
</section>
</div>

<div id="page-strategies" style="display:none">
<section id="catalog" style="padding-top:150px">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">01 — The strategies</div>
    <h2>Real numbers. Nothing invented.</h2>
    <p class="lead">Every stat is a 20-year backtest — true drawdowns included. The AI scanner's live read is on every card. Click a card for the full profile.</p></div>
    <div class="builderwrap" id="builderwrap">
      <p class="lead" style="margin-top:6px">Set weights with the steppers on the strategy cards below. Your mix must total 100%.</p>
      <div class="bbar" id="bbar">
        <div><div class="tot" id="btot">0%</div><div class="bs">allocated</div></div>
        <div style="flex:1;min-width:180px"><div class="mixbar" id="bmix" style="margin:0 0 6px"></div><div class="bs" id="bstats">—</div></div>
        <button class="cta" id="bgo" disabled onclick="showName(builderMix())">Let's go →</button>
      </div>
    </div>
    <div class="grid" id="cards"></div>
  </div>
</section>
</div>

<div id="page-intel" style="display:none">
<section style="padding-top:150px;padding-bottom:20px">
  <div class="wrap">
    <div class="rv in"><span class="corebar">◆ The core of Blackbox</span>
    <h2>Everything the AI sees</h2>
    <p class="lead">This page is the heart of the project. Blackbox runs four distinct AI systems — but the one that matters most is <b>Regime Radar</b>: five machine-learning models <b>we designed and trained ourselves</b>, from raw market data. The others read the live world and explain it. Below you'll find every system, each with a plain-English "how it works" — and further down, an in-depth account of exactly how the models were built, what data they use, and how they learn.</p></div>
    <div class="aioverview rv in">
      <div class="aiov"><div class="ati">🧠</div><h4>1 · Regime Radar</h4><p>Five neural networks we trained to forecast market turbulence.</p><span class="tag2">our own model</span></div>
      <div class="aiov"><div class="ati">📰</div><h4>2 · Sentiment scanner</h4><p>AI reads live news &amp; social mood across six asset classes.</p><span class="tag2">reads the world</span></div>
      <div class="aiov"><div class="ati">🕰️</div><h4>3 · Historical analog</h4><p>AI ties today's news to the closest real past market event.</p><span class="tag2">reads the world</span></div>
      <div class="aiov"><div class="ati">🎛️</div><h4>4 · Positioning desk</h4><p>AI explains how it's supporting each strategy right now.</p><span class="tag2">explains</span></div>
    </div>
  </div>
</section>

<section style="padding-top:20px">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">01 — Regime Radar · our own AI</div>
    <h2>Regime Radar</h2>
    <p class="lead">This is what we <em>built and trained ourselves</em> — five neural networks, one per asset class, not a borrowed language model. Each learned from ~20 years of real daily market data to forecast whether the days ahead will be <b>turbulent</b> (high volatility) or <b>calm</b>. Every other AI feature in Blackbox uses a pre-trained model; these we trained. Here are their live calls and honest, measured track records — including which market's turbulence turns out to be most predictable.</p></div>
    <details class="howto rv in"><summary>How it works</summary><div class="hbody"><ol>
      <li><b>Data.</b> ~20 years of real daily prices for each market (crypto, stocks, bonds, commodities, FX) — over 200,000 days in total.</li>
      <li><b>Features.</b> For every past day we compute 15 numbers describing recent behaviour (returns, volatility, trend, momentum) — using <b>only the past</b>, so the model can't cheat.</li>
      <li><b>Learning.</b> A neural network studies those days and <b>teaches itself</b> which patterns tend to come before turbulence. Nobody writes the rule — the model discovers it. This is the actual "AI".</li>
      <li><b>Honest test.</b> Each model is then graded on years of data it <b>never saw in training</b>. The score (AUC) runs from 0.50 (a coin-flip) to 1.00 (perfect).</li>
      <li><b>Live.</b> Today's 15 numbers go into the trained model, which outputs a turbulence probability → Calm or Turbulent.</li>
    </ol></div></details>
    <div id="regimeBox"></div>
  </div>
</section>

<section style="background:var(--surface);border-block:1px solid var(--line)">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">01 · in depth</div>
    <h2>How the models actually work</h2>
    <p class="lead">This is the long version — everything you'd want to know about the machine-learning core. What the models do, exactly which data they learn from, how they were created, and how a neural network actually teaches itself. No hand-waving.</p></div>

    <div class="longform rv in">
      <h3>1 · What these models do</h3>
      <p>Financial markets swing between <b>calm</b> stretches and <b>turbulent</b> ones — periods of large, rapid price moves. Turbulence is when investors get hurt: crashes, whipsaws, forced selling. So we built models to answer one precise question, separately for each kind of market: <em>will the next ten trading days be turbulent (unusually high volatility) or calm?</em> — using only information available today.</p>
      <p>We deliberately forecast <b>volatility</b> rather than <b>direction</b> (whether prices go up or down). Direction is very close to impossible to predict — markets are highly efficient, so a direction model is essentially a coin-flip and would make a dishonest demo. Volatility is different: it <b>clusters</b>. Turbulent days tend to be followed by more turbulent days, and calm by calm. That clustering is a real, well-documented statistical property, which means volatility is genuinely — if imperfectly — learnable. Choosing a target a model can actually learn is itself part of doing honest machine learning.</p>
      <p>The models never buy or sell anything. They output a <b>risk signal</b>: a probability of turbulence ahead, which becomes a simple Calm / Turbulent call. In the wider product this could tell a portfolio to carry less risk before a likely storm — but the trading decisions themselves always stay with the validated, rule-based strategies.</p>

      <h3>2 · The data — five real datasets</h3>
      <p>A model is only as good as what it learns from, and the challenge specifically rewards <b>working with a large dataset</b>. We assembled five, one per asset class, from real daily market history — over <b>220,000 labelled examples in total</b>. Each row is one real trading day for one real instrument, described by the features in the next section and labelled by what actually happened next.</p>
      <table class="dtable">
        <thead><tr><th>Model</th><th>What it learns from</th><th>History</th><th>Examples</th></tr></thead>
        <tbody>
          <tr><td><b>Cataclysm</b><br>crypto</td><td>13 coins — Bitcoin, Ethereum, Solana, Cardano, XRP, Dogecoin, Chainlink, Litecoin, Bitcoin&nbsp;Cash, Ethereum&nbsp;Classic, Avalanche, Polkadot, Polygon</td><td>2016–2026<br>Coinbase (free)</td><td>29,600</td></tr>
          <tr><td><b>Vantage</b><br>equities</td><td>8 global stock indices — S&amp;P&nbsp;500, Nasdaq&nbsp;100, Dow&nbsp;30, DAX&nbsp;40, FTSE&nbsp;100, Nikkei&nbsp;225, Euro&nbsp;Stoxx&nbsp;50, ASX&nbsp;200</td><td>2006–2026<br>OANDA</td><td>40,700</td></tr>
          <tr><td><b>Bastion</b><br>bonds &amp; rates</td><td>6 government-bond instruments — US 2/5/10/30-year, German 10-year (Bund), UK 10-year (Gilt)</td><td>2006–2026<br>OANDA</td><td>30,400</td></tr>
          <tr><td><b>Crucible</b><br>commodities</td><td>12 markets — gold, silver, platinum, palladium, copper, WTI oil, Brent oil, natural gas, corn, wheat, soybeans, sugar</td><td>2005–2026<br>OANDA</td><td>63,300</td></tr>
          <tr><td><b>Meridian</b><br>FX</td><td>10 currency pairs — EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY</td><td>2005–2026<br>OANDA</td><td>57,000</td></tr>
        </tbody>
      </table>
      <p>Crypto history comes from <b>Coinbase's public API</b> (free, open); the four traditional-asset classes use <b>OANDA</b>, a regulated broker whose data feed gives clean daily prices going back twenty years. We <em>pool</em> every instrument within a class — each coin, index or pair contributes its whole history as training examples — which is what multiplies a few thousand days into tens of thousands of examples and gives each model enough to learn from.</p>

      <h3>3 · What the model looks at — the 15 features</h3>
      <p>A model cannot read a price chart the way a person does. It reads <b>numbers</b>. So for every single day, we compute 15 numbers ("features") that describe how that market had been behaving <em>up to that point</em>. These are the model's senses. They fall into five families:</p>
      <ul>
        <li><b>Momentum / returns</b> (4 numbers) — how much the price changed over the last 5, 10, 20 and 60 days. Captures whether the market has been rising, falling, or flat, over roughly a week to three months.</li>
        <li><b>Recent volatility</b> (4 numbers) — how choppy the market has been over the last 5, 10, 20 and 30 days. Because volatility clusters, these are the model's single most powerful clues.</li>
        <li><b>Trend position</b> (3 numbers) — how far the price sits above or below its 20-, 50- and 200-day moving averages. A market stretched far from its long-term trend behaves differently from one sitting right on it.</li>
        <li><b>Momentum oscillator</b> (1 number, RSI) — a classic 0–100 gauge of whether a market looks "overbought" or "oversold".</li>
        <li><b>Drawdown &amp; acceleration</b> (3 numbers) — how far below its recent 30- and 60-day peak the market is, plus whether volatility is speeding up or slowing down.</li>
      </ul>
      <p>One rule governs all of them and it is the most important rule in the whole project: <b>every feature uses only information from the past.</b> The feature for a given day is computed strictly from prices up to and including that day — never after. If even a hint of the future leaked into the inputs, the model would look brilliant on paper and be worthless in reality. Getting this "no lookahead" discipline right is what separates honest financial machine learning from the kind that quietly cheats.</p>

      <h3>4 · The answer key — how we label "turbulent"</h3>
      <p>To teach a model, every example needs a correct answer to learn from. For each day we look at the <em>future</em> — the realised volatility over the following ten trading days — and mark the day <b>turbulent</b> if that future volatility lands in the top third of what that market experienced during the training era, and <b>calm</b> otherwise. Crucially, the cut-off that defines "top third" is measured <b>only on the training data</b>, then applied unchanged to later data, so the model is never tuned to the period it is tested on.</p>

      <h3>5 · How the model was built — and how it actually learns</h3>
      <p>The model at the heart of each class is a <b>neural network</b> — the exact kind of "self-learning algorithm" the challenge names. Here is what that means, concretely.</p>
      <p>A neural network is a web of simple maths units called <b>neurons</b>, arranged in <b>layers</b>. Ours has an input layer of the 15 features, two hidden layers (of 32 and then 16 neurons), and a single output that becomes the probability of turbulence. Every connection between neurons carries a <b>weight</b> — a number that says how much that input matters. A network with these layers has a few thousand of these weights.</p>
      <p>At the start, all those weights are <b>random</b> — the network knows nothing and guesses blindly. Then the learning happens, and this is the part that makes it "AI": we show it a training example (15 features → the true calm/turbulent answer), it makes a guess, we measure how wrong the guess was, and the network <b>nudges every weight a tiny amount in the direction that would have made the guess less wrong</b>. This nudging procedure is called backpropagation with gradient descent. We repeat it across all the training examples, over and over (each full pass is an "epoch"), and with every pass the network gets a little better at mapping "what the market looks like today" to "turbulent or calm ahead". After enough passes it stops improving, and we stop training automatically (this is "early stopping", which prevents it from simply memorising). At no point does a human write the rule — the network <b>discovers</b> the relationship in the data by itself.</p>
      <p>A few practical touches make it work honestly and well: we <b>standardise</b> the features (rescale them to a common range) so no single one dominates just because its numbers are bigger; we <b>balance the classes</b> so the model doesn't lazily learn to always say "calm" just because calm days are more common; and we tune the exact probability cut-off (how sure it must be before it says "turbulent") on a separate validation set, never on the final test. As a cross-check we also train a <b>random forest</b> — a completely different model made of hundreds of decision trees that vote — and keep whichever of the two does better on validation. For four of the five classes the neural network wins or ties, so that is what we deploy.</p>

      <h3>6 · Testing it honestly</h3>
      <p>The single easiest way to fool yourself in machine learning is to test a model on data it already studied. So we split each dataset <b>by time, never randomly</b>: the model trains on the oldest ~60% of history, we tune choices on the next ~15% (validation), and we grade it once on the most recent ~25% — <b>data it never saw during training</b>. Splitting by time matters because shuffling the days would let tomorrow leak into yesterday.</p>
      <p>We grade with a metric called <b>AUC</b>, which runs from 0.50 (a coin-flip — no skill at all) to 1.00 (perfect). AUC is the right choice here because it isn't fooled by the fact that calm days outnumber turbulent ones. We always compare against two deliberately dumb baselines: "always guess calm", and the strong one — "volatility is already high, so expect more" (persistence). Beating persistence is the real test, because volatility clustering makes it a genuinely tough benchmark.</p>

      <h3>7 · The results — and a real finding</h3>
      <p>On data none of the models had ever seen, every one beats both baselines. Bond and rate turbulence turned out the most forecastable (<b>AUC 0.93</b>), then commodities (0.85), equities (0.82) and FX (0.81); crypto — the wildest, most headline-driven market — is the hardest (0.70). The live scorecard above shows all of this, including the <b>edge</b> column, which is the honest measure of what our machine learning adds <em>beyond</em> simply noticing that volatility is already high.</p>
      <p>Running one identical method across five markets produced a genuine cross-asset insight: <b>the steadier the market, the more predictable its turbulence.</b> That makes economic sense — bond volatility is driven by slow, telegraphed forces like the interest-rate cycle, whereas crypto turbulence is often set off by sudden, unforecastable news. Discovering that by experiment, rather than assuming it, is exactly what a good study should do.</p>

      <h3>8 · How it runs live in this app</h3>
      <p>Once trained, each model is saved to disk as a small file — its learned "brain". When you open this page, the app takes the most recent real prices for a representative market in each class (Bitcoin, the S&amp;P 500, the US 10-year, gold, EUR/USD), computes today's 15 features, feeds them through the trained network, and reads out a fresh turbulence probability. That is the Calm / Turbulent call you see at the top of the page, updated from live data.</p>

      <h3>9 · Honesty &amp; limitations</h3>
      <p>We say plainly what these models are not. They are a <b>modest, real edge — not a crystal ball</b>. They miss sudden shocks that have no footprint in past prices, they raise occasional false alarms, and the high headline AUCs owe a lot to volatility simply being persistent (the model's own contribution is the smaller "edge"). They forecast how <em>bumpy</em> the ride will be, not which way prices go, and they inform risk rather than making a single trade. Every one of those caveats is stated because a model you can honestly critique is worth far more than an impressive number you cannot defend — which, in the end, is the whole point of the project.</p>
    </div>
  </div>
</section>

<section id="intel">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">02 — Sentiment scanner</div>
    <h2>The scanner reads the market's mood</h2>
    <p class="lead">An AI engine scans hundreds of news and social sources, scores each asset class, explains itself — and trims exposure in your Blackbox when it turns strongly bearish on a strategy's assets.</p></div>
    <details class="howto rv in"><summary>How it works</summary><div class="hbody"><ol>
      <li><b>Fetch.</b> It pulls hundreds of real, current items — news headlines (Google News) and social posts (Reddit) — across six asset classes: equities, gold, rates, FX, crypto, oil.</li>
      <li><b>Read.</b> An AI model scores each headline's mood for that asset, from −100 (very bearish) to +100 (very bullish). It reads for real meaning — handling context and negation — not just spotting keywords.</li>
      <li><b>Aggregate.</b> The scores combine into one mood reading per asset, with a plain-English explanation you can trace back to the actual headlines that drove it.</li>
      <li><b>Act.</b> This is where AI <b>supports the strategies</b>: if a strategy's assets score strongly bearish (below −40), that strategy's exposure is automatically trimmed to 75% — a risk brake on top of the validated rule.</li>
    </ol></div></details>
    <div class="sgrid" id="sgrid"></div>
  </div>
</section>

<section id="analog">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">03 — Historical analog</div>
    <h2>Where has the market seen this before?</h2>
    <p class="lead">AI reads today's real headlines and matches them to the closest event from a fixed library of real past market events — then shows how each strategy's real, backtested returns actually moved in the months after. AI only picks which past event fits; every number comes from the same 20-year backtests as the rest of the app.</p></div>
    <details class="howto rv in"><summary>How it works</summary><div class="hbody"><ol>
      <li><b>Read today.</b> It takes the top real headlines the scanner just collected.</li>
      <li><b>Match.</b> An AI model compares them to a fixed library of real past market events (the 2008 crash, the Taper Tantrum, COVID, the 2022 rate-hike cycle, and more) and picks the single closest analog.</li>
      <li><b>Look up real history.</b> The app then pulls how each strategy's <b>real, backtested</b> returns actually moved in the months after that past event.</li>
      <li><b>Honest split.</b> AI only chooses <em>which</em> past event fits — it never invents an outcome. Every number shown comes from the real 20-year backtests.</li>
    </ol></div></details>
    <div id="analogBox"></div>
  </div>
</section>

<section id="positioning" style="background:var(--surface);border-block:1px solid var(--line)">
  <div class="wrap">
    <div class="rv in"><div class="sec-k">04 — AI positioning desk</div>
    <h2>How AI is supporting each strategy right now</h2>
    <p class="lead">Each strategy's position is decided by its validated rule. On top of that, the AI reads the live news sentiment for that strategy's assets and acts as a risk brake — trimming exposure to 75% when the mood turns strongly bearish. Below, AI explains each strategy's current stance and how the overlay is supporting it. The rule is the driver; the AI is the co-pilot.</p></div>
    <details class="howto rv in"><summary>How it works</summary><div class="hbody"><ol>
      <li><b>Read the rule.</b> For each of the eight strategies, it takes today's rule-based stance — what the strategy holds right now, and why (e.g. "gold uptrend + falling yields → fully long gold").</li>
      <li><b>Read the mood.</b> It pulls the live sentiment score (from system 2) for that strategy's assets.</li>
      <li><b>Check the brake.</b> It checks whether the sentiment risk-overlay is currently trimming that strategy's exposure.</li>
      <li><b>Explain.</b> An AI model writes a short, plain-English note on the strategy's current position and how the AI overlay is supporting or easing off it. The validated rule always makes the call; the AI narrates and, within strict limits, brakes.</li>
    </ol></div></details>
    <div id="posGrid"></div>
  </div>
</section>
</div>

<div id="page-briefing" style="display:none">
<section style="padding-top:150px">
  <div class="wrap" style="max-width:900px">
    <div class="rv in"><div class="sec-k">Daily briefing</div>
    <h2>Today's market, read by AI</h2>
    <p class="lead">Every cycle, the AI scanner reads hundreds of real news and social sources and scores six asset classes. This briefing synthesises that live read — plus the closest historical analog — into a plain-English summary of the day. It describes the mood and what it means for a diversified systematic portfolio; it is not advice, and it never invents a number.</p></div>
    <div id="briefingScores" class="briefrow"></div>
    <div id="briefingBox" class="acard" style="margin-top:22px"></div>
    <div class="note" style="margin-top:14px">Regenerated periodically from the live sentiment scan. Educational synthesis only — every strategy still trades on its validated rules.</div>
  </div>
</section>
</div>

<div id="page-learn" style="display:none">
<section style="padding-top:150px">
  <div class="wrap" style="max-width:900px">
    <div class="rv in"><div class="sec-k">Learn</div>
    <h2>Understand what you're investing in</h2>
    <p class="lead">Systematic investing has its own vocabulary. Tap any term below for a plain-English explanation from the AI, or ask the tutor anything — how a strategy works, what a metric means, why diversification lowers risk. It's grounded in this app's own real strategies, and it never gives buy/sell advice.</p></div>

    <h3 style="margin-top:34px;font-size:20px">Key terms — tap to explain</h3>
    <div class="chips" id="glossChips" style="margin-top:14px"></div>
    <div id="glossBox" class="acard" style="margin-top:18px;display:none"></div>

    <h3 style="margin-top:40px;font-size:20px">Ask the AI tutor</h3>
    <div class="acard" style="margin-top:14px">
      <div id="coachLog"></div>
      <div class="qform"><input class="ain" id="coachInput" placeholder="e.g. Why does gold do well when yields fall?" onkeydown="if(event.key==='Enter')askCoach()">
        <button class="cta ghost small" onclick="askCoach()">Ask</button></div>
    </div>
  </div>
</section>
</div>

<div id="page-ai" style="display:none">
<section style="padding-top:150px">
  <div class="wrap" style="max-width:940px">
    <div class="rv in"><div class="sec-k">How it works</div>
    <h2>The AI inside Blackbox</h2>
    <p class="lead">Blackbox is built on eight strategies validated over 20 years of real market data. Artificial intelligence is layered through the whole product — but in a very deliberate way. This page explains exactly what the AI does, how it's implemented, and, just as importantly, what it does <em>not</em> do.</p></div>

    <div class="principle rv in">
      <div class="pbig">The rule is the driver. The AI is the co-pilot.</div>
      <p>Every buy, sell and hold decision comes from a validated, backtested rule — not from AI. The AI reads the market, supports those rules with a bounded risk brake, personalises your mix within strict limits, and explains everything in plain language. It never trades on its own, and it never invents a number.</p>
    </div>

    <h3 class="airole-h">0 · The AI we built ourselves — Regime Radar</h3>
    <div class="capgrid">
      <div class="capcard rv in" style="grid-column:1/-1;border-color:var(--acc)"><div class="capi">🧠</div><h4>Five neural networks we trained from scratch</h4>
        <p>There is one crucial distinction on this page. Every feature below <em>uses</em> a large language model that someone else pre-trained. <b>Regime Radar is different — these are models we built and trained ourselves.</b> We took ~20 years of real daily market data across five asset classes (crypto, equities, bonds, commodities, FX — over 200,000 examples in total), engineered the input features ourselves, defined the target (will the next 10 days be turbulent or calm?), and trained a separate <b>neural network</b> for each market to learn the pattern — no human wrote the rule, the models discovered it from the data. We then tested them honestly on years of data they never saw during training: they score <b>AUC from 0.70 (crypto) up to 0.93 (bonds)</b> — genuine, if imperfect, skill that beats the naive baselines everywhere (a coin-flip is 0.50). This is the machine-learning core of the project — see all five live on the <a onclick="go('intel')" style="color:var(--acc);cursor:pointer">Intelligence</a> page, with the honest results, cross-asset comparison and limitations in our written reflection.</p></div>
    </div>

    <h3 class="airole-h">1 · The AI reads the market</h3>
    <div class="capgrid">
      <div class="capcard rv in"><div class="capi">📰</div><h4>Live sentiment scanner</h4>
        <p>Every cycle, an AI model reads hundreds of real news headlines and social posts across six asset classes — equities, gold, rates, FX, crypto, oil — and scores the mood from −100 to +100, with a plain-English explanation traceable to the actual headlines. It reads for real meaning (handling context and negation), not just keywords.</p></div>
      <div class="capcard rv in"><div class="capi">🕰️</div><h4>Historical analog</h4>
        <p>AI matches today's real headlines to the closest event from a fixed library of real past market events — the 2008 crash, the Taper Tantrum, COVID, and more — then shows how each strategy's <em>real, backtested</em> returns actually moved in the months after. AI only picks which past event fits; every number is from the backtests.</p></div>
    </div>

    <h3 class="airole-h">2 · The AI supports the strategies</h3>
    <div class="capgrid">
      <div class="capcard rv in"><div class="capi">🛡️</div><h4>Sentiment risk overlay</h4>
        <p>This is where AI directly supports each strategy. When the scanner reads a strategy's assets as strongly bearish (a score below −40), that strategy's exposure is automatically trimmed to 75% — a risk brake layered on top of the validated rule. The rule decides the position; the AI decides whether to ease off it.</p></div>
      <div class="capcard rv in"><div class="capi">🎛️</div><h4>AI positioning desk</h4>
        <p>For every strategy, the AI reads its live rule-based stance and the live sentiment on its assets, then writes a short note explaining where it's positioned right now and whether the risk overlay is supporting or trimming it. You can see this on the Intelligence page.</p></div>
    </div>

    <h3 class="airole-h">3 · The AI personalises — within limits</h3>
    <div class="capgrid">
      <div class="capcard rv in"><div class="capi">✍️</div><h4>Free-text quiz nudge</h4>
        <p>After the six questions, you can describe your situation in your own words. AI reads it and may apply a <em>small, bounded</em> adjustment to your mix — never more than a gentle tilt, never adding or removing a strategy, never overriding the validated math. And it always explains, in writing, what in your words drove the change.</p></div>
      <div class="capcard rv in"><div class="capi">🧭</div><h4>AI risk review</h4>
        <p>Once your portfolio is live, AI reviews your actual mix — how diversified or concentrated it is, and its blend of return and risk — reading the exact figures for you. The blended numbers are computed precisely in code; the AI only puts them into words.</p></div>
    </div>

    <h3 class="airole-h">4 · The AI explains &amp; teaches</h3>
    <div class="capgrid">
      <div class="capcard rv in"><div class="capi">📊</div><h4>Daily briefing</h4>
        <p>A plain-English summary of the day's market, synthesised from the live sentiment scan and the historical analog.</p></div>
      <div class="capcard rv in"><div class="capi">💬</div><h4>Portfolio narration &amp; Q&amp;A</h4>
        <p>Your Blackbox writes its own summary, and you can ask it questions — answered only from your real portfolio data.</p></div>
      <div class="capcard rv in"><div class="capi">🎓</div><h4>Tutor &amp; glossary</h4>
        <p>An AI tutor answers investing questions grounded in this app's own strategies, and any metric can be explained on tap.</p></div>
      <div class="capcard rv in"><div class="capi">⚖️</div><h4>Strategy deep-dives &amp; compare</h4>
        <p>AI explains why any strategy works, and talks you through the trade-offs when you compare two — always from the real numbers.</p></div>
    </div>

    <div class="honesty rv in">
      <h3>What the AI does <span style="color:var(--red)">not</span> do</h3>
      <ul>
        <li><b>It does not make trades on its own.</b> Every buy/sell/hold comes from validated, backtested rules — moving-average trends, momentum, a rates gate on gold, seasonality. AI never originates a trade.</li>
        <li><b>It does not "learn" from your data or rewrite the strategies.</b> The language model was pre-trained on a vast amount of text — that's what lets it read and explain — but it does not train itself on this app, and the strategies come from 20-year statistical validation, not from AI. (So when we say "AI reads the news," we mean it applies what it already learned about language — not that it's inventing new trading rules.)</li>
        <li><b>It never invents a performance number.</b> Every figure traces to real backtests in the catalog. Where a blended number appears, it's calculated exactly in code and the AI is only allowed to state it verbatim.</li>
        <li><b>It does not give personalised buy/sell advice.</b> It explains and contextualises; it never tells you what to buy or sell. This is practice money, and none of it is investment advice.</li>
      </ul>
    </div>

    <div class="aitech rv in">
      <h4>Under the hood</h4>
      <p>All of the above runs through a single shared connector to a large language model (Llama, served by Groq), behind one API key. Every AI feature "fails open": if the model is ever unavailable, the app keeps working and the feature simply says so — the validated strategies and every real number are never affected. That's the whole design: real, tested rules at the core; AI as the analyst, risk co-pilot, and teacher around them.</p>
    </div>
  </div>
</section>
</div>

<div id="page-compare" style="display:none">
<section style="padding-top:150px">
  <div class="wrap" style="max-width:1000px">
    <div class="rv in"><div class="sec-k">Compare</div>
    <h2>Put two strategies side by side</h2>
    <p class="lead">Pick any two of the eight strategies to compare their real 20-year statistics head to head — return, risk, drawdown and how independent each is from the stock market — then let the AI talk you through the trade-offs. Every number is from the same backtests as the rest of the app.</p></div>
    <div class="cmppick">
      <div><label class="cmplabel">Strategy A</label><select class="ain cmpsel" id="cmpA"></select></div>
      <div class="cmpvs">vs</div>
      <div><label class="cmplabel">Strategy B</label><select class="ain cmpsel" id="cmpB"></select></div>
    </div>
    <div id="cmpTable" style="margin-top:24px"></div>
    <div id="cmpBox" class="acard" style="margin-top:20px;display:none"></div>
  </div>
</section>
</div>

<div id="page-account" style="display:none">
<section id="me" style="padding-top:150px">
  <div class="wrap">
    <div id="mebody"><div class="skel"><span class="sp"></span>Syncing your Blackbox with live markets…</div></div>
  </div>
</section>
</div>

<div class="mask" id="mask" onclick="if(event.target===this)closeModal()"><div class="modal" id="modal"></div></div>

<div id="mpop" class="mpop" style="display:none"></div>

<div class="mask" id="amask" onclick="if(event.target===this)closeAuth()">
  <div class="modal amodal">
    <button class="x" onclick="closeAuth()">×</button>
    <h2 style="font-size:26px;margin-bottom:6px">Your Blackbox account</h2>
    <p style="color:var(--mut);font-size:14px">Log back in from any device. Practice capital only — no real money.</p>
    <div id="gwrap" style="margin:20px 0 0;display:none"><div id="gbtn" style="display:flex;justify-content:center"></div>
      <div class="ordiv"><span>or</span></div></div>
    <div class="aview" id="av-choice" style="margin-top:18px">
      <button class="cta" style="width:100%" onclick="showAV('create')">Create my account</button>
      <button class="cta ghost" style="width:100%;margin-top:10px" onclick="showAV('key')">I have a recovery key</button>
    </div>
    <div class="aview" id="av-create" style="display:none;margin-top:18px">
      <input class="ain" id="regname" placeholder="Your name" maxlength="24">
      <button class="cta" style="width:100%;margin-top:12px" onclick="doRegister()">Create account →</button>
      <div class="aerr" id="regerr"></div>
    </div>
    <div class="aview" id="av-keyshow" style="display:none;margin-top:14px">
      <p style="font-size:14px;color:var(--mut)">This key <b style="color:var(--txt)">is your login</b>. Save it now — it's shown only once.</p>
      <div class="keybox" id="keybox"></div>
      <button class="cta ghost" style="width:100%;margin-top:14px" id="copybtn" onclick="copyKey()">Copy key</button>
      <button class="cta" style="width:100%;margin-top:10px" onclick="closeAuth();afterAuth()">I saved it — continue →</button>
    </div>
    <div class="aview" id="av-key" style="display:none;margin-top:18px">
      <input class="ain" id="keyin" placeholder="e.g. violet-falcon-ember-4821">
      <button class="cta" style="width:100%;margin-top:12px" onclick="doRecover()">Log in →</button>
      <div class="aerr" id="keyerr"></div>
    </div>
  </div>
</div>

<footer><div class="wrap">
  <div style="font-family:'Fraunces';font-style:italic;font-weight:600;font-size:19px;color:var(--txt);margin-bottom:12px">BLACKBOX</div>
  <div class="d" id="disc"></div>
</div></footer>

<script>
const $=s=>document.querySelector(s);
let CAT=null,SENT=null,MODE=null,PENDMIX=null,curveChart=null,projChart=null,navChart=null,ME=null,GCID=null;
const COLORS=["#C9A961","#7A9B76","#B5716B","#8B98A8","#C08552","#9C8AA5","#6B8F87","#B08D4F"];
const BUCKETS={stack_bal:["equities","gold","oil"],stack_agg:["equities","gold","oil"],equity_season:["equities"],
gold_macro:["gold","rates"],commod_trend:["oil","gold"],risk_parity:["equities","rates","fx"],
crypto_trend:["crypto"],spx_hold:["equities"]};
const uid=()=>localStorage.getItem('bb_uid');

(function(){const words=[['Your'],['own'],['Blackbox.'],['Live','grad'],['in'],['minutes.','grad']];
$('#hl').innerHTML=words.map((w,i)=>`<span class="w ${w[1]||''}" style="animation-delay:${i*90}ms">${w[0]}&nbsp;</span>`).join('')})();

const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}}),{rootMargin:'-80px'});
document.querySelectorAll('.rv').forEach(el=>io.observe(el));
document.addEventListener('click',e=>{const g=e.target.closest('.chips');if(!g)return;const b=e.target.closest('.chip');if(!b)return;
g.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));b.classList.add('on')});

/* ---------- pages/router ---------- */
const PAGES=['home','start','strategies','briefing','intel','learn','ai','compare','account'];
function pageFromPath(){const p=(location.pathname||'/').replace(/^\//,'');return PAGES.includes(p)?p:'home'}
function go(page,push){
  if(!PAGES.includes(page))page='home';
  PAGES.forEach(p=>{const el=document.getElementById('page-'+p);if(el)el.style.display=(p===page?'':'none')});
  document.querySelectorAll('.navlinks a').forEach(a=>a.classList.toggle('on',a.dataset.page===page));
  if(push!==false)history.pushState({page},'',page==='home'?'/':'/'+page);
  window.scrollTo(0,0);
  // reveal-on-scroll can't fire for elements that were display:none when observed,
  // so force-reveal this page's content when it's shown.
  const cur=document.getElementById('page-'+page);
  if(cur)cur.querySelectorAll('.rv').forEach(el=>el.classList.add('in'));
  onPageShown(page);
}
window.addEventListener('popstate',e=>go((e.state&&e.state.page)||pageFromPath(),false));
function onPageShown(page){
  if(page==='strategies'){ensureStrategiesPage();}
  if(page==='briefing'){loadBriefing();}
  if(page==='intel'){loadRegime();renderSentiment();loadAnalog();loadPositioning();}
  if(page==='learn'){ensureLearnPage();}
  if(page==='compare'){ensureComparePage();}
  if(page==='account'){ensureAccountPage();}
}

function heroGo(){ if((ME&&ME.signed_in&&ME.portfolio)||uid()){go('account')} else {go('start')} }

/* ---------- accounts ---------- */
function renderAuth(){
  const b=$('#authbox');
  if(ME&&ME.signed_in){
    b.innerHTML=`<div class="uchip">${ME.picture?`<img src="${ME.picture}" referrerpolicy="no-referrer" alt="">`:`<span class="uinit">${(ME.name||'?')[0].toUpperCase()}</span>`}
      <span class="un">${ME.name}</span>
      <button class="cta ghost small" onclick="doLogout()">Sign out</button></div>`;
  } else b.innerHTML='<button class="cta ghost small" onclick="openAuth()">Sign in</button>';
}
function openAuth(){$('#amask').classList.add('show');showAV('choice')}
function closeAuth(){$('#amask').classList.remove('show')}
function showAV(v){document.querySelectorAll('.aview').forEach(x=>x.style.display='none');
  const el=document.getElementById('av-'+v);if(el)el.style.display='block'}
async function refreshMe(){ME=await fetch('/api/me').then(r=>r.json()).catch(()=>({signed_in:false}));
  renderAuth();
  $('#heroCta').textContent=((ME.signed_in&&ME.portfolio)||uid())?'View my Blackbox →':"Let's go →";}
async function afterAuth(){await refreshMe();
  if(_pendStart&&PENDMIX&&ME.signed_in){_pendStart=false;startBlackbox();return;}
  if(ME.signed_in&&ME.portfolio){go('account')}
  else{go('start')}}
async function doRegister(){
  const name=$('#regname').value.trim()||'Trader';
  const r=await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,adopt_uid:uid()})}).then(r=>r.json()).catch(()=>null);
  if(!r||!r.ok){$('#regerr').textContent='Could not create the account — try again.';return}
  localStorage.removeItem('bb_uid');
  window._key=r.recovery_key;$('#keybox').textContent=r.recovery_key;
  ME={signed_in:true,name:r.name,portfolio:null};renderAuth();showAV('keyshow');
}
function copyKey(){navigator.clipboard.writeText(window._key||'').then(()=>{$('#copybtn').textContent='Copied ✓'}).catch(()=>{})}
async function doRecover(){
  const r=await fetch('/api/auth/recover',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:$('#keyin').value})}).then(r=>r.ok?r.json():null).catch(()=>null);
  if(!r||!r.ok){$('#keyerr').textContent='Key not recognised — check the spelling.';return}
  closeAuth();afterAuth();
}
async function doLogout(){await fetch('/api/auth/logout',{method:'POST'}).catch(()=>{});
  ME={signed_in:false};renderAuth();
  $('#mebody').innerHTML='<div class="skel"><span class="sp"></span>Syncing your Blackbox with live markets…</div>';
  $('#heroCta').textContent=uid()?'View my Blackbox →':"Let's go →";go('home')}
function onGoogleCred(resp){
  fetch('/api/auth/google',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({credential:resp.credential,adopt_uid:uid()})}).then(r=>r.json()).then(r=>{
      if(r&&r.ok){localStorage.removeItem('bb_uid');closeAuth();afterAuth();}
      else openAuth()}).catch(()=>{});
}
function initGIS(tries){
  if(!GCID)return;
  if(window.google&&google.accounts&&google.accounts.id){
    $('#gwrap').style.display='block';
    google.accounts.id.initialize({client_id:GCID,callback:onGoogleCred});
    google.accounts.id.renderButton($('#gbtn'),{theme:'filled_black',size:'large',shape:'pill',width:330});
  } else if((tries||0)<24) setTimeout(()=>initGIS((tries||0)+1),500);
}
function applyStepperVisibility(){
  document.querySelectorAll('.stepper').forEach(s=>s.style.display=MODE==='manual'?'flex':'none');
  const bw=document.getElementById('builderwrap');if(bw)bw.classList.toggle('show',MODE==='manual');
}
function setMode(m){MODE=m;
  $('#modeQuiz').classList.toggle('on',m==='quiz');$('#modeManual').classList.toggle('on',m==='manual');
  $('#quiz').classList.toggle('show',m==='quiz');
  $('#namecard').style.display='none';
  applyStepperVisibility();
  if(m==='manual')go('strategies');
}
let cardsLoaded=false;
function ensureStrategiesPage(){
  if(!cardsLoaded){cardsLoaded=true;renderCards();}
  applyStepperVisibility();
}
function fmt(v){return v.toLocaleString(undefined,{maximumFractionDigits:0})}
function fmt2(v){return v.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}

async function boot(){
  const [c,s,cfg,me]=await Promise.all([
    fetch('/api/catalog').then(r=>r.json()),
    fetch('/api/sentiment').then(r=>r.json()).catch(()=>null),
    fetch('/api/config').then(r=>r.json()).catch(()=>({})),
    fetch('/api/me').then(r=>r.json()).catch(()=>({signed_in:false}))]);
  CAT=c;SENT=s;GCID=cfg&&cfg.google_client_id;ME=me;
  $('#hs1').textContent=c.strategies.length;
  $('#hs2').textContent=s&&s.n_scanned?s.n_scanned:'600+';
  $('#disc').textContent=c.disclaimer||'';
  renderAuth();initGIS();
  const have=(ME.signed_in&&ME.portfolio)||uid();
  if(have)$('#heroCta').textContent='View my Blackbox →';
  go(pageFromPath(),false);
}

async function ensureAccountPage(){
  // Re-fetch fresh state — a portfolio may have just been created this session,
  // so we must not trust a stale local copy of ME.
  if(ME&&ME.signed_in){
    const m=await fetch('/api/me').then(r=>r.json()).catch(()=>null);
    if(m&&m.signed_in){ME=m;renderAuth();}
    if(ME&&ME.portfolio){renderMe(ME.portfolio);return;}
  } else if(uid()){
    const r=await fetch('/api/user/'+uid()).then(r=>r.ok?r.json():null).catch(()=>null);
    if(r){renderMe(r);return;}
    localStorage.removeItem('bb_uid');
  }
  $('#mebody').innerHTML=`<div class="skel" style="text-align:left;padding:60px 0">
    <h2 style="font-size:28px;margin-bottom:10px">You don't have a Blackbox yet</h2>
    <p style="color:var(--mut);margin-bottom:22px">Answer six questions or build your own mix to start a personal paper portfolio.</p>
    <button class="cta" onclick="go('start')">Start my Blackbox →</button>
  </div>`;
}

/* ---------- historical analog ---------- */
async function loadAnalog(){
  const box=$('#analogBox');
  box.innerHTML='<div class="askel"><span class="sp"></span>Reading today\'s headlines against market history…</div>';
  const a=await fetch('/api/historical-analog').then(r=>r.json()).catch(()=>null);
  if(!a||a.available===false||!a.event){
    box.innerHTML='<div class="askel">Historical-analog reading isn\'t available right now — it needs an AI key configured on the server.</div>';
    return;
  }
  const e=a.event,rx=a.reactions||{};
  const rows=Object.values(rx).sort((x,y)=>y.pct-x.pct).map(r=>
    `<div class="arow"><span>${r.name}</span><span style="color:${r.pct>0?'var(--green)':r.pct<0?'var(--red)':'var(--mut)'}">${r.pct>0?'+':''}${r.pct}%</span></div>`).join('');
  box.innerHTML=`<div class="acard rv in">
    <div class="ah"><h3>${e.title}</h3><span class="adate">${e.date}</span></div>
    <p style="color:var(--mut);font-size:14px;margin-top:8px">${e.summary}</p>
    <div class="areason">${a.reason||''}</div>
    <p style="font-size:12.5px;color:var(--mut);margin-bottom:10px">What each strategy actually returned in the ${a.horizon_days||90} days after ${e.date}:</p>
    <div class="agrid">${rows}</div>
  </div>`;
}

function sentFor(sid){
  if(!SENT||!SENT.assets)return null;
  const bs=BUCKETS[sid]||[];const vals=bs.map(b=>SENT.assets[b]).filter(Boolean);
  if(!vals.length)return null;
  const avg=vals.reduce((a,v)=>a+v.score,0)/vals.length;
  return {score:Math.round(avg),buckets:bs};
}
function sentChip(sid){
  const s=sentFor(sid);if(!s)return'';
  const col=s.score>12?'var(--green)':s.score<-12?'var(--red)':'var(--mut)';
  const word=s.score>12?'bullish':s.score<-12?'bearish':'neutral';
  return `<div class="sentchip" title="AI scanner read on ${s.buckets.join(', ')}"><span class="sd" style="background:${col};box-shadow:0 0 8px ${col}"></span>
    <span style="color:${col}">AI: ${word} ${s.score>0?'+':''}${s.score}</span></div>`;
}

function renderCards(){
  const el=$('#cards');
  el.innerHTML=CAT.strategies.map((s,i)=>{const st=s.stats;
    return `<div class="card rv" style="transition-delay:${(i%3)*60}ms" data-sid="${s.id}">
      <div class="top" onclick="openModal('${s.id}')"><h3>${s.name}</h3><span class="badge ${s.risk.replace(' ','')}">${s.risk}</span></div>
      <div onclick="openModal('${s.id}')">${sentChip(s.id)}
      <div class="tag">${s.tagline}</div>
      <canvas class="spark" id="sp_${s.id}"></canvas>
      <div class="stats">
        <div class="st"><div class="n">${st.cagr}%</div><div class="l">CAGR</div></div>
        <div class="st"><div class="n">${st.sharpe}</div><div class="l">Sharpe</div></div>
        <div class="st"><div class="n" style="color:var(--red)">${st.max_dd}%</div><div class="l">Max DD</div></div>
        <div class="st"><div class="n">${st.beta==null?'—':st.beta}</div><div class="l">Beta</div></div>
      </div></div>
      <div class="stepper step" style="display:none" data-sid="${s.id}">
        <button onclick="bump('${s.id}',-10)">−</button><span class="v" id="w_${s.id}">0%</span><button onclick="bump('${s.id}',10)">+</button>
        <span style="font-size:12px;color:var(--mut)">weight</span></div>
    </div>`}).join('');
  el.querySelectorAll('.rv').forEach(x=>io.observe(x));
  CAT.strategies.forEach(s=>drawSpark(document.getElementById('sp_'+s.id),s.curve));
}
function drawSpark(cv,curve){if(!cv||!curve||curve.length<2)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth||300,H=56;cv.width=W*dpr;cv.height=H*dpr;
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const vs=curve.map(p=>p.v),mn=Math.min(...vs),mx=Math.max(...vs),r=(mx-mn)||1;
  const X=i=>i/(vs.length-1)*W,Y=v=>H-4-((v-mn)/r)*(H-8);
  const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,'rgba(201,168,106,.35)');g.addColorStop(1,'rgba(201,168,106,0)');
  x.beginPath();x.moveTo(0,H);vs.forEach((v,i)=>x.lineTo(X(i),Y(v)));x.lineTo(W,H);x.closePath();x.fillStyle=g;x.fill();
  x.beginPath();vs.forEach((v,i)=>{i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v))});
  x.strokeStyle='#C9A961';x.lineWidth=2;x.lineJoin='round';x.stroke()}

/* ---------- builder ---------- */
const weights={};
function bump(sid,d){weights[sid]=Math.max(0,Math.min(100,(weights[sid]||0)+d));
  document.getElementById('w_'+sid).textContent=(weights[sid]||0)+'%';refreshBar()}
function builderMix(){const m={};Object.entries(weights).forEach(([k,v])=>{if(v>0)m[k]=v});return m}
function refreshBar(){
  const m=builderMix();const tot=Object.values(m).reduce((a,b)=>a+b,0);
  $('#btot').textContent=tot+'%';$('#bgo').disabled=tot!==100;
  const ids=Object.keys(m);
  $('#bmix').innerHTML=ids.map((id,i)=>`<i style="width:${m[id]}%;background:${COLORS[i%COLORS.length]}"></i>`).join('');
  if(ids.length&&CAT){let c=0,v=0,d=0;ids.forEach(id=>{const s=CAT.strategies.find(z=>z.id===id);if(s){c+=m[id]/100*s.stats.cagr;v+=m[id]/100*s.stats.vol;d+=m[id]/100*s.stats.max_dd}});
    $('#bstats').textContent=`est. CAGR ${c.toFixed(1)}% · vol ${v.toFixed(1)}% · weighted max DD ${d.toFixed(1)}%`}
  else $('#bstats').textContent='—';
}

/* ---------- quiz ---------- */
async function recommend(){
  const p={};document.querySelectorAll('#quiz .chips').forEach(g=>{const on=g.querySelector('.on');if(on)p[g.dataset.k]=on.dataset.v});
  const ft=($('#freetext').value||'').trim();if(ft)p.freetext=ft;
  const btn=document.querySelector('#quiz .cta');const oldTxt=btn.textContent;
  btn.disabled=true;btn.textContent=ft?'Reading your answer…':'Building…';
  const r=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}).then(r=>r.json());
  btn.disabled=false;btn.textContent=oldTxt;
  const el=$('#reco');
  el.innerHTML=`<h3 style="font-size:24px">Your custom mix: ${r.label}</h3>
    <div style="margin:10px 0 4px;font-size:13.5px;color:var(--mut)">Risk capacity
      <b style="color:var(--txt)">${r.capacity}/100</b> — built from your six answers</div>
    <div class="gauge" style="background:linear-gradient(90deg,#5FBF8F,#DCBB78,#E2716B);max-width:340px"><i style="left:calc(${r.capacity}% - 2px)"></i></div>
    <ul class="why">${r.why.map(w=>`<li>${w}</li>`).join('')}</ul>
    <div class="blend">
      <div class="b"><div class="n">${r.blend.cagr_est}%</div><div class="l">est. CAGR (backtest)</div></div>
      <div class="b"><div class="n">${r.blend.vol_est}%</div><div class="l">est. volatility</div></div>
      <div class="b"><div class="n" style="color:var(--red)">${r.blend.dd_est}%</div><div class="l">weighted max drawdown</div></div>
    </div>
    <div class="mixbar">${r.parts.map((x,i)=>`<i style="width:${x.weight}%;background:${COLORS[i%COLORS.length]}"></i>`).join('')}</div>
    <div class="mixlegend">${r.parts.map((x,i)=>`<span><span class="dotc" style="background:${COLORS[i%COLORS.length]}"></span><b>${x.weight}%</b> ${x.name}</span>`).join('')}</div>
    <button class="cta" style="margin-top:22px" onclick='showName(${JSON.stringify(r.mix)})'>Use this mix →</button>`;
  el.classList.add('show');setTimeout(()=>el.scrollIntoView({behavior:'smooth',block:'nearest'}),80);
}

function showName(mix){PENDMIX=mix;go('start');
  setTimeout(()=>{const nc=$('#namecard');nc.style.display='block';
    nc.scrollIntoView({behavior:'smooth',block:'center'});$('#uname').focus()},80)}
let _pendStart=false;
async function startBlackbox(){
  if(!PENDMIX)return;
  // A Blackbox is tied to an account — sign in first, then we resume automatically.
  if(!(ME&&ME.signed_in)){_pendStart=true;openAuth();return;}
  const btn=$('#startBtn');if(btn){btn.disabled=true;btn.textContent='Starting…';}
  const name=($('#uname').value||'My').trim();
  const r=await fetch('/api/user/create',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,mix:PENDMIX})}).then(r=>r.json()).catch(()=>null);
  if(btn){btn.disabled=false;btn.textContent='🚀 Start my Blackbox';}
  if(r&&r.need_auth){_pendStart=true;openAuth();return;}
  if(!r||!r.user_id){alert('Could not start — check the mix totals 100%.');return}
  localStorage.removeItem('bb_uid');
  $('#heroCta').textContent='View my Blackbox →';
  $('#mebody').innerHTML='<div class="skel"><span class="sp"></span>Syncing your Blackbox with live markets… first sync takes ~20 seconds.</div>';
  go('account');
}
async function resetMe(){if(!confirm('Delete this Blackbox and start over?'))return;
  const puid=window._meUid||uid();
  if(puid)await fetch('/api/user/'+puid+'/delete',{method:'POST'}).catch(()=>{});
  localStorage.removeItem('bb_uid');location.reload()}

async function loadMe(){
  let r=null;
  if(ME&&ME.signed_in){
    const m=await fetch('/api/me').then(r=>r.json()).catch(()=>null);
    if(m&&m.signed_in){ME=m;r=m.portfolio}
  } else if(uid()){
    r=await fetch('/api/user/'+uid()).then(r=>r.ok?r.json():null).catch(()=>null);
    if(!r)localStorage.removeItem('bb_uid');
  }
  if(!r){$('#mebody').innerHTML='<div class="skel">Couldn\'t find that Blackbox — it may have been deleted.</div>';return}
  renderMe(r);
}
function renderMe(r){
  window._meUid=r.uid;LAST_PORTFOLIO=r;
  const up=r.since_start>=0,dayUp=r.day_ret>=0;
  const names={};CAT.strategies.forEach(s=>names[s.id]=s.name);
  $('#mebody').innerHTML=`
    <div class="mehead rv in">
      <div><div class="sec-k">Your Blackbox</div>
        <h2 style="margin-bottom:4px">${r.name}'s Blackbox</h2>
        <div style="color:var(--mut);font-size:13.5px">since ${r.created} · paper portfolio · marked to live markets</div></div>
      <div class="mehead-btns"><button class="cta ghost" onclick="printStatement()">🖨 Statement</button>
      <button class="cta ghost" onclick="resetMe()">Change strategy</button></div></div>
    <div style="margin-top:26px">
      <span class="nav-big">${fmt2(r.nav)}</span>
      <span class="pill ${up?'up':'dn'}">${up?'+':''}${r.since_start}% all-time</span>
      <span class="pill ${dayUp?'up':'dn'}">${dayUp?'+':''}${r.day_ret}% today</span></div>
    <div id="badges" class="badges"></div>
    <div class="megrid">
      <div>
        <div class="mbox" style="background:var(--surface2)"><h4>Portfolio value</h4><div class="chartbox short"><canvas id="navchart"></canvas></div>
          <div class="note" style="margin-top:10px">${r.note}</div></div>
        <div class="mbox" style="background:var(--surface2);margin-top:14px"><h4>Your mix</h4>
          <div class="mixbar">${Object.entries(r.mix).map(([id,w],i)=>`<i style="width:${w}%;background:${COLORS[i%COLORS.length]}"></i>`).join('')}</div>
          <div class="mixlegend">${Object.entries(r.mix).map(([id,w],i)=>`<span><span class="dotc" style="background:${COLORS[i%COLORS.length]}"></span><b>${w}%</b> ${names[id]||id}</span>`).join('')}</div></div>
        <div class="mbox" style="background:var(--surface2);margin-top:14px"><h4>AI summary</h4>
          <div id="narrtext" style="margin-top:8px"><span class="askel"><span class="sp"></span>Reading your portfolio…</span></div></div>
        <div class="mbox" style="background:var(--surface2);margin-top:14px"><h4>🧠 AI risk review</h4>
          <div id="reviewtext" style="margin-top:8px"><span class="askel"><span class="sp"></span>Analysing your mix's diversification and risk…</span></div></div>
        <div class="mbox" style="background:var(--surface2);margin-top:14px"><h4>Ask your Blackbox</h4>
          <div id="qalog"></div>
          <div class="qform"><input class="ain" id="qainput" placeholder="e.g. why did I lose money today?" onkeydown="if(event.key==='Enter')askPortfolio()">
            <button class="cta ghost small" onclick="askPortfolio()">Ask</button></div></div>
      </div>
      <div><h4 style="font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin:4px 0 0">What the machine is doing right now</h4>
        <div id="mestrats">${r.strategies.map(s=>{
          const dr=(s.day_ret*100),drUp=dr>=0;
          return `<div class="scard2"><div class="h2r"><h3>${names[s.id]||s.id}</h3>
            <div><span class="wtag">${s.weight}%</span> <span class="pill ${drUp?'up':'dn'}" style="vertical-align:0;margin-left:6px">${drUp?'+':''}${dr.toFixed(2)}% today</span></div></div>
          <div class="note">${s.note}</div>
          ${s.holdings.length?`<div class="hlist">${s.holdings.map(h=>`<span class="htag"><b>${h.name}</b> ${h.w}%</span>`).join('')}</div>`:'<div class="hlist"><span class="htag">100% cash</span></div>'}
          ${s.tilt_why?`<div class="tiltwarn">⚠ ${s.tilt_why}</div>`:''}
          </div>`}).join('')}</div></div>
    </div>`;
  const hist=r.history.map(h=>({t:h[0],v:h[1]}));
  if(hist[hist.length-1].t!==new Date().toISOString().slice(0,10))hist.push({t:'today',v:r.nav});
  else hist[hist.length-1]={t:'today',v:r.nav};
  if(navChart)navChart.destroy();
  const ng=document.getElementById('navchart').getContext('2d').createLinearGradient(0,0,0,230);
  ng.addColorStop(0,'rgba(201,168,106,.30)');ng.addColorStop(1,'rgba(201,168,106,0)');
  navChart=new Chart(document.getElementById('navchart'),{type:'line',
    data:{labels:hist.map(h=>h.t),datasets:[{data:hist.map(h=>h.v),borderColor:'#C9A961',borderWidth:2.6,
      pointRadius:hist.length<15?4:0,pointBackgroundColor:'#0B0906',pointBorderColor:'#E8D5A3',pointBorderWidth:2,
      pointHoverRadius:6,fill:true,backgroundColor:ng,tension:.3,borderCapStyle:'round'}]},
    options:{responsive:true,maintainAspectRatio:false,resizeDelay:60,interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:c=>fmt2(c.parsed.y)}}},
      scales:{x:{grid:{display:false},border:{display:false},ticks:{...AXIS,maxTicksLimit:6}},
        y:{position:'right',grid:{color:'rgba(201,168,106,.08)',drawTicks:false},border:{display:false},
          ticks:{...AXIS,maxTicksLimit:5,padding:8,callback:kfmt}}}}});
  loadNarration(r.uid);loadReview(r.uid);renderBadges(r);
}

/* ---------- AI narration + Q&A ---------- */
async function loadNarration(uid){
  const box=$('#narrtext');if(!box)return;
  const r=await fetch('/api/user/'+uid+'/narrate').then(r=>r.json()).catch(()=>null);
  box.textContent=(r&&r.text)?r.text:"AI summary isn't available right now — it needs an AI key configured on the server.";
}
async function loadReview(uid){
  const box=$('#reviewtext');if(!box)return;
  const r=await fetch('/api/user/'+uid+'/review').then(r=>r.json()).catch(()=>null);
  box.innerHTML=(r&&r.text)?`<div class="airich">${esc(r.text)}</div>`:"AI risk review isn't available right now — it needs an AI key on the server.";
}

/* ---------- milestone badges (computed from real portfolio data) ---------- */
function renderBadges(r){
  const box=$('#badges');if(!box)return;
  const hist=r.history||[];
  const days=hist.length;
  const navs=hist.map(h=>h[1]);
  const peak=Math.max(...navs,r.nav);
  const atHigh=r.nav>=peak-0.005;
  // count down days survived (consecutive-agnostic: any day that fell but book still here)
  let downDays=0;for(let i=1;i<navs.length;i++)if(navs[i]<navs[i-1])downDays++;
  const b=[];
  b.push(['🚀','Launched','You started a live paper Blackbox']);
  if(days>=1)b.push(['📅',days+' day'+(days>1?'s':'')+' live','Your book has been marked to market '+days+' day'+(days>1?'s':'')]);
  if(downDays>=1)b.push(['💎','Held through a red day','You stayed invested through '+downDays+' down day'+(downDays>1?'s':'')]);
  if(days>=7)b.push(['🗓️','One week in','A full week of live marks accumulated']);
  if(days>=30)b.push(['🏆','One month in','30+ days of real history']);
  if(r.since_start>0)b.push(['📈','In the green','Positive since you started']);
  if(atHigh&&days>2)b.push(['⛰️','All-time high','Your NAV is at its highest yet']);
  const mixN=Object.keys(r.mix||{}).length;
  if(mixN>=5)b.push(['🌐','Well diversified','Your mix spans '+mixN+' strategies']);
  box.innerHTML=b.map(([e,t,d])=>`<div class="badge2" title="${d}"><span class="be">${e}</span><span class="bt">${t}</span></div>`).join('');
}

/* ---------- printable statement (real data) ---------- */
let LAST_PORTFOLIO=null;
function printStatement(){
  const r=LAST_PORTFOLIO;if(!r)return;
  const names={};(CAT&&CAT.strategies||[]).forEach(s=>names[s.id]=s.name);
  const rows=Object.entries(r.mix).map(([id,w])=>`<tr><td>${names[id]||id}</td><td style="text-align:right">${w}%</td></tr>`).join('');
  const stances=r.strategies.map(s=>`<tr><td>${names[s.id]||s.id}</td><td>${(s.note||'').replace(/</g,'&lt;')}</td></tr>`).join('');
  const w=window.open('','_blank');
  w.document.write(`<html><head><title>Blackbox Statement — ${esc(r.name)}</title>
    <style>body{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:#111;max-width:720px;margin:40px auto;padding:0 20px}
    h1{font-size:24px;margin-bottom:2px}.muted{color:#666;font-size:13px}table{width:100%;border-collapse:collapse;margin:16px 0}
    td,th{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left;font-size:13px}h2{font-size:16px;margin-top:28px}
    .big{font-size:32px;font-weight:800;margin:10px 0}.disc{color:#777;font-size:11px;margin-top:30px;line-height:1.5}</style></head><body>
    <h1>Blackbox — Portfolio Statement</h1>
    <div class="muted">${esc(r.name)}'s Blackbox · started ${r.created} · generated ${new Date().toISOString().slice(0,10)}</div>
    <div class="big">${fmt2(r.nav)} <span style="font-size:15px;font-weight:500;color:#666">practice units</span></div>
    <div class="muted">${r.since_start>=0?'+':''}${r.since_start}% since start · ${r.day_ret>=0?'+':''}${r.day_ret}% today</div>
    <h2>Allocation</h2><table><tr><th>Strategy</th><th style="text-align:right">Weight</th></tr>${rows}</table>
    <h2>Current positioning</h2><table><tr><th>Strategy</th><th>Stance today</th></tr>${stances}</table>
    <div class="disc">Paper portfolio of practice capital — no real money, no brokerage account. Every figure derives from real historical backtests (2006–2026) and live market marks. Backtested performance does not guarantee future results. Not investment advice.</div>
    </body></html>`);
  w.document.close();setTimeout(()=>w.print(),300);
}
async function askPortfolio(){
  const inp=$('#qainput');const q=(inp.value||'').trim();if(!q)return;
  const log=$('#qalog');const uid=window._meUid;
  const row=document.createElement('div');row.className='qarow';
  row.innerHTML=`<div class="q">${q.replace(/</g,'&lt;')}</div><div class="a"><span class="sp"></span> thinking…</div>`;
  log.appendChild(row);inp.value='';inp.disabled=true;
  const r=await fetch('/api/user/'+uid+'/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})}).then(r=>r.json()).catch(()=>null);
  inp.disabled=false;inp.focus();
  row.querySelector('.a').textContent=(r&&r.answer)?r.answer:"Couldn't get an answer — AI may not be configured on the server.";
}

/* ---------- modal + charts (redesigned) ---------- */
function openModal(id){
  const s=CAT.strategies.find(z=>z.id===id);if(!s)return;
  const st=s.stats,pj=s.projections,sc=sentFor(id);
  $('#modal').innerHTML=`<button class="x" onclick="closeModal()">×</button>
    <span class="badge ${s.risk.replace(' ','')}">${s.risk} risk</span> ${sc?sentChip(id):''}
    <h2 style="font-size:30px;margin:12px 0 6px">${s.name}</h2>
    <p style="color:var(--mut)">${s.tagline}</p>
    <p style="color:var(--mut);font-size:12.5px;margin-top:16px">💡 Tap any underlined metric below for a plain-English AI explanation.</p>
    <div class="stats" style="margin-top:10px">
      <div class="st"><div class="n">${st.cagr}%</div><div class="l">${mt('CAGR')}</div></div>
      <div class="st"><div class="n">${st.sharpe}</div><div class="l">${mt('Sharpe','Sharpe ratio')}</div></div>
      <div class="st"><div class="n">${st.vol}%</div><div class="l">${mt('Volatility')}</div></div>
      <div class="st"><div class="n" style="color:var(--red)">${st.max_dd}%</div><div class="l">${mt('Max DD','Maximum drawdown')}</div></div>
      <div class="st"><div class="n">${st.beta==null?'—':st.beta}</div><div class="l">${mt('Beta vs S&P','Beta (stock market)')}</div></div>
      <div class="st"><div class="n">${st.sharpe_h1} / ${st.sharpe_h2}</div><div class="l">${mt('Sharpe by decade','Split-half validation')}</div></div>
    </div>
    <div class="mrow">
      <div class="mbox"><h4>Growth of 10,000 — backtest vs S&P 500</h4><div class="chartbox"><canvas id="mcurve"></canvas></div>
        <div class="note" style="margin-top:12px"><span class="dotc" style="background:#C9A961"></span>${s.name}
        &nbsp;&nbsp;<span class="dotc" style="background:#6a6a78"></span>S&P 500 buy &amp; hold · ${s.span}</div></div>
      <div class="mbox"><h4>Where 10,000 could be — bootstrap projection</h4><div class="chartbox"><canvas id="mproj"></canvas></div>
        <div class="hchips">
          ${['1y','3y','5y'].map(h=>`<div class="hchip"><span class="n">${fmt(pj[h].p50)}</span>median · ${h}<br>
            <span style="color:var(--mut)">${fmt(pj[h].p25)} – ${fmt(pj[h].p75)} likely</span></div>`).join('')}
        </div>
        <div class="note" style="margin-top:10px">Bands: 800 resamples of real history — inner = likely (25-75%), outer = extremes (5-95%). Chance of loss after 5y: <b>${pj['5y'].prob_loss}%</b>.</div></div>
    </div>
    <div class="mbox" style="margin-top:22px"><h4>Split-half validation — did the edge hold in both halves of history?</h4>
      <div class="dstats">
        <div class="dstat"><div class="dn">${st.sharpe_h1}</div><div class="dl">Sharpe, first decade</div></div>
        <div class="dstat"><div class="dn">${st.sharpe_h2}</div><div class="dl">Sharpe, second decade</div></div>
        <div class="dstat"><div class="dn" style="color:var(--green)">${st.best_year}%</div><div class="dl">Best year</div></div>
        <div class="dstat"><div class="dn" style="color:var(--red)">${st.worst_year}%</div><div class="dl">Worst year</div></div>
        <div class="dstat"><div class="dn">${st.years}</div><div class="dl">Years of data</div></div>
        <div class="dstat"><div class="dn">${st.corr_spx==null?'—':st.corr_spx}</div><div class="dl">Correlation to S&P</div></div>
      </div>
      <div class="note" style="margin-top:12px">Every edge here was found on the first decade and confirmed on the second — a rule that only works in one half is discarded. Two positive Sharpe figures above mean the edge repeated across independent halves of history, which is the honest test that kills most curve-fit strategies.</div></div>
    <div class="method"><b>Methodology.</b> ${s.method}</div>
    <div class="mbox" style="margin-top:18px"><h4>🧠 AI deep-dive — why this works</h4>
      <div id="deepdive"><button class="cta ghost small" onclick="loadDeepDive('${s.id}')">Explain this strategy with AI →</button></div></div>
    <button class="cta" style="margin-top:20px" onclick='closeModal();showName({"${s.id}":100});'>Trade this strategy →</button>`;
  $('#mask').classList.add('show');document.body.style.overflow='hidden';
  setTimeout(()=>{drawCurve(s);drawProj(s)},30);
}
async function loadDeepDive(sid){
  const box=$('#deepdive');
  box.innerHTML='<div class="askel"><span class="sp"></span>AI is analysing the strategy…</div>';
  const d=await fetch('/api/ai/strategy/'+sid).then(r=>r.json()).catch(()=>null);
  box.innerHTML=`<div class="airich">${d&&d.text?esc(d.text):"AI isn't available right now — needs an AI key on the server."}</div>`;
}
function closeModal(){$('#mask').classList.remove('show');document.body.style.overflow='';
  if(projChart){projChart.destroy();projChart=null}if(curveChart){curveChart.destroy();curveChart=null}}

const AXIS={color:'#9C9080',font:{size:11,family:'Inter'}};
const TT={backgroundColor:'rgba(23,19,16,.96)',borderColor:'rgba(201,168,106,.3)',borderWidth:1,
  titleColor:'#A79A82',bodyColor:'#F3ECDD',padding:12,cornerRadius:10,displayColors:false,
  titleFont:{family:'Inter',size:11},bodyFont:{family:'Inter',size:13,weight:'600'}};
function kfmt(v){return v>=1000?(v/1000).toLocaleString(undefined,{maximumFractionDigits:1})+'k':fmt(v)}

function drawCurve(s){
  const ctx=document.getElementById('mcurve');if(!ctx)return;
  const spx=CAT.strategies.find(z=>z.id==='spx_hold');
  const labels=s.curve.map(p=>p.t);
  let spxData=labels.map(()=>null);
  if(spx&&s.id!=='spx_hold'){const sp=spx.curve;let j=0;
    spxData=labels.map(t=>{while(j<sp.length-1&&sp[j+1].t<=t)j++;
      return sp[j].t<=t?sp[j].v*10000:null});}
  const g=ctx.getContext('2d').createLinearGradient(0,0,0,280);
  g.addColorStop(0,'rgba(201,168,106,.32)');g.addColorStop(.6,'rgba(201,168,106,.06)');g.addColorStop(1,'rgba(201,168,106,0)');
  // yearly x labels only
  let lastYr='';
  const yrLabel=t=>{const y=t.slice(0,4);if(y!==lastYr){lastYr=y;return y}return ''};
  curveChart=new Chart(ctx,{type:'line',data:{labels,datasets:[
    {data:s.curve.map(p=>p.v*10000),borderColor:'#C9A961',borderWidth:2.5,pointRadius:0,pointHitRadius:16,
      fill:true,backgroundColor:g,tension:.3,borderCapStyle:'round'},
    {data:spxData,borderColor:'rgba(145,135,115,.7)',borderWidth:1.4,borderDash:[6,5],pointRadius:0,
      pointHitRadius:12,fill:false,tension:.3,spanGaps:true}]},
    options:{responsive:true,maintainAspectRatio:false,resizeDelay:60,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{...TT,
        callbacks:{title:i=>i[0]?i[0].label:'',
          label:c=>(c.datasetIndex?'S&P 500   ':'This strategy   ')+fmt(c.parsed.y)}}},
      scales:{x:{grid:{display:false},border:{display:false},
        ticks:{...AXIS,maxRotation:0,autoSkip:false,callback:function(v){return yrLabel(this.getLabelForValue(v)||'')}}},
      y:{position:'right',grid:{color:'rgba(201,168,106,.08)',drawTicks:false},border:{display:false},
        ticks:{...AXIS,maxTicksLimit:5,padding:8,callback:kfmt}}}}});
}
function drawProj(s){
  const ctx=document.getElementById('mproj');if(!ctx)return;
  const pj=s.projections,xs=['today','1 year','3 years','5 years'];
  const band=k=>[10000,pj['1y'][k],pj['3y'][k],pj['5y'][k]];
  const gi=ctx.getContext('2d').createLinearGradient(0,0,0,280);
  gi.addColorStop(0,'rgba(201,168,106,.30)');gi.addColorStop(1,'rgba(201,168,106,.10)');
  projChart=new Chart(ctx,{type:'line',data:{labels:xs,datasets:[
    {data:band('p95'),borderColor:'rgba(160,150,130,.30)',borderWidth:1,borderDash:[4,5],pointRadius:0,fill:false,tension:.35},
    {data:band('p5'),borderColor:'rgba(160,150,130,.30)',borderWidth:1,borderDash:[4,5],pointRadius:0,tension:.35,
      fill:'-1',backgroundColor:'rgba(160,150,130,.06)'},
    {data:band('p75'),borderColor:'transparent',borderWidth:0,pointRadius:0,tension:.35,fill:false},
    {data:band('p25'),borderColor:'transparent',borderWidth:0,pointRadius:0,tension:.35,fill:'-1',backgroundColor:gi},
    {data:band('p50'),borderColor:'#E8D5A3',borderWidth:3,pointRadius:4.5,pointBackgroundColor:'#0B0906',
      pointBorderColor:'#E8D5A3',pointBorderWidth:2.5,pointHoverRadius:6,tension:.35,fill:false}]},
    options:{responsive:true,maintainAspectRatio:false,resizeDelay:60,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{...TT,filter:i=>[1,3,4].includes(i.datasetIndex),
        itemSort:(a,b)=>b.parsed.y-a.parsed.y,
        callbacks:{label:c=>({1:'Worst 5%   ',3:'Likely low   ',4:'Median   '}[c.datasetIndex]||'')+fmt(c.parsed.y)}}},
      scales:{x:{grid:{display:false},border:{display:false},ticks:{...AXIS,padding:6}},
        y:{position:'right',grid:{color:'rgba(201,168,106,.08)',drawTicks:false},border:{display:false},
          ticks:{...AXIS,maxTicksLimit:5,padding:8,callback:kfmt}}}}});
}

/* ---------- sentiment section ---------- */
function renderSentiment(){
  if(!SENT||!SENT.assets)return;
  const names={equities:'Equities',gold:'Gold & metals',rates:'Rates & Fed',fx:'FX & dollar',crypto:'Crypto',oil:'Oil & energy'};
  $('#sgrid').innerHTML=Object.entries(SENT.assets).map(([a,v],i)=>{
    const pos=Math.max(0,Math.min(100,(v.score+100)/2));
    const col=v.score>12?'var(--green)':v.score<-12?'var(--red)':'var(--mut)';
    return `<div class="scard rv in">
      <div class="h"><h3>${names[a]||a}</h3><span class="sscore" style="color:${col}">${v.score>0?'+':''}${v.score}</span></div>
      <div class="gauge"><i style="left:calc(${pos}% - 2px)"></i></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--mut)"><span>bearish</span><span>${v.n_items} items · ${v.trend}</span><span>bullish</span></div>
      <div class="sexp">${v.explanation}</div>
      ${(v.top||[]).slice(0,3).map(t=>`<div class="head"><span class="t">${t.title}</span><span class="s" style="color:${t.score>0?'var(--green)':t.score<0?'var(--red)':'var(--mut)'}">${t.src}</span></div>`).join('')}
    </div>`}).join('');
}
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const aiUnavail="AI isn't available right now — this feature needs an AI key configured on the server.";

/* ---------- inline 'explain this metric' AI popover ---------- */
const GLOSS_CACHE={};
async function explainMetric(ev,term){
  ev.stopPropagation();
  const pop=$('#mpop');
  const x=Math.min(ev.clientX,innerWidth-320),y=ev.clientY+14;
  pop.style.left=Math.max(12,x)+'px';pop.style.top=y+'px';pop.style.display='block';
  pop.innerHTML=`<h5>${esc(term)}</h5><div><span class="sp"></span> asking AI…</div>`;
  let txt=GLOSS_CACHE[term];
  if(!txt){
    const d=await fetch('/api/ai/glossary',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({term})}).then(r=>r.json()).catch(()=>null);
    txt=(d&&d.text)?d.text:aiUnavail;GLOSS_CACHE[term]=txt;
  }
  pop.innerHTML=`<h5>${esc(term)}</h5><div>${esc(txt)}</div>`;
}
document.addEventListener('click',e=>{const p=$('#mpop');if(p&&p.style.display!=='none'&&!p.contains(e.target)&&!e.target.classList.contains('metric-term'))p.style.display='none'});
// helper to build a clickable metric label
function mt(label,term){return `<span class="metric-term" onclick="explainMetric(event,'${(term||label).replace(/'/g,"\\'")}')">${label}</span>`}

/* ---------- Regime Radar suite (five models we trained ourselves) ---------- */
let regimeLoaded=false;
async function loadRegime(){
  if(regimeLoaded)return;regimeLoaded=true;
  const box=$('#regimeBox');
  box.innerHTML='<div class="askel"><span class="sp"></span>Running five neural networks on the latest market data…</div>';
  const d=await fetch('/api/regime').then(r=>r.json()).catch(()=>null);
  if(!d||d.available===false||!d.predictions){
    box.innerHTML='<div class="askel">The models aren\'t available — run <code>python3 regime_ai.py train</code> to build them.</div>';return;
  }
  const order=['crypto','equities','bonds','commodities','fx'];
  const preds=d.predictions,rc=(d.report&&d.report.classes)||{};
  // live prediction cards
  const cards=order.filter(c=>preds[c]).map(c=>{
    const p=preds[c],pct=Math.round(p.prob_turbulent*100),calm=p.regime==='calm';
    return `<div class="regmini">
      <div class="rmname">${p.name}<span class="rmasset">${p.label}</span></div>
      <div class="regverdict2 ${p.regime}">${calm?'Calm':'Turbulent'}</div>
      <div class="reggauge sm"><i style="left:calc(${Math.max(0,Math.min(100,pct))}% - 2px)"></i></div>
      <div class="rmmeta">${pct}% turbulence · ${p.instrument} · next ${p.horizon_days}d<br>test skill (AUC) <b style="color:var(--txt)">${p.test_auc!=null?p.test_auc.toFixed(2):'—'}</b></div>
    </div>`}).join('');
  // honest cross-asset comparison table
  const rows=order.filter(c=>rc[c]).map(c=>{
    const r=rc[c],m=r.chosen_test,pers=r.test.baseline_persistence.roc_auc,ds=r.dataset;
    return `<tr><td class="win">${r.name}</td><td>${r.label}</td>
      <td class="win">${m.roc_auc.toFixed(3)}</td><td>${pers.toFixed(3)}</td>
      <td>${r.edge_over_persistence>=0?'+':''}${r.edge_over_persistence.toFixed(3)}</td>
      <td>${(ds.rows/1000).toFixed(0)}k</td></tr>`}).join('');
  const ranked=order.filter(c=>rc[c]).map(c=>rc[c]).sort((a,b)=>b.chosen_test.roc_auc-a.chosen_test.roc_auc);
  const best=ranked[0];
  box.innerHTML=`<div class="regcard rv in">
    <div class="regmgrid">${cards}</div>
    <div class="regnote"><b>How to read this honestly.</b> AUC measures skill from 0.50 (a coin-flip) to 1.00 (perfect).
      Each model is tested on years of data it never saw in training. Every one beats its naive baseline — and the
      <b>edge</b> column is the honest measure of what the machine learning adds beyond simply noticing "volatility is already high".</div>
    <h4 style="margin:22px 0 8px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.08em">Cross-asset scorecard — out-of-sample</h4>
    <div style="overflow-x:auto"><table class="regtable">
      <thead><tr><th>Model</th><th>Asset class</th><th>AUC</th><th>Naive AUC</th><th>Edge</th><th>Data</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    ${best?`<div class="regfind">🔬 Finding: <b>${best.name}</b> (${best.label}) turbulence is the most forecastable — AUC ${best.chosen_test.roc_auc.toFixed(2)}. In general, the steadier the market, the more predictable its turbulence; crypto is the wildest and the hardest.</div>`:''}
    <p style="font-size:12px;color:var(--mut);margin-top:14px">Five separate neural networks, each trained on ~20 years of real daily data (${(Object.values(rc).reduce((a,r)=>a+r.dataset.rows,0)/1000).toFixed(0)}k total examples). Full method on the <a onclick="go('ai')" style="color:var(--acc);cursor:pointer">AI Inside</a> page.</p>
  </div>`;
}

/* ---------- AI positioning desk (Intelligence page) ---------- */
let posLoaded=false;
async function loadPositioning(){
  if(posLoaded)return;posLoaded=true;
  const box=$('#posGrid');
  box.innerHTML='<div class="askel"><span class="sp"></span>AI is reading every strategy\'s live positioning…</div>';
  const d=await fetch('/api/ai/positioning').then(r=>r.json()).catch(()=>null);
  if(!d||d.available===false){box.innerHTML='<div class="askel">'+aiUnavail+'</div>';return}
  const names={};(CAT&&CAT.strategies||[]).forEach(s=>names[s.id]=s.name);
  box.innerHTML='<div class="posgrid">'+Object.entries(d).map(([sid,v])=>{
    const brake=v.tilt<1;
    return `<div class="poscard"><h3>${names[sid]||v.name||sid}
      <span class="postag ${brake?'brake':'clear'}">${brake?'AI brake · '+Math.round(v.tilt*100)+'%':'AI clear · 100%'}</span></h3>
      <div class="posnote">${esc(v.ai_note)}</div>
      <div class="poshold">${(v.holdings&&v.holdings.length?v.holdings.map(h=>`<span class="htag">${h.name} ${h.w}%</span>`).join(''):'<span class="htag">100% cash</span>')}</div>
    </div>`}).join('')+'</div>';
}

/* ---------- Daily briefing page ---------- */
let briefLoaded=false;
async function loadBriefing(){
  if(briefLoaded)return;briefLoaded=true;
  const box=$('#briefingBox'),scores=$('#briefingScores');
  box.innerHTML='<div class="askel"><span class="sp"></span>AI is writing today\'s briefing from the live scan…</div>';
  const d=await fetch('/api/ai/briefing').then(r=>r.json()).catch(()=>null);
  if(!d||d.available===false||!d.text){box.innerHTML='<div class="askel">'+aiUnavail+'</div>';return}
  const names={equities:'Equities',gold:'Gold',rates:'Rates',fx:'FX',crypto:'Crypto',oil:'Oil'};
  scores.innerHTML=Object.entries(d.scores||{}).map(([a,v])=>{
    const col=v>12?'var(--green)':v<-12?'var(--red)':'var(--mut)';
    return `<div class="bscore"><div class="ba">${names[a]||a}</div><div class="bv" style="color:${col}">${v>0?'+':''}${v}</div></div>`}).join('');
  box.innerHTML=`<div class="airich">${esc(d.text)}</div>`;
}

/* ---------- Learn page (glossary + tutor) ---------- */
let learnLoaded=false;
const GLOSS_TERMS=['CAGR','Sharpe ratio','Max drawdown','Volatility','Beta','Risk parity',
  'Momentum investing','Trend following','Diversification','Volatility targeting',
  'Block bootstrap','Paper trading','Seasonality (Sell in May)','Sentiment overlay'];
function ensureLearnPage(){
  if(learnLoaded)return;learnLoaded=true;
  $('#glossChips').innerHTML=GLOSS_TERMS.map(t=>`<button class="glchip" onclick="explainTerm(this,'${t.replace(/'/g,"\\'")}')">${t}</button>`).join('');
}
async function explainTerm(btn,term){
  document.querySelectorAll('.glchip').forEach(c=>c.classList.remove('on'));btn.classList.add('on');
  const box=$('#glossBox');box.style.display='block';
  box.innerHTML='<div class="askel"><span class="sp"></span>Explaining “'+esc(term)+'”…</div>';
  const d=await fetch('/api/ai/glossary',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({term})}).then(r=>r.json()).catch(()=>null);
  box.innerHTML=`<h3 style="font-size:17px;margin-bottom:8px">${esc(term)}</h3><div class="airich">${d&&d.text?esc(d.text):aiUnavail}</div>`;
}
let COACH_HIST=[];
async function askCoach(){
  const inp=$('#coachInput');const q=(inp.value||'').trim();if(!q)return;
  const log=$('#coachLog');
  const row=document.createElement('div');row.className='qarow';
  row.innerHTML=`<div class="q">${esc(q)}</div><div class="a"><span class="sp"></span> thinking…</div>`;
  log.appendChild(row);inp.value='';inp.disabled=true;
  const d=await fetch('/api/ai/coach',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q,history:COACH_HIST})}).then(r=>r.json()).catch(()=>null);
  inp.disabled=false;inp.focus();
  const ans=(d&&d.answer)?d.answer:aiUnavail;
  row.querySelector('.a').textContent=ans;
  COACH_HIST.push({role:'user',text:q},{role:'ai',text:ans});
}

/* ---------- Compare page ---------- */
let cmpLoaded=false;
function ensureComparePage(){
  if(!cmpLoaded){cmpLoaded=true;
    const opts=(CAT&&CAT.strategies||[]).map(s=>`<option value="${s.id}">${s.name}</option>`).join('');
    $('#cmpA').innerHTML=opts;$('#cmpB').innerHTML=opts;
    const ss=CAT&&CAT.strategies||[];
    if(ss[1])$('#cmpA').value=ss[0].id;if(ss[3])$('#cmpB').value=ss[3].id;
    $('#cmpA').onchange=$('#cmpB').onchange=runCompare;
  }
  runCompare();
}
function runCompare(){
  const a=$('#cmpA').value,b=$('#cmpB').value;
  const sa=CAT.strategies.find(s=>s.id===a),sb=CAT.strategies.find(s=>s.id===b);
  if(!sa||!sb)return;
  const rows=[['CAGR','cagr','%','CAGR'],['Sharpe','sharpe','','Sharpe ratio'],['Volatility','vol','%','Volatility'],
    ['Max drawdown','max_dd','%','Maximum drawdown'],['Beta vs S&P','beta','','Beta (stock market)'],['Risk','risk','','Risk rating']];
  $('#cmpTable').innerHTML=`<table class="cmptable"><thead><tr><th>Metric</th><th>${sa.name}</th><th>${sb.name}</th></tr></thead><tbody>`+
    rows.map(([lab,k,u,term])=>{const va=k==='risk'?sa.risk:sa.stats[k],vb=k==='risk'?sb.risk:sb.stats[k];
      return `<tr><td class="metric">${mt(lab,term)}</td><td class="val">${va}${u}</td><td class="val">${vb}${u}</td></tr>`}).join('')+
    '</tbody></table>';
  const box=$('#cmpBox');box.style.display='block';
  box.innerHTML='<div class="askel"><span class="sp"></span>AI is comparing the two…</div>';
  fetch('/api/ai/compare',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({a,b})}).then(r=>r.json()).then(d=>{
      box.innerHTML=`<h3 style="font-size:17px;margin-bottom:8px">AI verdict</h3><div class="airich">${d&&d.text?esc(d.text):aiUnavail}</div>`;
    }).catch(()=>{box.innerHTML='<div class="askel">'+aiUnavail+'</div>'});
}
/* ==================== i18n: English / German ==================== */
let LANG=localStorage.getItem('bb_lang')||'en';
// route the language preference to the AI backend on every request
const _oFetch=window.fetch.bind(window);
window.fetch=(u,o)=>{o=o||{};const h=new Headers(o.headers||{});h.set('X-Lang',LANG);o.headers=h;return _oFetch(u,o);};

const DE={}, REV={};
function tnorm(s){return (s||'').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&').replace(/ /g,' ').replace(/\s+/g,' ').trim();}
function d(en,de){DE[tnorm(en)]=de;REV[tnorm(de)]=en;}
const PH=[  // [id, english, german]
  ['freetext',"e.g. I just had a baby and want to be a bit more careful than usual, even though my risk tolerance on paper is high…","z. B. Ich habe gerade ein Kind bekommen und möchte etwas vorsichtiger sein als sonst, auch wenn meine Risikobereitschaft auf dem Papier hoch ist…"],
  ['uname',"Name your Blackbox (e.g. Nico's)","Benenne deine Blackbox (z. B. Nicos)"],
  ['coachInput',"e.g. Why does gold do well when yields fall?","z. B. Warum steigt Gold, wenn die Zinsen fallen?"],
  ['qainput',"e.g. why did I lose money today?","z. B. Warum habe ich heute Geld verloren?"],
  ['regname',"Your name","Dein Name"],
  ['keyin',"e.g. violet-falcon-ember-4821","z. B. violet-falcon-ember-4821"]];

/* ---- nav / chrome ---- */
d("Home","Startseite");d("Start","Loslegen");d("Strategies","Strategien");d("Briefing","Briefing");
d("Intelligence","Intelligenz");d("Learn","Lernen");d("AI Inside","KI im Detail");d("Compare","Vergleichen");
d("Your Blackbox","Deine Blackbox");d("Sign in","Anmelden");d("Sign out","Abmelden");

/* ---- hero ---- */
d('<span class="dot"></span>Live engine · 5 asset classes · AI market scanner','<span class="dot"></span>Live-Engine · 5 Anlageklassen · KI-Marktscanner');
d("Choose an institutional-grade systematic strategy — or let six questions choose it for you — then press go. Your own Blackbox starts trading on paper, marked to live markets, guided by validated rules and an AI that reads the news.","Wähle eine systematische Strategie auf institutionellem Niveau — oder lass sechs Fragen sie für dich auswählen — und leg los. Deine eigene Blackbox handelt auf Papierbasis, bewertet zu Live-Marktpreisen, geführt von validierten Regeln und einer KI, die die Nachrichten liest.");
d("Let's go →","Los geht's →");d("View my Blackbox →","Meine Blackbox ansehen →");
d("strategies, fully backtested","Strategien, vollständig backgetestet");
d("of real market data","echte Marktdaten");d("sources scanned by the AI","von der KI gescannte Quellen");

/* ---- start / journey ---- */
d("01 — Start your journey","01 — Starte deine Reise");
d("Two ways in. One press of go.","Zwei Wege hinein. Ein Druck auf Los.");
d("Answer six questions and every answer visibly shapes the outcome — or skip the quiz and build your own mix straight from the numbers.","Beantworte sechs Fragen und jede Antwort formt sichtbar das Ergebnis — oder überspringe das Quiz und stelle deinen eigenen Mix direkt aus den Zahlen zusammen.");
d("🧭 Guide me","🧭 Führe mich");
d("A six-question profile. We show exactly which answers capped or shaped your mix — no black box.","Ein Profil aus sechs Fragen. Wir zeigen genau, welche Antworten deinen Mix begrenzt oder geformt haben — keine Blackbox.");
d("🎛 I'll choose myself","🎛 Ich wähle selbst");
d("Browse the strategies below and set your own weights by CAGR, volatility and max drawdown.","Durchstöbere die Strategien unten und setze deine eigenen Gewichte nach CAGR, Volatilität und maximalem Drawdown.");
d("1 · How experienced are you with investing?","1 · Wie erfahren bist du beim Investieren?");
d("I'm new","Ich bin neu");d("Some experience","Etwas Erfahrung");d("Experienced","Erfahren");
d("2 · How long can this money stay invested?","2 · Wie lange kann dieses Geld investiert bleiben?");
d("<1 year","<1 Jahr");d("1–3 years","1–3 Jahre");d("3–10 years","3–10 Jahre");d("10+ years","10+ Jahre");
d("3 · The most you could watch it fall without selling?","3 · Wie viel Verlust könntest du aushalten, ohne zu verkaufen?");
d("−50% or more","−50% oder mehr");
d("4 · Your portfolio drops 20% in a month. You…","4 · Dein Portfolio fällt in einem Monat um 20%. Du…");
d("Sell everything","Verkaufe alles");d("Hold, but worry","Halte, aber sorge mich");d("Buy more","Kaufe mehr");
d("5 · What's the goal?","5 · Was ist das Ziel?");
d("Preserve capital","Kapital erhalten");d("Grow steadily","Stetig wachsen");d("Maximize growth","Wachstum maximieren");
d("6 · What matters most to you?","6 · Was ist dir am wichtigsten?");
d("A smooth ride","Eine ruhige Fahrt");d("Beating the market","Den Markt schlagen");d("Max long-term wealth","Max. langfristiges Vermögen");
d("In your own words — anything about your situation, goals or concerns the six questions didn't capture? (optional)","In deinen eigenen Worten — etwas zu deiner Situation, deinen Zielen oder Sorgen, das die sechs Fragen nicht erfasst haben? (optional)");
d("If you write something here, AI reads it and may apply a small, bounded adjustment to your mix — it can never override the validated math above, only nudge it, and it always explains why.","Wenn du hier etwas schreibst, liest die KI es und wendet womöglich eine kleine, begrenzte Anpassung auf deinen Mix an — sie kann die validierte Mathematik oben nie außer Kraft setzen, nur leicht anstoßen, und sie erklärt immer warum.");
d("Build my mix →","Meinen Mix erstellen →");
d("🚀 Start my Blackbox","🚀 Meine Blackbox starten");
d("Starts a personal paper portfolio of 10,000 marked to live markets. Demo capital — no real money.","Startet ein persönliches Papier-Portfolio von 10.000, bewertet zu Live-Marktpreisen. Demo-Kapital — kein echtes Geld.");

/* ---- strategies ---- */
d("01 — The strategies","01 — Die Strategien");
d("Real numbers. Nothing invented.","Echte Zahlen. Nichts erfunden.");
d("Every stat is a 20-year backtest — true drawdowns included. The AI scanner's live read is on every card. Click a card for the full profile.","Jede Kennzahl ist ein 20-Jahres-Backtest — echte Drawdowns inklusive. Die Live-Einschätzung des KI-Scanners steht auf jeder Karte. Klicke eine Karte für das vollständige Profil.");
d("Set weights with the steppers on the strategy cards below. Your mix must total 100%.","Setze die Gewichte mit den Reglern auf den Strategiekarten unten. Dein Mix muss in Summe 100% ergeben.");
d("allocated","zugewiesen");

/* ---- intelligence overview ---- */
d("◆ The core of Blackbox","◆ Das Herzstück von Blackbox");
d("Everything the AI sees","Alles, was die KI sieht");
d("This page is the heart of the project. Blackbox runs four distinct AI systems — but the one that matters most is <b>Regime Radar</b>: five machine-learning models <b>we designed and trained ourselves</b>, from raw market data. The others read the live world and explain it. Below you'll find every system, each with a plain-English \"how it works\" — and further down, an in-depth account of exactly how the models were built, what data they use, and how they learn.","Diese Seite ist das Herz des Projekts. Blackbox betreibt vier eigenständige KI-Systeme — doch das wichtigste ist <b>Regime Radar</b>: fünf Machine-Learning-Modelle, die <b>wir selbst entworfen und trainiert haben</b>, aus rohen Marktdaten. Die anderen lesen die aktuelle Welt und erklären sie. Unten findest du jedes System, jeweils mit einer verständlichen Erklärung \"wie es funktioniert\" — und weiter unten eine ausführliche Darstellung, wie die Modelle gebaut wurden, welche Daten sie nutzen und wie sie lernen.");
d("1 · Regime Radar","1 · Regime Radar");d("Five neural networks we trained to forecast market turbulence.","Fünf neuronale Netze, die wir trainiert haben, um Marktturbulenzen vorherzusagen.");d("our own model","unser eigenes Modell");
d("2 · Sentiment scanner","2 · Sentiment-Scanner");d("AI reads live news &amp; social mood across six asset classes.","KI liest Live-Nachrichten &amp; Social-Media-Stimmung über sechs Anlageklassen.");d("reads the world","liest die Welt");
d("3 · Historical analog","3 · Historisches Analogon");d("AI ties today's news to the closest real past market event.","KI verknüpft die heutigen Nachrichten mit dem ähnlichsten realen Marktereignis der Vergangenheit.");
d("4 · Positioning desk","4 · Positionierungs-Desk");d("AI explains how it's supporting each strategy right now.","KI erklärt, wie sie jede Strategie gerade unterstützt.");d("explains","erklärt");

/* ---- regime radar section ---- */
d("01 — Regime Radar · our own AI","01 — Regime Radar · unsere eigene KI");d("Regime Radar","Regime Radar");
d("This is what we <em>built and trained ourselves</em> — five neural networks, one per asset class, not a borrowed language model. Each learned from ~20 years of real daily market data to forecast whether the days ahead will be <b>turbulent</b> (high volatility) or <b>calm</b>. Every other AI feature in Blackbox uses a pre-trained model; these we trained. Here are their live calls and honest, measured track records — including which market's turbulence turns out to be most predictable.","Das haben wir <em>selbst gebaut und trainiert</em> — fünf neuronale Netze, eines pro Anlageklasse, kein geliehenes Sprachmodell. Jedes lernte aus rund 20 Jahren echter täglicher Marktdaten, ob die kommenden Tage <b>turbulent</b> (hohe Volatilität) oder <b>ruhig</b> werden. Jede andere KI-Funktion in Blackbox nutzt ein vortrainiertes Modell; diese hier haben wir trainiert. Hier sind ihre Live-Einschätzungen und ehrlich gemessenen Erfolgsbilanzen — einschließlich der Frage, welche Marktturbulenz sich am besten vorhersagen lässt.");
d("How it works","Wie es funktioniert");
d("<b>Data.</b> ~20 years of real daily prices for each market (crypto, stocks, bonds, commodities, FX) — over 200,000 days in total.","<b>Daten.</b> ~20 Jahre echter Tagespreise für jeden Markt (Krypto, Aktien, Anleihen, Rohstoffe, Devisen) — insgesamt über 200.000 Tage.");
d("<b>Features.</b> For every past day we compute 15 numbers describing recent behaviour (returns, volatility, trend, momentum) — using <b>only the past</b>, so the model can't cheat.","<b>Merkmale.</b> Für jeden vergangenen Tag berechnen wir 15 Zahlen, die das jüngste Verhalten beschreiben (Renditen, Volatilität, Trend, Momentum) — <b>nur mit der Vergangenheit</b>, damit das Modell nicht schummeln kann.");
d("<b>Learning.</b> A neural network studies those days and <b>teaches itself</b> which patterns tend to come before turbulence. Nobody writes the rule — the model discovers it. This is the actual \"AI\".","<b>Lernen.</b> Ein neuronales Netz studiert diese Tage und <b>bringt sich selbst bei</b>, welche Muster typischerweise vor Turbulenzen auftreten. Niemand schreibt die Regel — das Modell entdeckt sie. Das ist die eigentliche \"KI\".");
d("<b>Honest test.</b> Each model is then graded on years of data it <b>never saw in training</b>. The score (AUC) runs from 0.50 (a coin-flip) to 1.00 (perfect).","<b>Ehrlicher Test.</b> Jedes Modell wird dann an Jahren von Daten bewertet, die es <b>im Training nie gesehen hat</b>. Der Wert (AUC) reicht von 0,50 (Münzwurf) bis 1,00 (perfekt).");
d("<b>Live.</b> Today's 15 numbers go into the trained model, which outputs a turbulence probability → Calm or Turbulent.","<b>Live.</b> Die 15 Zahlen von heute gehen in das trainierte Modell, das eine Turbulenz-Wahrscheinlichkeit ausgibt → Ruhig oder Turbulent.");

/* ---- in-depth longform ---- */
d("01 · in depth","01 · im Detail");
d("How the models actually work","Wie die Modelle wirklich funktionieren");
d("This is the long version — everything you'd want to know about the machine-learning core. What the models do, exactly which data they learn from, how they were created, and how a neural network actually teaches itself. No hand-waving.","Das ist die lange Fassung — alles, was du über den Machine-Learning-Kern wissen möchtest. Was die Modelle tun, aus welchen Daten sie genau lernen, wie sie entstanden sind und wie ein neuronales Netz sich tatsächlich selbst beibringt. Kein Herumreden.");
d("1 · What these models do","1 · Was diese Modelle tun");
d("Financial markets swing between <b>calm</b> stretches and <b>turbulent</b> ones — periods of large, rapid price moves. Turbulence is when investors get hurt: crashes, whipsaws, forced selling. So we built models to answer one precise question, separately for each kind of market: <em>will the next ten trading days be turbulent (unusually high volatility) or calm?</em> — using only information available today.","Finanzmärkte schwanken zwischen <b>ruhigen</b> und <b>turbulenten</b> Phasen — Zeiten großer, schneller Kursbewegungen. Turbulenz ist, wenn Anleger verlieren: Crashs, Fehlausbrüche, Zwangsverkäufe. Deshalb haben wir Modelle gebaut, die für jede Marktart eine präzise Frage beantworten: <em>Werden die nächsten zehn Handelstage turbulent (ungewöhnlich hohe Volatilität) oder ruhig?</em> — nur mit heute verfügbaren Informationen.");
d("We deliberately forecast <b>volatility</b> rather than <b>direction</b> (whether prices go up or down). Direction is very close to impossible to predict — markets are highly efficient, so a direction model is essentially a coin-flip and would make a dishonest demo. Volatility is different: it <b>clusters</b>. Turbulent days tend to be followed by more turbulent days, and calm by calm. That clustering is a real, well-documented statistical property, which means volatility is genuinely — if imperfectly — learnable. Choosing a target a model can actually learn is itself part of doing honest machine learning.","Wir prognostizieren bewusst die <b>Volatilität</b> statt die <b>Richtung</b> (ob Kurse steigen oder fallen). Die Richtung ist nahezu unmöglich vorherzusagen — Märkte sind hocheffizient, ein Richtungsmodell wäre im Grunde ein Münzwurf und eine unehrliche Demo. Volatilität ist anders: sie <b>häuft sich</b>. Auf turbulente Tage folgen tendenziell weitere turbulente, auf ruhige ruhige. Dieses Clustering ist eine reale, gut dokumentierte statistische Eigenschaft — also ist Volatilität tatsächlich, wenn auch unvollkommen, lernbar. Ein Ziel zu wählen, das ein Modell wirklich lernen kann, gehört selbst zum ehrlichen maschinellen Lernen.");
d("The models never buy or sell anything. They output a <b>risk signal</b>: a probability of turbulence ahead, which becomes a simple Calm / Turbulent call. In the wider product this could tell a portfolio to carry less risk before a likely storm — but the trading decisions themselves always stay with the validated, rule-based strategies.","Die Modelle kaufen oder verkaufen nie etwas. Sie geben ein <b>Risikosignal</b> aus: eine Wahrscheinlichkeit für kommende Turbulenz, die zu einer einfachen Einschätzung Ruhig / Turbulent wird. Im größeren Produkt könnte das einem Portfolio sagen, vor einem wahrscheinlichen Sturm weniger Risiko zu tragen — doch die Handelsentscheidungen selbst bleiben stets bei den validierten, regelbasierten Strategien.");
d("2 · The data — five real datasets","2 · Die Daten — fünf echte Datensätze");
d("A model is only as good as what it learns from, and the challenge specifically rewards <b>working with a large dataset</b>. We assembled five, one per asset class, from real daily market history — over <b>220,000 labelled examples in total</b>. Each row is one real trading day for one real instrument, described by the features in the next section and labelled by what actually happened next.","Ein Modell ist nur so gut wie das, woraus es lernt, und die Challenge belohnt gezielt <b>die Arbeit mit einem großen Datensatz</b>. Wir haben fünf zusammengestellt, einen pro Anlageklasse, aus echter täglicher Markthistorie — insgesamt über <b>220.000 gelabelte Beispiele</b>. Jede Zeile ist ein echter Handelstag für ein echtes Instrument, beschrieben durch die Merkmale im nächsten Abschnitt und gelabelt nach dem, was tatsächlich als Nächstes geschah.");
d("Model","Modell");d("What it learns from","Woraus es lernt");d("History","Historie");d("Examples","Beispiele");
d("Crypto history comes from <b>Coinbase's public API</b> (free, open); the four traditional-asset classes use <b>OANDA</b>, a regulated broker whose data feed gives clean daily prices going back twenty years. We <em>pool</em> every instrument within a class — each coin, index or pair contributes its whole history as training examples — which is what multiplies a few thousand days into tens of thousands of examples and gives each model enough to learn from.","Die Krypto-Historie stammt aus <b>Coinbases öffentlicher API</b> (kostenlos, offen); die vier traditionellen Anlageklassen nutzen <b>OANDA</b>, einen regulierten Broker, dessen Datenfeed saubere Tagespreise zwanzig Jahre zurück liefert. Wir <em>bündeln</em> jedes Instrument innerhalb einer Klasse — jede Münze, jeder Index oder jedes Paar steuert seine ganze Historie als Trainingsbeispiele bei — was aus ein paar tausend Tagen Zehntausende Beispiele macht und jedem Modell genug zum Lernen gibt.");
d("3 · What the model looks at — the 15 features","3 · Worauf das Modell schaut — die 15 Merkmale");
d("A model cannot read a price chart the way a person does. It reads <b>numbers</b>. So for every single day, we compute 15 numbers (\"features\") that describe how that market had been behaving <em>up to that point</em>. These are the model's senses. They fall into five families:","Ein Modell kann ein Kursdiagramm nicht wie ein Mensch lesen. Es liest <b>Zahlen</b>. Für jeden einzelnen Tag berechnen wir also 15 Zahlen (\"Merkmale\"), die beschreiben, wie sich dieser Markt <em>bis zu diesem Zeitpunkt</em> verhalten hat. Das sind die Sinne des Modells. Sie fallen in fünf Familien:");
d("<b>Momentum / returns</b> (4 numbers) — how much the price changed over the last 5, 10, 20 and 60 days. Captures whether the market has been rising, falling, or flat, over roughly a week to three months.","<b>Momentum / Renditen</b> (4 Zahlen) — wie stark sich der Preis über die letzten 5, 10, 20 und 60 Tage verändert hat. Erfasst, ob der Markt gestiegen, gefallen oder seitwärts gelaufen ist, über etwa eine Woche bis drei Monate.");
d("<b>Recent volatility</b> (4 numbers) — how choppy the market has been over the last 5, 10, 20 and 30 days. Because volatility clusters, these are the model's single most powerful clues.","<b>Jüngste Volatilität</b> (4 Zahlen) — wie unruhig der Markt über die letzten 5, 10, 20 und 30 Tage war. Da Volatilität sich häuft, sind dies die stärksten Hinweise des Modells.");
d("<b>Trend position</b> (3 numbers) — how far the price sits above or below its 20-, 50- and 200-day moving averages. A market stretched far from its long-term trend behaves differently from one sitting right on it.","<b>Trendposition</b> (3 Zahlen) — wie weit der Preis über oder unter seinen 20-, 50- und 200-Tage-Durchschnitten liegt. Ein Markt, der weit von seinem langfristigen Trend entfernt ist, verhält sich anders als einer, der genau darauf liegt.");
d("<b>Momentum oscillator</b> (1 number, RSI) — a classic 0–100 gauge of whether a market looks \"overbought\" or \"oversold\".","<b>Momentum-Oszillator</b> (1 Zahl, RSI) — ein klassisches Maß von 0–100 dafür, ob ein Markt \"überkauft\" oder \"überverkauft\" aussieht.");
d("<b>Drawdown &amp; acceleration</b> (3 numbers) — how far below its recent 30- and 60-day peak the market is, plus whether volatility is speeding up or slowing down.","<b>Drawdown &amp; Beschleunigung</b> (3 Zahlen) — wie weit der Markt unter seinem jüngsten 30- und 60-Tage-Hoch liegt, plus ob die Volatilität zu- oder abnimmt.");
d("One rule governs all of them and it is the most important rule in the whole project: <b>every feature uses only information from the past.</b> The feature for a given day is computed strictly from prices up to and including that day — never after. If even a hint of the future leaked into the inputs, the model would look brilliant on paper and be worthless in reality. Getting this \"no lookahead\" discipline right is what separates honest financial machine learning from the kind that quietly cheats.","Eine Regel bestimmt sie alle und sie ist die wichtigste im ganzen Projekt: <b>jedes Merkmal nutzt nur Informationen aus der Vergangenheit.</b> Das Merkmal für einen Tag wird strikt aus Preisen bis einschließlich dieses Tages berechnet — nie danach. Würde auch nur ein Hauch der Zukunft in die Eingaben sickern, sähe das Modell auf dem Papier brillant aus und wäre in der Realität wertlos. Diese \"Kein-Vorausschauen\"-Disziplin richtig hinzubekommen unterscheidet ehrliches Finanz-Machine-Learning von jenem, das heimlich schummelt.");
d("4 · The answer key — how we label \"turbulent\"","4 · Der Lösungsschlüssel — wie wir \"turbulent\" labeln");
d("To teach a model, every example needs a correct answer to learn from. For each day we look at the <em>future</em> — the realised volatility over the following ten trading days — and mark the day <b>turbulent</b> if that future volatility lands in the top third of what that market experienced during the training era, and <b>calm</b> otherwise. Crucially, the cut-off that defines \"top third\" is measured <b>only on the training data</b>, then applied unchanged to later data, so the model is never tuned to the period it is tested on.","Um ein Modell zu lehren, braucht jedes Beispiel eine korrekte Antwort zum Lernen. Für jeden Tag betrachten wir die <em>Zukunft</em> — die realisierte Volatilität über die folgenden zehn Handelstage — und markieren den Tag als <b>turbulent</b>, wenn diese künftige Volatilität im obersten Drittel dessen liegt, was der Markt in der Trainingsära erlebte, sonst als <b>ruhig</b>. Entscheidend: Die Grenze, die das \"oberste Drittel\" definiert, wird <b>nur auf den Trainingsdaten</b> gemessen und dann unverändert auf spätere Daten angewandt, sodass das Modell nie auf den Testzeitraum abgestimmt ist.");
d("5 · How the model was built — and how it actually learns","5 · Wie das Modell gebaut wurde — und wie es wirklich lernt");
d("The model at the heart of each class is a <b>neural network</b> — the exact kind of \"self-learning algorithm\" the challenge names. Here is what that means, concretely.","Das Modell im Kern jeder Klasse ist ein <b>neuronales Netz</b> — genau die Art \"selbstlernender Algorithmus\", die die Challenge nennt. Hier ist, was das konkret bedeutet.");
d("A neural network is a web of simple maths units called <b>neurons</b>, arranged in <b>layers</b>. Ours has an input layer of the 15 features, two hidden layers (of 32 and then 16 neurons), and a single output that becomes the probability of turbulence. Every connection between neurons carries a <b>weight</b> — a number that says how much that input matters. A network with these layers has a few thousand of these weights.","Ein neuronales Netz ist ein Geflecht einfacher Recheneinheiten, genannt <b>Neuronen</b>, angeordnet in <b>Schichten</b>. Unseres hat eine Eingabeschicht mit den 15 Merkmalen, zwei verdeckte Schichten (mit 32 und dann 16 Neuronen) und einen einzelnen Ausgang, der zur Turbulenz-Wahrscheinlichkeit wird. Jede Verbindung zwischen Neuronen trägt ein <b>Gewicht</b> — eine Zahl, die sagt, wie wichtig diese Eingabe ist. Ein Netz mit diesen Schichten hat einige tausend dieser Gewichte.");
d("At the start, all those weights are <b>random</b> — the network knows nothing and guesses blindly. Then the learning happens, and this is the part that makes it \"AI\": we show it a training example (15 features → the true calm/turbulent answer), it makes a guess, we measure how wrong the guess was, and the network <b>nudges every weight a tiny amount in the direction that would have made the guess less wrong</b>. This nudging procedure is called backpropagation with gradient descent. We repeat it across all the training examples, over and over (each full pass is an \"epoch\"), and with every pass the network gets a little better at mapping \"what the market looks like today\" to \"turbulent or calm ahead\". After enough passes it stops improving, and we stop training automatically (this is \"early stopping\", which prevents it from simply memorising). At no point does a human write the rule — the network <b>discovers</b> the relationship in the data by itself.","Am Anfang sind all diese Gewichte <b>zufällig</b> — das Netz weiß nichts und rät blind. Dann geschieht das Lernen, und das ist der Teil, der es zur \"KI\" macht: Wir zeigen ihm ein Trainingsbeispiel (15 Merkmale → die wahre Antwort ruhig/turbulent), es rät, wir messen, wie falsch die Schätzung war, und das Netz <b>schiebt jedes Gewicht ein winziges Stück in die Richtung, die die Schätzung weniger falsch gemacht hätte</b>. Dieses Verfahren heißt Backpropagation mit Gradientenabstieg. Wir wiederholen es über alle Trainingsbeispiele, immer wieder (jeder volle Durchlauf ist eine \"Epoche\"), und mit jedem Durchlauf wird das Netz etwas besser darin, \"wie der Markt heute aussieht\" auf \"turbulent oder ruhig voraus\" abzubilden. Nach genügend Durchläufen verbessert es sich nicht mehr, und wir stoppen das Training automatisch (das ist \"Early Stopping\", das reines Auswendiglernen verhindert). An keinem Punkt schreibt ein Mensch die Regel — das Netz <b>entdeckt</b> den Zusammenhang in den Daten selbst.");
d("A few practical touches make it work honestly and well: we <b>standardise</b> the features (rescale them to a common range) so no single one dominates just because its numbers are bigger; we <b>balance the classes</b> so the model doesn't lazily learn to always say \"calm\" just because calm days are more common; and we tune the exact probability cut-off (how sure it must be before it says \"turbulent\") on a separate validation set, never on the final test. As a cross-check we also train a <b>random forest</b> — a completely different model made of hundreds of decision trees that vote — and keep whichever of the two does better on validation. For four of the five classes the neural network wins or ties, so that is what we deploy.","Ein paar praktische Kniffe lassen es ehrlich und gut funktionieren: Wir <b>standardisieren</b> die Merkmale (skalieren sie auf einen gemeinsamen Bereich), damit keines allein dominiert, nur weil seine Zahlen größer sind; wir <b>balancieren die Klassen</b>, damit das Modell nicht bequem lernt, immer \"ruhig\" zu sagen, nur weil ruhige Tage häufiger sind; und wir justieren die genaue Wahrscheinlichkeitsgrenze (wie sicher es sein muss, bevor es \"turbulent\" sagt) auf einem separaten Validierungssatz, nie am finalen Test. Als Gegenprobe trainieren wir auch einen <b>Random Forest</b> — ein völlig anderes Modell aus hunderten abstimmenden Entscheidungsbäumen — und behalten, welches der beiden in der Validierung besser abschneidet. Bei vier der fünf Klassen gewinnt das neuronale Netz oder liegt gleichauf, also setzen wir dieses ein.");
d("6 · Testing it honestly","6 · Ehrlich testen");
d("The single easiest way to fool yourself in machine learning is to test a model on data it already studied. So we split each dataset <b>by time, never randomly</b>: the model trains on the oldest ~60% of history, we tune choices on the next ~15% (validation), and we grade it once on the most recent ~25% — <b>data it never saw during training</b>. Splitting by time matters because shuffling the days would let tomorrow leak into yesterday.","Der einfachste Weg, sich beim maschinellen Lernen selbst zu täuschen, ist, ein Modell an Daten zu testen, die es bereits studiert hat. Deshalb teilen wir jeden Datensatz <b>nach Zeit, nie zufällig</b>: Das Modell trainiert auf den ältesten ~60% der Historie, wir justieren Entscheidungen auf den nächsten ~15% (Validierung) und bewerten es einmal auf den jüngsten ~25% — <b>Daten, die es im Training nie sah</b>. Die Aufteilung nach Zeit ist wichtig, denn ein Mischen der Tage ließe das Morgen ins Gestern sickern.");
d("We grade with a metric called <b>AUC</b>, which runs from 0.50 (a coin-flip — no skill at all) to 1.00 (perfect). AUC is the right choice here because it isn't fooled by the fact that calm days outnumber turbulent ones. We always compare against two deliberately dumb baselines: \"always guess calm\", and the strong one — \"volatility is already high, so expect more\" (persistence). Beating persistence is the real test, because volatility clustering makes it a genuinely tough benchmark.","Wir bewerten mit einer Kennzahl namens <b>AUC</b>, die von 0,50 (Münzwurf — keinerlei Können) bis 1,00 (perfekt) reicht. AUC ist hier die richtige Wahl, weil sie sich nicht davon täuschen lässt, dass ruhige Tage die turbulenten übertreffen. Wir vergleichen stets mit zwei bewusst simplen Referenzen: \"immer ruhig raten\" und der starken — \"Volatilität ist schon hoch, also erwarte mehr\" (Persistenz). Persistenz zu schlagen ist der eigentliche Test, denn das Volatilitäts-Clustering macht sie zu einer wirklich harten Messlatte.");
d("7 · The results — and a real finding","7 · Die Ergebnisse — und ein echter Befund");
d("On data none of the models had ever seen, every one beats both baselines. Bond and rate turbulence turned out the most forecastable (<b>AUC 0.93</b>), then commodities (0.85), equities (0.82) and FX (0.81); crypto — the wildest, most headline-driven market — is the hardest (0.70). The live scorecard above shows all of this, including the <b>edge</b> column, which is the honest measure of what our machine learning adds <em>beyond</em> simply noticing that volatility is already high.","Auf Daten, die keines der Modelle je gesehen hatte, schlägt jedes beide Referenzen. Turbulenz bei Anleihen und Zinsen erwies sich als am besten vorhersagbar (<b>AUC 0,93</b>), dann Rohstoffe (0,85), Aktien (0,82) und Devisen (0,81); Krypto — der wildeste, am stärksten von Schlagzeilen getriebene Markt — ist am schwersten (0,70). Die Live-Scorecard oben zeigt all das, inklusive der Spalte <b>Edge</b>, dem ehrlichen Maß dafür, was unser maschinelles Lernen <em>über</em> das bloße Bemerken bereits hoher Volatilität hinaus beiträgt.");
d("Running one identical method across five markets produced a genuine cross-asset insight: <b>the steadier the market, the more predictable its turbulence.</b> That makes economic sense — bond volatility is driven by slow, telegraphed forces like the interest-rate cycle, whereas crypto turbulence is often set off by sudden, unforecastable news. Discovering that by experiment, rather than assuming it, is exactly what a good study should do.","Dieselbe Methode über fünf Märkte hinweg lieferte eine echte anlageklassenübergreifende Erkenntnis: <b>je ruhiger der Markt, desto vorhersagbarer seine Turbulenz.</b> Das ergibt ökonomisch Sinn — Anleihe-Volatilität wird von langsamen, angekündigten Kräften wie dem Zinszyklus getrieben, während Krypto-Turbulenz oft von plötzlichen, unvorhersehbaren Nachrichten ausgelöst wird. Das per Experiment zu entdecken, statt es anzunehmen, ist genau das, was eine gute Studie tun sollte.");
d("8 · How it runs live in this app","8 · Wie es in dieser App live läuft");
d("Once trained, each model is saved to disk as a small file — its learned \"brain\". When you open this page, the app takes the most recent real prices for a representative market in each class (Bitcoin, the S&amp;P 500, the US 10-year, gold, EUR/USD), computes today's 15 features, feeds them through the trained network, and reads out a fresh turbulence probability. That is the Calm / Turbulent call you see at the top of the page, updated from live data.","Einmal trainiert, wird jedes Modell als kleine Datei auf der Festplatte gespeichert — sein gelerntes \"Gehirn\". Wenn du diese Seite öffnest, nimmt die App die jüngsten echten Preise für einen repräsentativen Markt jeder Klasse (Bitcoin, den S&amp;P 500, die US-10-Jährige, Gold, EUR/USD), berechnet die 15 Merkmale von heute, führt sie durch das trainierte Netz und liest eine frische Turbulenz-Wahrscheinlichkeit aus. Das ist die Einschätzung Ruhig / Turbulent oben auf der Seite, aktualisiert aus Live-Daten.");
d("9 · Honesty &amp; limitations","9 · Ehrlichkeit &amp; Grenzen");
d("We say plainly what these models are not. They are a <b>modest, real edge — not a crystal ball</b>. They miss sudden shocks that have no footprint in past prices, they raise occasional false alarms, and the high headline AUCs owe a lot to volatility simply being persistent (the model's own contribution is the smaller \"edge\"). They forecast how <em>bumpy</em> the ride will be, not which way prices go, and they inform risk rather than making a single trade. Every one of those caveats is stated because a model you can honestly critique is worth far more than an impressive number you cannot defend — which, in the end, is the whole point of the project.","Wir sagen klar, was diese Modelle nicht sind. Sie sind ein <b>bescheidener, echter Vorteil — keine Kristallkugel</b>. Sie verpassen plötzliche Schocks, die keine Spur in vergangenen Preisen haben, sie lösen gelegentlich Fehlalarme aus, und die hohen AUC-Zahlen verdanken sich stark der bloßen Persistenz der Volatilität (der eigene Beitrag des Modells ist der kleinere \"Edge\"). Sie prognostizieren, wie <em>holprig</em> die Fahrt wird, nicht in welche Richtung die Preise gehen, und sie informieren das Risiko, statt einen einzelnen Trade zu machen. Jeder dieser Vorbehalte wird genannt, weil ein Modell, das man ehrlich kritisieren kann, weit mehr wert ist als eine beeindruckende Zahl, die man nicht verteidigen kann — was letztlich der ganze Sinn des Projekts ist.");

/* ---- sentiment / analog / positioning ---- */
d("02 — Sentiment scanner","02 — Sentiment-Scanner");
d("The scanner reads the market's mood","Der Scanner liest die Stimmung des Marktes");
d("An AI engine scans hundreds of news and social sources, scores each asset class, explains itself — and trims exposure in your Blackbox when it turns strongly bearish on a strategy's assets.","Eine KI-Engine durchsucht hunderte Nachrichten- und Social-Quellen, bewertet jede Anlageklasse, erklärt sich — und reduziert das Exposure in deiner Blackbox, wenn sie bei den Anlagen einer Strategie stark bärisch wird.");
d("<b>Fetch.</b> It pulls hundreds of real, current items — news headlines (Google News) and social posts (Reddit) — across six asset classes: equities, gold, rates, FX, crypto, oil.","<b>Abrufen.</b> Sie zieht hunderte echte, aktuelle Beiträge — Nachrichtenschlagzeilen (Google News) und Social-Posts (Reddit) — über sechs Anlageklassen: Aktien, Gold, Zinsen, Devisen, Krypto, Öl.");
d("<b>Read.</b> An AI model scores each headline's mood for that asset, from −100 (very bearish) to +100 (very bullish). It reads for real meaning — handling context and negation — not just spotting keywords.","<b>Lesen.</b> Ein KI-Modell bewertet die Stimmung jeder Schlagzeile für diese Anlage, von −100 (sehr bärisch) bis +100 (sehr bullish). Es liest nach echter Bedeutung — mit Kontext und Verneinung — nicht nur nach Stichwörtern.");
d("<b>Aggregate.</b> The scores combine into one mood reading per asset, with a plain-English explanation you can trace back to the actual headlines that drove it.","<b>Aggregieren.</b> Die Bewertungen fließen zu einem Stimmungswert je Anlage zusammen, mit einer verständlichen Erklärung, die sich auf die tatsächlichen Schlagzeilen zurückführen lässt.");
d("<b>Act.</b> This is where AI <b>supports the strategies</b>: if a strategy's assets score strongly bearish (below −40), that strategy's exposure is automatically trimmed to 75% — a risk brake on top of the validated rule.","<b>Handeln.</b> Hier <b>unterstützt die KI die Strategien</b>: Wenn die Anlagen einer Strategie stark bärisch bewertet werden (unter −40), wird deren Exposure automatisch auf 75% reduziert — eine Risikobremse zusätzlich zur validierten Regel.");
d("03 — Historical analog","03 — Historisches Analogon");
d("Where has the market seen this before?","Wo hat der Markt das schon einmal erlebt?");
d("AI reads today's real headlines and matches them to the closest event from a fixed library of real past market events — then shows how each strategy's real, backtested returns actually moved in the months after. AI only picks which past event fits; every number comes from the same 20-year backtests as the rest of the app.","Die KI liest die heutigen echten Schlagzeilen und ordnet sie dem ähnlichsten Ereignis aus einer festen Bibliothek realer vergangener Marktereignisse zu — und zeigt dann, wie sich die echten, backgetesteten Renditen jeder Strategie in den Monaten danach tatsächlich entwickelten. Die KI wählt nur, welches vergangene Ereignis passt; jede Zahl stammt aus denselben 20-Jahres-Backtests wie der Rest der App.");
d("<b>Read today.</b> It takes the top real headlines the scanner just collected.","<b>Heute lesen.</b> Sie nimmt die wichtigsten echten Schlagzeilen, die der Scanner gerade gesammelt hat.");
d("<b>Match.</b> An AI model compares them to a fixed library of real past market events (the 2008 crash, the Taper Tantrum, COVID, the 2022 rate-hike cycle, and more) and picks the single closest analog.","<b>Zuordnen.</b> Ein KI-Modell vergleicht sie mit einer festen Bibliothek realer vergangener Marktereignisse (dem Crash 2008, dem Taper Tantrum, COVID, dem Zinserhöhungszyklus 2022 und mehr) und wählt das ähnlichste Analogon.");
d("<b>Look up real history.</b> The app then pulls how each strategy's <b>real, backtested</b> returns actually moved in the months after that past event.","<b>Echte Historie nachschlagen.</b> Die App ruft dann ab, wie sich die <b>echten, backgetesteten</b> Renditen jeder Strategie in den Monaten nach diesem Ereignis tatsächlich entwickelten.");
d("<b>Honest split.</b> AI only chooses <em>which</em> past event fits — it never invents an outcome. Every number shown comes from the real 20-year backtests.","<b>Ehrliche Trennung.</b> Die KI wählt nur, <em>welches</em> vergangene Ereignis passt — sie erfindet nie ein Ergebnis. Jede gezeigte Zahl stammt aus den echten 20-Jahres-Backtests.");
d("04 — AI positioning desk","04 — KI-Positionierungs-Desk");
d("How AI is supporting each strategy right now","Wie die KI jede Strategie gerade unterstützt");
d("Each strategy's position is decided by its validated rule. On top of that, the AI reads the live news sentiment for that strategy's assets and acts as a risk brake — trimming exposure to 75% when the mood turns strongly bearish. Below, AI explains each strategy's current stance and how the overlay is supporting it. The rule is the driver; the AI is the co-pilot.","Die Position jeder Strategie wird von ihrer validierten Regel bestimmt. Darüber hinaus liest die KI das Live-Nachrichten-Sentiment für die Anlagen dieser Strategie und wirkt als Risikobremse — reduziert das Exposure auf 75%, wenn die Stimmung stark bärisch wird. Unten erklärt die KI die aktuelle Ausrichtung jeder Strategie und wie das Overlay sie unterstützt. Die Regel ist der Fahrer; die KI ist der Co-Pilot.");
d("<b>Read the rule.</b> For each of the eight strategies, it takes today's rule-based stance — what the strategy holds right now, and why (e.g. \"gold uptrend + falling yields → fully long gold\").","<b>Die Regel lesen.</b> Für jede der acht Strategien nimmt sie die heutige regelbasierte Ausrichtung — was die Strategie gerade hält und warum (z. B. \"Gold-Aufwärtstrend + fallende Renditen → voll long Gold\").");
d("<b>Read the mood.</b> It pulls the live sentiment score (from system 2) for that strategy's assets.","<b>Die Stimmung lesen.</b> Sie ruft den Live-Sentiment-Wert (aus System 2) für die Anlagen dieser Strategie ab.");
d("<b>Check the brake.</b> It checks whether the sentiment risk-overlay is currently trimming that strategy's exposure.","<b>Die Bremse prüfen.</b> Sie prüft, ob das Sentiment-Risiko-Overlay gerade das Exposure dieser Strategie reduziert.");
d("<b>Explain.</b> An AI model writes a short, plain-English note on the strategy's current position and how the AI overlay is supporting or easing off it. The validated rule always makes the call; the AI narrates and, within strict limits, brakes.","<b>Erklären.</b> Ein KI-Modell schreibt eine kurze, verständliche Notiz zur aktuellen Position der Strategie und wie das KI-Overlay sie unterstützt oder zurücknimmt. Die validierte Regel entscheidet immer; die KI erzählt und bremst innerhalb strenger Grenzen.");

/* ---- briefing ---- */
d("Daily briefing","Tägliches Briefing");
d("Today's market, read by AI","Der heutige Markt, gelesen von der KI");
d("Every cycle, the AI scanner reads hundreds of real news and social sources and scores six asset classes. This briefing synthesises that live read — plus the closest historical analog — into a plain-English summary of the day. It describes the mood and what it means for a diversified systematic portfolio; it is not advice, and it never invents a number.","In jedem Zyklus liest der KI-Scanner hunderte echte Nachrichten- und Social-Quellen und bewertet sechs Anlageklassen. Dieses Briefing fasst diese Live-Einschätzung — plus das ähnlichste historische Analogon — zu einer verständlichen Tageszusammenfassung. Es beschreibt die Stimmung und was sie für ein diversifiziertes systematisches Portfolio bedeutet; es ist keine Beratung und erfindet nie eine Zahl.");
d("Regenerated periodically from the live sentiment scan. Educational synthesis only — every strategy still trades on its validated rules.","Regelmäßig aus dem Live-Sentiment-Scan neu erzeugt. Nur eine Bildungszusammenfassung — jede Strategie handelt weiterhin nach ihren validierten Regeln.");

/* ---- learn ---- */
d("Understand what you're investing in","Verstehe, worin du investierst");
d("Systematic investing has its own vocabulary. Tap any term below for a plain-English explanation from the AI, or ask the tutor anything — how a strategy works, what a metric means, why diversification lowers risk. It's grounded in this app's own real strategies, and it never gives buy/sell advice.","Systematisches Investieren hat sein eigenes Vokabular. Tippe unten auf einen Begriff für eine verständliche Erklärung der KI, oder frag den Tutor alles — wie eine Strategie funktioniert, was eine Kennzahl bedeutet, warum Diversifikation das Risiko senkt. Es basiert auf den echten Strategien dieser App und gibt nie Kauf-/Verkaufsempfehlungen.");
d("Key terms — tap to explain","Wichtige Begriffe — zum Erklären antippen");
d("Ask the AI tutor","Frag den KI-Tutor");
d("Ask","Fragen");

/* ---- AI Inside page ---- */
d("How it works","Wie es funktioniert");
d("The AI inside Blackbox","Die KI in Blackbox");
d("Blackbox is built on eight strategies validated over 20 years of real market data. Artificial intelligence is layered through the whole product — but in a very deliberate way. This page explains exactly what the AI does, how it's implemented, and, just as importantly, what it does <em>not</em> do.","Blackbox basiert auf acht Strategien, validiert über 20 Jahre echter Marktdaten. Künstliche Intelligenz ist durch das ganze Produkt geschichtet — aber sehr bewusst. Diese Seite erklärt genau, was die KI tut, wie sie umgesetzt ist und, ebenso wichtig, was sie <em>nicht</em> tut.");
d("The rule is the driver. The AI is the co-pilot.","Die Regel ist der Fahrer. Die KI ist der Co-Pilot.");
d("Every buy, sell and hold decision comes from a validated, backtested rule — not from AI. The AI reads the market, supports those rules with a bounded risk brake, personalises your mix within strict limits, and explains everything in plain language. It never trades on its own, and it never invents a number.","Jede Kauf-, Verkaufs- und Halteentscheidung stammt aus einer validierten, backgetesteten Regel — nicht von der KI. Die KI liest den Markt, unterstützt diese Regeln mit einer begrenzten Risikobremse, personalisiert deinen Mix innerhalb strenger Grenzen und erklärt alles in klarer Sprache. Sie handelt nie von sich aus und erfindet nie eine Zahl.");
d("0 · The AI we built ourselves — Regime Radar","0 · Die KI, die wir selbst gebaut haben — Regime Radar");
d("Five neural networks we trained from scratch","Fünf neuronale Netze, die wir von Grund auf trainiert haben");
d("1 · The AI reads the market","1 · Die KI liest den Markt");
d("Live sentiment scanner","Live-Sentiment-Scanner");
d("Every cycle, an AI model reads hundreds of real news headlines and social posts across six asset classes — equities, gold, rates, FX, crypto, oil — and scores the mood from −100 to +100, with a plain-English explanation traceable to the actual headlines. It reads for real meaning (handling context and negation), not just keywords.","In jedem Zyklus liest ein KI-Modell hunderte echte Nachrichtenschlagzeilen und Social-Posts über sechs Anlageklassen — Aktien, Gold, Zinsen, Devisen, Krypto, Öl — und bewertet die Stimmung von −100 bis +100, mit einer verständlichen, auf die tatsächlichen Schlagzeilen zurückführbaren Erklärung. Es liest nach echter Bedeutung (Kontext und Verneinung), nicht nur nach Stichwörtern.");
d("Historical analog","Historisches Analogon");
d("AI matches today's real headlines to the closest event from a fixed library of real past market events — the 2008 crash, the Taper Tantrum, COVID, and more — then shows how each strategy's <em>real, backtested</em> returns actually moved in the months after. AI only picks which past event fits; every number is from the backtests.","Die KI ordnet die heutigen echten Schlagzeilen dem ähnlichsten Ereignis aus einer festen Bibliothek realer vergangener Marktereignisse zu — dem Crash 2008, dem Taper Tantrum, COVID und mehr — und zeigt dann, wie sich die <em>echten, backgetesteten</em> Renditen jeder Strategie in den Monaten danach entwickelten. Die KI wählt nur das passende Ereignis; jede Zahl stammt aus den Backtests.");
d("2 · The AI supports the strategies","2 · Die KI unterstützt die Strategien");
d("Sentiment risk overlay","Sentiment-Risiko-Overlay");
d("This is where AI directly supports each strategy. When the scanner reads a strategy's assets as strongly bearish (a score below −40), that strategy's exposure is automatically trimmed to 75% — a risk brake layered on top of the validated rule. The rule decides the position; the AI decides whether to ease off it.","Hier unterstützt die KI jede Strategie direkt. Liest der Scanner die Anlagen einer Strategie als stark bärisch (ein Wert unter −40), wird deren Exposure automatisch auf 75% reduziert — eine Risikobremse über der validierten Regel. Die Regel entscheidet die Position; die KI entscheidet, ob sie sie zurücknimmt.");
d("AI positioning desk","KI-Positionierungs-Desk");
d("For every strategy, the AI reads its live rule-based stance and the live sentiment on its assets, then writes a short note explaining where it's positioned right now and whether the risk overlay is supporting or trimming it. You can see this on the Intelligence page.","Für jede Strategie liest die KI ihre live regelbasierte Ausrichtung und das Live-Sentiment ihrer Anlagen und schreibt dann eine kurze Notiz, wo sie gerade positioniert ist und ob das Risiko-Overlay sie unterstützt oder reduziert. Das siehst du auf der Intelligenz-Seite.");
d("3 · The AI personalises — within limits","3 · Die KI personalisiert — in Grenzen");
d("Free-text quiz nudge","Freitext-Anstoß im Quiz");
d("After the six questions, you can describe your situation in your own words. AI reads it and may apply a <em>small, bounded</em> adjustment to your mix — never more than a gentle tilt, never adding or removing a strategy, never overriding the validated math. And it always explains, in writing, what in your words drove the change.","Nach den sechs Fragen kannst du deine Situation in eigenen Worten beschreiben. Die KI liest es und wendet womöglich eine <em>kleine, begrenzte</em> Anpassung auf deinen Mix an — nie mehr als eine sanfte Neigung, nie eine Strategie hinzufügend oder entfernend, nie die validierte Mathematik überschreibend. Und sie erklärt immer schriftlich, was in deinen Worten die Änderung auslöste.");
d("AI risk review","KI-Risikoprüfung");
d("Once your portfolio is live, AI reviews your actual mix — how diversified or concentrated it is, and its blend of return and risk — reading the exact figures for you. The blended numbers are computed precisely in code; the AI only puts them into words.","Sobald dein Portfolio live ist, prüft die KI deinen tatsächlichen Mix — wie diversifiziert oder konzentriert er ist und seine Mischung aus Rendite und Risiko — und liest die genauen Zahlen für dich. Die gemischten Zahlen werden präzise im Code berechnet; die KI fasst sie nur in Worte.");
d("4 · The AI explains &amp; teaches","4 · Die KI erklärt &amp; lehrt");
d("Daily briefing","Tägliches Briefing");
d("A plain-English summary of the day's market, synthesised from the live sentiment scan and the historical analog.","Eine verständliche Zusammenfassung des Tagesmarkts, erzeugt aus dem Live-Sentiment-Scan und dem historischen Analogon.");
d("Portfolio narration &amp; Q&amp;A","Portfolio-Erzählung &amp; Fragen");
d("Your Blackbox writes its own summary, and you can ask it questions — answered only from your real portfolio data.","Deine Blackbox schreibt ihre eigene Zusammenfassung, und du kannst ihr Fragen stellen — beantwortet nur aus deinen echten Portfoliodaten.");
d("Tutor &amp; glossary","Tutor &amp; Glossar");
d("An AI tutor answers investing questions grounded in this app's own strategies, and any metric can be explained on tap.","Ein KI-Tutor beantwortet Anlagefragen auf Basis der Strategien dieser App, und jede Kennzahl lässt sich auf Tippen erklären.");
d("Strategy deep-dives &amp; compare","Strategie-Tiefblicke &amp; Vergleich");
d("AI explains why any strategy works, and talks you through the trade-offs when you compare two — always from the real numbers.","Die KI erklärt, warum eine Strategie funktioniert, und führt dich durch die Abwägungen, wenn du zwei vergleichst — immer aus den echten Zahlen.");
d("It does not make trades on its own.","Sie macht keine Trades von sich aus.");
d("It does not \"learn\" from your data or rewrite the strategies.","Sie \"lernt\" nicht aus deinen Daten und schreibt die Strategien nicht um.");
d("It never invents a performance number.","Sie erfindet nie eine Performance-Zahl.");
d("It does not give personalised buy/sell advice.","Sie gibt keine persönlichen Kauf-/Verkaufsempfehlungen.");
d("Under the hood","Unter der Haube");
d("All of the above runs through a single shared connector to a large language model (Llama, served by Groq), behind one API key. Every AI feature \"fails open\": if the model is ever unavailable, the app keeps working and the feature simply says so — the validated strategies and every real number are never affected. That's the whole design: real, tested rules at the core; AI as the analyst, risk co-pilot, and teacher around them.","All das läuft über einen einzigen gemeinsamen Konnektor zu einem großen Sprachmodell (Llama, bereitgestellt von Groq), hinter einem API-Schlüssel. Jede KI-Funktion \"fällt offen aus\": Ist das Modell einmal nicht verfügbar, arbeitet die App weiter und die Funktion sagt es einfach — die validierten Strategien und jede echte Zahl bleiben unberührt. Das ist das ganze Design: echte, getestete Regeln im Kern; die KI als Analyst, Risiko-Co-Pilot und Lehrer darum herum.");

/* ---- compare ---- */
d("Compare","Vergleichen");
d("Put two strategies side by side","Stelle zwei Strategien nebeneinander");
d("Pick any two of the eight strategies to compare their real 20-year statistics head to head — return, risk, drawdown and how independent each is from the stock market — then let the AI talk you through the trade-offs. Every number is from the same backtests as the rest of the app.","Wähle zwei der acht Strategien, um ihre echten 20-Jahres-Statistiken direkt zu vergleichen — Rendite, Risiko, Drawdown und wie unabhängig jede vom Aktienmarkt ist — und lass dich dann von der KI durch die Abwägungen führen. Jede Zahl stammt aus denselben Backtests wie der Rest der App.");
d("Strategy A","Strategie A");d("Strategy B","Strategie B");

/* ---- account chrome / auth / footer ---- */
d("Syncing your Blackbox with live markets…","Deine Blackbox wird mit den Live-Märkten synchronisiert…");
d("Your Blackbox account","Dein Blackbox-Konto");
d("Log back in from any device. Practice capital only — no real money.","Melde dich von jedem Gerät wieder an. Nur Übungskapital — kein echtes Geld.");
d("Create my account","Konto erstellen");
d("I have a recovery key","Ich habe einen Wiederherstellungsschlüssel");
d("Create account →","Konto erstellen →");
d("This key <b style=\"color:var(--txt)\">is your login</b>. Save it now — it's shown only once.","Dieser Schlüssel <b style=\"color:var(--txt)\">ist dein Login</b>. Speichere ihn jetzt — er wird nur einmal gezeigt.");
d("Copy key","Schlüssel kopieren");
d("I saved it — continue →","Gespeichert — weiter →");
d("Log in →","Anmelden →");

/* ---- dynamic status / labels ---- */
d("Reading today's headlines against market history…","Die heutigen Schlagzeilen werden mit der Markthistorie abgeglichen…");
d("Historical-analog reading isn't available right now — it needs an AI key configured on the server.","Das historische Analogon ist gerade nicht verfügbar — es benötigt einen auf dem Server konfigurierten KI-Schlüssel.");
d("AI is comparing the two…","Die KI vergleicht die beiden…");
d("AI verdict","KI-Urteil");
d("Reading your portfolio…","Dein Portfolio wird gelesen…");
d("Analysing your mix's diversification and risk…","Diversifikation und Risiko deines Mixes werden analysiert…");
d("Portfolio value","Portfoliowert");d("Your mix","Dein Mix");d("AI summary","KI-Zusammenfassung");
d("🧠 AI risk review","🧠 KI-Risikoprüfung");d("Ask your Blackbox","Frag deine Blackbox");
d("🖨 Statement","🖨 Kontoauszug");d("Change strategy","Strategie ändern");
d("You don't have a Blackbox yet","Du hast noch keine Blackbox");
d("Answer six questions or build your own mix to start a personal paper portfolio.","Beantworte sechs Fragen oder stelle deinen eigenen Mix zusammen, um ein persönliches Papier-Portfolio zu starten.");
d("Start my Blackbox →","Meine Blackbox starten →");

const SEL='nav a[data-page],.kicker,.sub,h1,h2,h3,h4,summary,p,li,.lead,.sec-k,.corebar,.tag2,.pbig,.ql,.chip,button,label,th,td,option,.cmpvs,.cmplabel,.note,.airole-h,.l';
let _obs=null;
function _tr(el){
  if(!el||el.nodeType!==1)return;
  const cur=el.innerHTML; if(cur==null)return;
  const k=tnorm(cur);
  if(LANG==='de'){ if(DE[k]!==undefined&&cur!==DE[k])el.innerHTML=DE[k]; }
  else{ if(REV[k]!==undefined&&cur!==REV[k])el.innerHTML=REV[k]; }
}
function buildHero(){
  const hl=$('#hl');if(!hl)return;
  const w=LANG==='de'?[['Dein'],['eigenes'],['Blackbox.'],['Live','grad'],['in'],['Minuten.','grad']]
                     :[['Your'],['own'],['Blackbox.'],['Live','grad'],['in'],['minutes.','grad']];
  hl.innerHTML=w.map((x,i)=>`<span class="w ${x[1]||''}" style="animation-delay:${i*90}ms">${x[0]}&nbsp;</span>`).join('');
}
function applyLang(){
  if(_obs)_obs.disconnect();
  try{
    document.documentElement.setAttribute('lang',LANG);
    document.querySelectorAll(SEL).forEach(_tr);
    buildHero();
    PH.forEach(([id,en,de])=>{const el=document.getElementById(id);if(el)el.setAttribute('placeholder',LANG==='de'?de:en);});
    const lt=$('#langtog');if(lt)lt.textContent=LANG==='de'?'EN':'DE';
  }finally{ if(_obs)_obs.observe(document.body,{childList:true,subtree:true,characterData:true}); }
}
function toggleLang(){
  LANG=LANG==='de'?'en':'de';localStorage.setItem('bb_lang',LANG);
  applyLang();
  try{const p=(typeof pageFromPath==='function')?pageFromPath():'home';
      if(typeof onPageShown==='function')onPageShown(p);}catch(e){}
}
_obs=new MutationObserver(()=>{ if(LANG==='en')return; window.requestAnimationFrame(()=>{ if(_obs)_obs.disconnect(); try{document.querySelectorAll(SEL).forEach(_tr);}finally{ if(_obs)_obs.observe(document.body,{childList:true,subtree:true,characterData:true}); } }); });

boot();
window.requestAnimationFrame(applyLang);
</script></body></html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)
