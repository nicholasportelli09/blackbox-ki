"""
USER ENGINE — gives every app user their own live paper Blackbox.

For each catalog strategy it derives TODAY's holdings from the same validated
rules (200dMA trends, seasonal window, rates gate, crypto trend) using live OANDA/
Coinbase data, then marks a user's chosen mix to market:
    user day-return = sum_s weight_s * scanner_tilt_s * strategy_day_return_s
Scanner tilt: if the AI sentiment for a strategy's asset bucket is strongly
negative (< -40), that strategy's exposure is scaled 0.75x — the ONE place the
scanner acts, clearly labelled in the UI. Everything cached so the app stays fast.
"""
import os, time
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
_cache = {}


def _c(key, ttl, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    v = fn()
    _cache[key] = (now, v)
    return v


NAMES = {"XAU_USD": "Gold", "XAG_USD": "Silver", "XPT_USD": "Platinum", "XPD_USD": "Palladium",
         "XCU_USD": "Copper", "WTICO_USD": "WTI Oil", "BCO_USD": "Brent Oil", "NATGAS_USD": "Nat Gas",
         "CORN_USD": "Corn", "WHEAT_USD": "Wheat", "SOYBN_USD": "Soybeans", "SUGAR_USD": "Sugar",
         "SPX500_USD": "S&P 500", "NAS100_USD": "Nasdaq 100", "US30_USD": "Dow 30", "DE30_EUR": "DAX 40",
         "UK100_GBP": "FTSE 100", "JP225_USD": "Nikkei 225", "EU50_EUR": "Euro Stoxx 50", "AU200_AUD": "ASX 200",
         "USB02Y_USD": "US 2Y Note", "USB05Y_USD": "US 5Y Note", "USB10Y_USD": "US 10Y Note",
         "EUR_USD": "EUR/USD", "USD_JPY": "USD/JPY", "GBP_USD": "GBP/USD", "AUD_USD": "AUD/USD"}

INDICES = ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR", "UK100_GBP", "JP225_USD", "EU50_EUR", "AU200_AUD"]
COMMOD = ["XAU_USD", "XAG_USD", "XPT_USD", "XPD_USD", "XCU_USD", "WTICO_USD", "BCO_USD",
          "NATGAS_USD", "CORN_USD", "WHEAT_USD", "SOYBN_USD", "SUGAR_USD"]
RP_BASKET = INDICES[:4] + ["XAU_USD", "WTICO_USD", "USB02Y_USD", "USB05Y_USD", "USB10Y_USD",
                           "EUR_USD", "USD_JPY", "AUD_USD"]
HALLOWEEN = {11, 12, 1, 2, 3, 4}

# which sentiment buckets each strategy lives in (scanner "utilised in every strategy")
SENT_BUCKETS = {
    "stack_bal": ["equities", "gold", "oil"], "stack_agg": ["equities", "gold", "oil"],
    "equity_season": ["equities"], "gold_macro": ["gold", "rates"],
    "commod_trend": ["oil", "gold"], "risk_parity": ["equities", "rates", "fx"],
    "crypto_trend": ["crypto"], "spx_hold": ["equities"],
}


def _client():
    from oanda_client import OandaClient
    return OandaClient()


def _daily(inst):
    return _c("d_" + inst, 1800, lambda: _client().get_candles(inst, "D", count=230))


def _px(inst):
    return _c("p_" + inst, 420, lambda: _client().get_price(inst))


def _above_ma(inst, ma):
    d = _daily(inst)["close"]
    return len(d) >= ma and float(d.iloc[-1]) > float(d.rolling(ma).mean().iloc[-1])


def _dayret(inst):
    d = _daily(inst)["close"]
    return float(_px(inst) / float(d.iloc[-1]) - 1)


def _crypto_state():
    def f():
        from crypto_bot import PRODUCTS, fetch_daily, spot
        import numpy as np
        out = {}
        for p in PRODUCTS:
            s = fetch_daily(p)
            if s is None or len(s) < 205:
                continue
            on = bool(s.iloc[-1] > s.rolling(200).mean().iloc[-1])
            vol = float(s.pct_change().tail(60).std() * (365 ** 0.5))
            sp = spot(p) or float(s.iloc[-1])
            out[p] = {"on": on, "vol": vol, "ret": float(sp / float(s.iloc[-1]) - 1)}
        return out
    return _c("crypto", 1200, f)


def _holdings(sid):
    """Today's target weights {inst: weight} for one strategy, per its rules."""
    if sid == "spx_hold":
        return {"SPX500_USD": 1.0}, "Always fully invested in the index."
    if sid == "equity_season":
        month = datetime.now(timezone.utc).month
        if month not in HALLOWEEN:
            return {}, "Out of the Nov-Apr seasonal window — sitting in cash."
        act = [i for i in INDICES if _above_ma(i, 200)]
        return ({i: 1 / len(INDICES) for i in act},
                f"In season; {len(act)}/{len(INDICES)} indices above their 200-day trend.")
    if sid == "gold_macro":
        gate = _above_ma("USB10Y_USD", 100) and _above_ma("XAU_USD", 200)
        if gate:
            return {"XAU_USD": 1.0}, "Gold uptrend + falling yields — fully long gold."
        return {}, "Rates gate off (yields rising) or gold below trend — in cash."
    if sid == "commod_trend":
        act = [i for i in COMMOD if _above_ma(i, 200)]
        return ({i: 1 / len(COMMOD) for i in act},
                f"{len(act)}/{len(COMMOD)} commodities in uptrends.")
    if sid == "risk_parity":
        import numpy as np
        w = {}
        for i in RP_BASKET:
            d = _daily(i)["close"].pct_change().tail(63)
            v = float(d.std() * np.sqrt(252))
            if v > 0:
                w[i] = 1 / v
        tot = sum(w.values())
        return ({i: 0.7 * x / tot for i, x in w.items()},
                "Owns a representative global basket, weighted to equal risk.")
    if sid in ("stack_bal", "stack_agg"):
        mult = 1.0 if sid == "stack_bal" else 1.8
        parts = ["equity_season", "commod_trend", "gold_macro"]
        merged, notes = {}, []
        for p in parts:
            hw, note = _holdings(p)
            for i, x in hw.items():
                merged[i] = merged.get(i, 0) + mult * x / 3
            notes.append(note)
        return merged, "Blend of three edges — " + " ".join(notes)
    if sid == "crypto_trend":
        cs = _crypto_state()
        on = {p: v for p, v in cs.items() if v["on"] and v["vol"] > 0}
        if not on:
            return {}, "Every coin below its 200-day trend — 100% cash (crypto winter)."
        inv = {p: 1 / v["vol"] for p, v in on.items()}
        tot = sum(inv.values())
        avg_vol = sum(v["vol"] for v in on.values()) / len(on)
        gross = min(1.0, 0.35 / avg_vol)
        return ({p: gross * x / tot for p, x in inv.items()},
                f"{len(on)} coins in uptrends; {int(gross*100)}% deployed.")
    return {}, "Unknown strategy."


def _strategy_dayret(sid, holdings):
    cs = _crypto_state() if sid == "crypto_trend" else None
    r = 0.0
    for inst, w in holdings.items():
        try:
            r += w * (cs[inst]["ret"] if cs else _dayret(inst))
        except Exception:
            continue
    return r


def _close_ret(inst):
    """Yesterday-close -> today-close return from COMPLETED daily candles.
    (After 21:00 UTC today's OANDA candle is complete, so [-1]/[-2] is the full
    trading day just ended — the nightly settle must use THIS, not spot-vs-close
    day_ret, which collapses to ~0 right after the close.)"""
    d = _c("dc_" + inst, 900, lambda: _client().get_candles(inst, "D", count=3))["close"]
    return float(d.iloc[-1] / d.iloc[-2] - 1) if len(d) >= 2 else 0.0


def _strategy_closeret(sid, holdings):
    if sid == "crypto_trend":
        from crypto_bot import fetch_daily
        r = 0.0
        for inst, w in holdings.items():
            try:
                s = fetch_daily(inst)
                r += w * float(s.iloc[-1] / s.iloc[-2] - 1)
            except Exception:
                continue
        return r
    return sum(w * _close_ret(inst) for inst, w in holdings.items())


def settle_users():
    """Daily close-of-book settle for EVERY user's paper portfolio, run by the
    scanner daemon in the 22:00-23:59 UTC window (>=1h after the OANDA close so
    the completed daily candle is guaranteed fresh). Rolls each book by the
    CLOSE-TO-CLOSE return of its mix (current holdings, scanner tilt applied).
    This is the ONE place books roll; the app's GET only displays. Users who
    never visit still accrue every day. Atomic write."""
    import json
    now = datetime.now(timezone.utc)
    if now.hour < 22:
        return 0
    today = now.strftime("%Y-%m-%d")
    path = os.path.join(HERE, "logs", "app_users.json")
    try:
        users = json.load(open(path, encoding="utf-8"))
    except Exception:
        return 0
    if not users or all(u.get("base_date") == today for u in users.values()):
        return 0
    mk = marks()["strategies"]              # tilt only
    dayret = {}
    for sid in SENT_BUCKETS:
        try:
            hw, _note = _holdings(sid)
            dayret[sid] = _strategy_closeret(sid, hw)
        except Exception:
            dayret[sid] = 0.0
    n = 0
    for uid, u in users.items():
        if u.get("base_date") == today:
            continue
        day = 0.0
        for sid, w in (u.get("mix") or {}).items():
            tilt = mk.get(sid, {}).get("tilt", 1.0)
            day += (w / 100.0) * tilt * dayret.get(sid, 0.0)
        u["base_nav"] = round(u.get("base_nav", 10000.0) * (1 + day), 2)
        u["base_date"] = today
        u.pop("base_day_ret", None)  # creation-day baseline no longer applies once settled
        u.setdefault("history", []).append([today, u["base_nav"]])
        u["history"] = u["history"][-400:]
        n += 1
    if n:
        tmp = path + ".tmp"
        json.dump(users, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, path)
    return n


def marks():
    """Per-strategy live state: holdings, day-return-so-far, note, sentiment tilt."""
    def f():
        import json
        sent = None
        try:
            sent = json.load(open(os.path.join(HERE, "logs", "sentiment.json"), encoding="utf-8"))
        except Exception:
            pass
        out = {}
        for sid in SENT_BUCKETS:
            try:
                hw, note = _holdings(sid)
                dr = _strategy_dayret(sid, hw)
            except Exception as e:
                hw, note, dr = {}, f"mark unavailable ({type(e).__name__})", 0.0
            tilt, tilt_why = 1.0, None
            scores = {}
            if sent:
                for b in SENT_BUCKETS[sid]:
                    v = sent.get("assets", {}).get(b)
                    if v:
                        scores[b] = v["score"]
                bad = [b for b, s in scores.items() if s < -40]
                if bad:
                    tilt, tilt_why = 0.75, f"Scanner strongly bearish on {', '.join(bad)} — exposure trimmed to 75%."
            top = sorted(hw.items(), key=lambda z: -z[1])[:6]
            out[sid] = {"day_ret": round(dr, 5), "gross": round(sum(hw.values()), 3),
                        "note": note, "tilt": tilt, "tilt_why": tilt_why,
                        "sentiment": scores,
                        "holdings": [{"inst": i, "name": NAMES.get(i, i), "w": round(w * 100, 1)}
                                     for i, w in top]}
        return {"generated": datetime.now(timezone.utc).isoformat(), "strategies": out}
    return _c("marks", 1200, f)
