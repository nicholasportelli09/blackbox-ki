"""
Crypto trend-following engine — PAPER TRADED (no broker, free Coinbase public data).
Long-only 200-day-MA trend on liquid majors, inverse-vol weighted, vol-targeted
(holds cash when coins are below trend / to cap risk). Maintains a simulated 100k
book marked to live prices. This is the validated 20%+ avenue (Sharpe ~1.2 backtest)
run in paper until it proves out, then it can go live on a crypto broker.

State: logs/crypto_state.json (cash, positions, peak).  NAV log: logs/crypto_nav.csv
"""
import os, json, time, urllib.request
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from logger import log
import notifier

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data", "crypto"); os.makedirs(CACHE, exist_ok=True)
STATE = os.path.join(HERE, "logs", "crypto_state.json")
NAVLOG = os.path.join(HERE, "logs", "crypto_nav.csv")
CB = "https://api.exchange.coinbase.com"

PRODUCTS = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD",
            "LINK-USD","AVAX-USD","LTC-USD","DOT-USD"]
START_CAP = 100_000.0
TARGET_VOL = 0.35          # annualized portfolio vol target (long-only, partial cash)
MA = 200                   # trend filter window (days)
COST = 0.0015              # 15 bps per unit traded


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "blackbox/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def fetch_daily(product, refresh_h=8):
    """Daily closes, cached; refreshed if older than refresh_h hours."""
    path = os.path.join(CACHE, product.replace("-", "_") + "_live.csv")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < refresh_h * 3600:
        return pd.read_csv(path, parse_dates=["t"]).set_index("t")["close"]
    import datetime as dt
    rows = {}
    start = dt.datetime.utcnow() - dt.timedelta(days=420); now = dt.datetime.utcnow(); cur = start
    while cur < now:
        end = min(cur + dt.timedelta(days=290), now)
        try:
            d = _get(f"{CB}/products/{product}/candles?granularity=86400"
                     f"&start={cur.isoformat()}Z&end={end.isoformat()}Z")
            for c in d:
                rows[dt.datetime.utcfromtimestamp(c[0])] = c[4]
        except Exception:
            pass
        cur = end; time.sleep(0.25)
    if len(rows) < MA + 10:
        # fall back to existing cache if refresh failed
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["t"]).set_index("t")["close"]
        return None
    s = pd.Series(rows).sort_index(); s.index.name = "t"; s.name = "close"
    s.to_frame().to_csv(path)
    return s


def spot(product):
    try:
        return float(_get(f"{CB}/products/{product}/ticker")["price"])
    except Exception:
        return None


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"cash": START_CAP, "positions": {}, "peak": START_CAP, "last_rebal": "",
            "started": datetime.now(timezone.utc).isoformat()}


def save_state(st):
    tmp = STATE + ".tmp"; json.dump(st, open(tmp, "w"), indent=2); os.replace(tmp, STATE)


def log_nav(nav, gross, n_on):
    new = not os.path.exists(NAVLOG)
    with open(NAVLOG, "a", newline="") as f:
        import csv; w = csv.writer(f)
        if new: w.writerow(["ts", "nav", "gross", "n_on"])
        w.writerow([datetime.now(timezone.utc).isoformat(), round(nav, 2), round(gross, 3), n_on])


def signals():
    """Per product: trend on/off + annualized vol + last close, from daily data."""
    out = {}
    for p in PRODUCTS:
        s = fetch_daily(p)
        if s is None or len(s) < MA + 5:
            continue
        ma = s.rolling(MA).mean().iloc[-1]
        vol = s.pct_change().tail(60).std() * np.sqrt(365)
        out[p] = {"on": bool(s.iloc[-1] > ma), "vol": float(vol), "close": float(s.iloc[-1])}
    return out


def run_cycle():
    st = load_state()
    sig = signals()
    if not sig:
        log.warning("crypto: no signals (data fetch failed); skipping."); return
    prices = {p: spot(p) or sig[p]["close"] for p in sig}
    # mark current NAV
    pos = st["positions"]
    holdings = sum(pos.get(p, 0.0) * prices.get(p, 0.0) for p in pos)
    nav = st["cash"] + holdings
    st["peak"] = max(st.get("peak", START_CAP), nav)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    on = [p for p in sig if sig[p]["on"] and sig[p]["vol"] > 0]
    avg_vol = np.mean([sig[p]["vol"] for p in on]) if on else 0
    gross = min(1.0, TARGET_VOL / avg_vol) if avg_vol > 0 else 0.0
    # AI sentiment RISK TILT (bounded, symmetric): bearish trims gross deployment
    # (<=-40 -> 0.75x, <=-60 -> 0.5x); bullish can only RESTORE toward full
    # deployment (>=+40 -> 1.1x, >=+60 -> 1.2x, hard-capped at gross=1.0 so the
    # book is never levered past its vol target). Fails open on stale data.
    try:
        from sentiment_scanner import read_latest
        _sv = (read_latest() or {}).get("assets", {}).get("crypto", {})
        if _sv and gross > 0:
            _sc = float(_sv.get("score", 0))
            _tilt = (0.5 if _sc <= -60 else 0.75 if _sc <= -40 else
                     1.2 if _sc >= 60 else 1.1 if _sc >= 40 else 1.0)
            if _tilt != 1.0:
                log.info(f"crypto sentiment tilt: score {_sc:+.0f} -> gross x{_tilt}")
                gross = min(1.0, gross * _tilt)
    except Exception:
        pass
    log_nav(nav, gross, len(on))

    # AI sentiment SHADOW advisory (informs/explains; validated rules trade)
    try:
        from sentiment_scanner import read_latest
        snt = read_latest()
        if snt:
            cs = snt["assets"].get("crypto", {})
            log.info(f"crypto sentiment advisory: score {cs.get('score')} ({cs.get('trend')}) — "
                     f"{cs.get('explanation','')[:110]}")
    except Exception:
        pass

    # rebalance once per day
    if st.get("last_rebal") == today:
        save_state(st)
        log.info(f"crypto: NAV {nav:,.0f} ({(nav/START_CAP-1)*100:+.2f}%) | on {sorted(on)} | gross {gross:.2f} | already rebalanced today")
        return

    inv = {p: 1.0 / sig[p]["vol"] for p in on}
    tot = sum(inv.values())
    target_val = {p: nav * gross * inv[p] / tot for p in on} if tot else {}
    # trade every product toward target (0 if off)
    cash = st["cash"]; newpos = dict(pos)
    traded = 0.0
    for p in set(list(pos.keys()) + list(sig.keys())):
        px = prices.get(p) or sig.get(p, {}).get("close")
        if not px:
            continue
        cur_val = newpos.get(p, 0.0) * px
        tgt_val = target_val.get(p, 0.0)
        d_val = tgt_val - cur_val
        traded += abs(d_val)
        cash -= d_val
        newpos[p] = (tgt_val / px) if tgt_val > 0 else 0.0
    cash -= traded * COST                      # transaction cost
    newpos = {p: u for p, u in newpos.items() if u > 1e-9}
    st["cash"] = cash; st["positions"] = newpos; st["last_rebal"] = today
    save_state(st)
    log.info(f"crypto REBALANCE: NAV {nav:,.0f} ({(nav/START_CAP-1)*100:+.2f}%) | "
             f"on {sorted(on)} | gross {gross:.2f} | traded {traded:,.0f} | held {sorted(newpos)}")
    # ntfy alert only when the held set actually changes (entries/exits), not on resizes
    prev_held = {p for p, u in pos.items() if u > 1e-9}
    new_held = set(newpos)
    if prev_held != new_held:
        entered = sorted(c.replace("-USD", "") for c in new_held - prev_held)
        exited = sorted(c.replace("-USD", "") for c in prev_held - new_held)
        if not new_held:
            notifier.send(f"Crypto PAPER -> 100% CASH (all coins below trend). NAV {nav:,.0f} "
                          f"({(nav/START_CAP-1)*100:+.1f}%)", title="Blackbox Crypto", priority="high")
        else:
            parts = []
            if entered: parts.append("bought " + ", ".join(entered))
            if exited: parts.append("sold " + ", ".join(exited))
            held = ", ".join(sorted(c.replace("-USD", "") for c in new_held))
            notifier.send(f"Crypto PAPER: {'; '.join(parts)} | holding {held} | NAV {nav:,.0f}",
                          title="Blackbox Crypto")


if __name__ == "__main__":
    run_cycle()
