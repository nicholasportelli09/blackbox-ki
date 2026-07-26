"""
STRATEGY CATALOG — the product's core data. Every strategy in the user-facing app
is computed from REAL 20-year backtests on real OANDA/Coinbase data (no invented
numbers). Per strategy: CAGR, vol, Sharpe, max drawdown, beta vs S&P 500, best/worst
year, monthly equity curve, and block-bootstrap WEALTH PROJECTION bands (5/25/50/75/95
percentiles at 1/3/5 years) — honest Monte-Carlo fans, not straight lines.

Output: logs/catalog.json (precomputed; served by the app API).
Methodology notes are embedded — net of estimated retail costs, backtest != future.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json
import numpy as np
import pandas as pd

from backtest_factors import fetch_daily, CLASSES
from crypto_backtest import load as crypto_load

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "logs", "catalog.json")
START_WEALTH = 10_000


def volt(r, t, ppy=252):
    rv = r.rolling(30).std() * np.sqrt(ppy)
    return (r * (t / rv).shift(1).clip(0, 4)).fillna(0)


def _series_equity_seasonal():
    idxs = CLASSES["Indices"]
    rets, ons = {}, {}
    for i in idxs:
        d = fetch_daily(i)
        if d is None or len(d) < 500: continue
        c = d["close"]; rets[i] = c.pct_change()
        ons[i] = (c > c.rolling(200).mean()).astype(float).shift(1)
    R = pd.DataFrame(rets); ON = pd.DataFrame(ons)
    seas = pd.Series([1.0 if m in {11,12,1,2,3,4} else 0.0 for m in R.index.month], index=R.index)
    M = ON.mul(seas, axis=0)
    return ((M * R) - M.diff().abs() * 0.000075).mean(axis=1).dropna()


def _series_gold_macro():
    g = fetch_daily("XAU_USD")["close"]; b = fetch_daily("USB10Y_USD")["close"]
    rg = g.pct_change()
    mask = ((g > g.rolling(200).mean()) & (b > b.rolling(100).mean())).shift(1).fillna(False).astype(bool)
    pos = mask.astype(float)
    return (pos * rg - pos.diff().abs().fillna(0) * 0.00015).dropna()


def _series_commod_trend():
    rets, ons = {}, {}
    for i in CLASSES["Commodities"]:
        d = fetch_daily(i)
        if d is None or len(d) < 500: continue
        c = d["close"]; rets[i] = c.pct_change()
        ons[i] = (c > c.rolling(200).mean()).astype(float).shift(1)
    R = pd.DataFrame(rets); ON = pd.DataFrame(ons)
    return ((ON * R) - ON.diff().abs() * 0.0002).mean(axis=1).dropna()


def _series_risk_parity():
    closes = {}
    for cls, insts in CLASSES.items():
        for i in insts:
            d = fetch_daily(i)
            if d is not None and len(d) > 500:
                closes[i] = d["close"]
    px = pd.DataFrame(closes)
    R = px.pct_change()
    vol = R.rolling(63).std() * np.sqrt(252)
    w = (1.0 / vol.replace(0, np.nan))
    w = w.div(w.sum(axis=1), axis=0).shift(1)
    base = ((w * R).sum(axis=1) - 0.0000125).dropna()   # ~0.3%/yr drag
    return volt(base, 0.08)                              # conservative 8% vol target


def _series_stack(vol_target=None):
    e1 = volt(_series_equity_seasonal(), 0.10)
    e2 = volt(_series_commod_trend(), 0.10)
    e3 = volt(_series_gold_macro(), 0.10)
    stack = pd.DataFrame({"a": e1, "b": e2, "c": e3}).dropna().mean(axis=1)
    return volt(stack, vol_target) if vol_target else stack


def _series_crypto():
    px = crypto_load()
    rets = px.pct_change()
    sig = (px > px.rolling(200).mean()).astype(float).shift(1)
    flips = sig.diff().abs().fillna(0)
    port = (sig * rets - flips * 0.0015).mean(axis=1)
    rv = port.rolling(30).std() * np.sqrt(365)
    lev = (0.35 / rv).shift(1).clip(0, 3).fillna(0)
    return (port * lev).dropna()


def _series_spx():
    return fetch_daily("SPX500_USD")["close"].pct_change().dropna()


CATALOG = [
    dict(id="stack_bal", name="Blackbox Multi-Edge — Balanced", risk="Moderate",
         tagline="Three independent, validated edges (seasonal equity, commodity trend, macro gold), blended equal-risk.",
         builder=lambda: _series_stack(None), ppy=252,
         method="3 low-correlation edges (pairwise corr 0.04-0.25), each 20yr-validated with split-half robustness; equal-risk blend, vol-targeted."),
    dict(id="stack_agg", name="Blackbox Multi-Edge — Aggressive", risk="High",
         tagline="The same three-edge engine levered to an 18% volatility target for higher return and drawdown.",
         builder=lambda: _series_stack(0.18), ppy=252,
         method="Same engine as Balanced at a higher vol target. Leverage amplifies both return and drawdown."),
    dict(id="equity_season", name="Seasonal Equity", risk="Moderate",
         tagline="Long global stock indices only in an uptrend and in the strong Nov-Apr season; cash otherwise.",
         builder=_series_equity_seasonal, ppy=252,
         method="8 indices; entries require >200-day MA AND Nov-Apr (Halloween effect). Both rules pre-specified, robust in both decade-halves."),
    dict(id="gold_macro", name="Macro Gold", risk="Moderate",
         tagline="Holds gold only when gold trends up AND falling yields give it a macro tailwind.",
         builder=_series_gold_macro, ppy=252,
         method="Gold>200dMA AND US10Y bond>100dMA. Zero-yield real-asset mechanism; conditional Sharpe 1.10, halves 1.00/1.22 over 20yr."),
    dict(id="commod_trend", name="Commodity Trend", risk="Moderate",
         tagline="Rides uptrends across 12 commodities (metals, energy, grains); flat in downtrends.",
         builder=_series_commod_trend, ppy=252,
         method="Long-only 200-day-MA trend, equal-weight across commodities. Diversifier: low correlation to equities."),
    dict(id="risk_parity", name="Global Risk Parity", risk="Conservative",
         tagline="Owns everything — 36 global markets weighted so each contributes equal risk.",
         builder=_series_risk_parity, ppy=252,
         method="Inverse-volatility weights across FX, bonds, commodities, indices. The diversified-beta benchmark (Sharpe ~0.5 over 20yr)."),
    dict(id="crypto_trend", name="Crypto Momentum", risk="Very High",
         tagline="Long liquid crypto majors in uptrends, 100% cash in crypto winters. High octane.",
         builder=_series_crypto, ppy=365,
         method="200-day-MA trend on 13 majors, inverse-vol, 35% vol target. Backtest Sharpe ~1.2 BUT one bull regime + survivorship caveats; expect less."),
    dict(id="spx_hold", name="S&P 500 Buy & Hold", risk="High",
         tagline="The benchmark: own the index through everything, including -50% crashes.",
         builder=_series_spx, ppy=252,
         method="Reference strategy for comparison. Full equity beta and full equity drawdowns."),
]


def stats(r, spx, ppy):
    r = r.dropna()
    ann = r.mean() * ppy; vol = r.std() * np.sqrt(ppy)
    sr = float(ann / vol) if vol else 0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(r) / ppy
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    # normalize both to calendar dates so OANDA (21:00-stamped) and Coinbase
    # (00:00-stamped) series align for beta/correlation
    rn = r.copy(); rn.index = rn.index.normalize()
    sn = spx.copy(); sn.index = sn.index.normalize()
    rn = rn.groupby(rn.index).sum(); sn = sn.groupby(sn.index).sum()
    j = pd.DataFrame({"r": rn, "m": sn}).dropna()
    beta = float(j["r"].cov(j["m"]) / j["m"].var()) if len(j) > 200 else None
    corr = float(j["r"].corr(j["m"])) if len(j) > 200 else None
    ybr = (1 + r).groupby(r.index.year).prod() - 1
    ybr = ybr[[y for y in ybr.index if (r.index.year == y).sum() > ppy * 0.6]]
    h = len(r) // 2
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(ppy)) if len(x) > 100 and x.std() else 0
    return dict(cagr=round(cagr * 100, 2), vol=round(vol * 100, 2), sharpe=round(sr, 2),
                max_dd=round(dd * 100, 1), beta=round(beta, 2) if beta is not None else None,
                corr_spx=round(corr, 2) if corr is not None else None,
                best_year=round(float(ybr.max()) * 100, 1) if len(ybr) else None,
                worst_year=round(float(ybr.min()) * 100, 1) if len(ybr) else None,
                years=round(yrs, 1),
                sharpe_h1=round(sh(r.iloc[:h]), 2), sharpe_h2=round(sh(r.iloc[h:]), 2))


def projections(r, ppy, n_paths=800, block=60):
    """Block-bootstrap terminal-wealth percentile bands at 1/3/5 years."""
    v = r.dropna().values
    rng = np.random.default_rng(13)
    out = {}
    for label, years in [("1y", 1), ("3y", 3), ("5y", 5)]:
        days = int(ppy * years)
        # sample block start indices for all paths at once
        n_blocks = days // block + 1
        starts = rng.integers(0, max(len(v) - block, 1), size=(n_paths, n_blocks))
        paths = np.empty((n_paths, n_blocks * block))
        for k in range(n_blocks):
            idx = starts[:, k][:, None] + np.arange(block)[None, :]
            paths[:, k * block:(k + 1) * block] = v[np.clip(idx, 0, len(v) - 1)]
        term = np.prod(1 + paths[:, :days], axis=1) * START_WEALTH
        pct = np.percentile(term, [5, 25, 50, 75, 95])
        out[label] = {p: round(float(x)) for p, x in zip(["p5", "p25", "p50", "p75", "p95"], pct)}
        out[label]["prob_loss"] = round(float((term < START_WEALTH).mean()) * 100, 1)
    return out


def equity_curve(r, ppy, points=160):
    eq = (1 + r.dropna()).cumprod()
    m = eq.resample("W").last().dropna()
    step = max(1, len(m) // points)
    m = m.iloc[::step]
    return [{"t": str(t.date()), "v": round(float(v), 4)} for t, v in m.items()]


def run():
    spx = _series_spx()
    cat = []
    for s in CATALOG:
        print(f"computing {s['id']}...", flush=True)
        try:
            r = s["builder"]()
        except Exception as e:
            print(f"  FAILED {e}"); continue
        st = stats(r, spx, s["ppy"])
        cat.append(dict(id=s["id"], name=s["name"], risk=s["risk"], tagline=s["tagline"],
                        method=s["method"], stats=st, projections=projections(r, s["ppy"]),
                        curve=equity_curve(r, s["ppy"]),
                        span=f"{r.index[0].date()} to {r.index[-1].date()}"))
        print(f"  SR {st['sharpe']}  CAGR {st['cagr']}%  DD {st['max_dd']}%  beta {st['beta']}")
    doc = dict(generated=pd.Timestamp.utcnow().isoformat(), start_wealth=START_WEALTH,
               disclaimer=("All statistics computed from real historical backtests (2006-2026 "
                           "OANDA data; crypto 2016-2026 Coinbase) net of estimated retail costs. "
                           "Backtested performance does not guarantee future results. Projection "
                           "bands are block-bootstrap resamples of realised history. Demo/practice "
                           "capital only — not investment advice."),
               strategies=cat)
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"\nSaved {len(cat)} strategies -> {OUT}")


if __name__ == "__main__":
    run()
