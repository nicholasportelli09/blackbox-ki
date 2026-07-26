"""
Phase-1 modular cross-asset FACTOR backtest (price-only): TSMOM + XS-momentum +
Value(5y reversal) + Defensive(low-vol), combined risk-parity, inverse-vol sized,
then vol-TARGETED to maximise return. ~20yr daily OANDA data.

Honest by construction: reports Sharpe, Deflated Sharpe (Bailey/Lopez de Prado),
CAGR & max drawdown at several vol targets, plus the single-factor breakdown so we
see the diversification uplift. No-lookahead: signals at month-end t, held t->t+1.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from oanda_client import OandaClient

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data", "hist")
os.makedirs(CACHE, exist_ok=True)
START_LONG = "2006-01-01T00:00:00Z"     # ~20 years
COST_BPS = 10            # per unit turnover, round-trip-ish (monthly rebalance)


def fetch_daily(inst):
    """Paginated daily candles from 2006, cached to disk (true ~20yr history)."""
    import time
    path = os.path.join(CACHE, f"{inst}_Dlong.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["t"]).set_index("t")
    c = OandaClient(); frm = START_LONG; rows = []
    while True:
        try:
            d = c._get(f"/v3/instruments/{inst}/candles",
                       params={"granularity": "D", "from": frm, "count": 4500, "price": "M"})
        except Exception:
            break
        cs = [x for x in d.get("candles", []) if x["complete"]]
        if not cs:
            break
        for x in cs:
            m = x["mid"]
            rows.append((x["time"], float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"])))
        if len(cs) < 4500:
            break
        frm = (pd.Timestamp(cs[-1]["time"]) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
    df = df.drop_duplicates("t").set_index("t").sort_index()
    df.to_csv(path)
    return df
CLASSES = {
    "FX": ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD","USD_CHF","NZD_USD",
           "EUR_AUD","GBP_AUD","AUD_NZD"],
    "Rates": ["USB02Y_USD","USB05Y_USD","USB10Y_USD","USB30Y_USD","DE10YB_EUR","UK10YB_GBP"],
    "Commodities": ["XAU_USD","XAG_USD","XPT_USD","XPD_USD","XCU_USD","WTICO_USD","BCO_USD",
                    "NATGAS_USD","CORN_USD","WHEAT_USD","SOYBN_USD","SUGAR_USD"],
    "Indices": ["SPX500_USD","NAS100_USD","US30_USD","DE30_EUR","UK100_GBP","JP225_USD",
                "EU50_EUR","AU200_AUD"],
}
ALL = [i for v in CLASSES.values() for i in v]
CLS_OF = {i: c for c, v in CLASSES.items() for i in v}
out = []
def P(s=""): out.append(s); print(s, flush=True)


def load_daily():
    closes = {}
    for inst in ALL:
        try:
            d = fetch_daily(inst)
            if len(d) > 300:
                closes[inst] = d["close"]
        except Exception as e:
            P(f"  {inst}: skip ({e})")
    px = pd.DataFrame(closes).sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def zscore_within_class(s):
    """Cross-sectional z-score within each asset class, per row (month)."""
    out = pd.Series(index=s.index, dtype=float)
    for c, insts in CLASSES.items():
        cols = [i for i in insts if i in s.index]
        v = s[cols].astype(float)
        m, sd = v.mean(), v.std()
        out[cols] = (v - m) / sd if sd and sd > 0 else 0.0
    return out.fillna(0.0)


def build_signals(px):
    """Returns dict month-> DataFrame of per-instrument factor scores + vol + fwd ret."""
    me = px.resample("ME").last()                       # month-end closes
    dret = px.pct_change()
    # ex-ante annualized vol (≈63 trading-day), as of each month-end
    vol_d = dret.rolling(63).std() * np.sqrt(252)
    vol_m = vol_d.resample("ME").last()
    r1 = me.pct_change(1); r3 = me.pct_change(3); r12 = me.pct_change(12)
    r12_1 = me.shift(1).pct_change(11)                  # t-12 -> t-1
    r5y = me.pct_change(60)
    fwd = me.pct_change().shift(-1)                     # next-month return (held t->t+1)
    months = me.index[13:-1]                            # need 12m history, 1m fwd
    panel = {}
    for t in months:
        if t not in r12.index:
            continue
        tsmom = (np.sign(r1.loc[t]) + np.sign(r3.loc[t]) + np.sign(r12.loc[t])) / 3.0
        xsmom = zscore_within_class(r12_1.loc[t])
        value = zscore_within_class(-r5y.loc[t])
        defen = zscore_within_class(-vol_m.loc[t])
        df = pd.DataFrame({"tsmom": tsmom, "xsmom": xsmom, "value": value,
                           "defen": defen, "vol": vol_m.loc[t], "fwd": fwd.loc[t]})
        df = df[(df["vol"] > 0) & np.isfinite(df["fwd"])]
        panel[t] = df.dropna(subset=["vol", "fwd"])
    return panel


def strat_returns(panel, weights):
    """weights = dict factor->weight. Returns monthly net return series (gross-1.0 base)."""
    wts_prev = None
    rows = []
    for t, df in panel.items():
        score = (weights["tsmom"] * df["tsmom"].fillna(0)
                 + weights["xsmom"] * df["xsmom"].fillna(0)
                 + weights["value"] * df["value"].fillna(0)
                 + weights["defen"] * df["defen"].fillna(0))
        w = score / df["vol"]                            # inverse-vol sizing
        gross = w.abs().sum()
        if gross > 0:
            w = w / gross                                # normalise to gross 1.0
        ret = float((w * df["fwd"]).sum())
        # turnover cost
        tc = 0.0
        if wts_prev is not None:
            allidx = w.index.union(wts_prev.index)
            turn = (w.reindex(allidx).fillna(0) - wts_prev.reindex(allidx).fillna(0)).abs().sum()
            tc = turn * COST_BPS / 1e4
        wts_prev = w
        rows.append((t, ret - tc))
    return pd.Series(dict(rows))


def vol_target(r, target_ann, cap=4.0):
    """Scale to target annualized vol using trailing 6m realized vol (lagged, no lookahead)."""
    rv = r.rolling(6).std() * np.sqrt(12)
    lev = (target_ann / rv).shift(1).clip(0, cap).fillna(0)
    return r * lev


def metrics(r):
    r = r.dropna()
    if len(r) < 24:
        return None
    ann = r.mean() * 12
    vol = r.std() * np.sqrt(12)
    sr = ann / vol if vol else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return dict(ann=ann, vol=vol, sr=sr, cagr=cagr, dd=dd, n=len(r), eq=eq.iloc[-1])


def deflated_sharpe(r, n_trials):
    """Bailey & Lopez de Prado Deflated Sharpe Ratio (prob true SR>0 after #trials)."""
    r = r.dropna(); T = len(r)
    if T < 24:
        return None
    sr = r.mean() / r.std() if r.std() else 0          # per-period (monthly) SR
    g3 = r.skew(); g4 = r.kurtosis() + 3.0             # non-excess kurtosis
    # expected max SR across N independent trials (variance of SR across trials ~ 1/T proxy)
    sr_trials_std = (1.0 / np.sqrt(T))                  # std of SR estimate under null
    g = 0.5772156649
    e = np.e
    sr0 = sr_trials_std * ((1 - g) * norm.ppf(1 - 1.0 / n_trials)
                           + g * norm.ppf(1 - 1.0 / (n_trials * e)))
    denom = np.sqrt((1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (T - 1))
    if denom <= 0:
        return None
    return float(norm.cdf((sr - sr0) / denom))


def run():
    P("Loading ~20yr daily for %d instruments..." % len(ALL))
    px = load_daily()
    P("  loaded %d instruments, %s .. %s" % (px.shape[1], px.index[0].date(), px.index[-1].date()))
    panel = build_signals(px)
    P("  %d monthly rebalances\n" % len(panel))

    # single-factor vs combined (shows diversification uplift)
    P("=" * 80)
    P(f"  FACTOR BREAKDOWN (each at 10% vol target, net of {COST_BPS}bps turnover cost)")
    P("=" * 80)
    P(f"  {'strategy':<26}{'CAGR%':>7}{'vol%':>7}{'Sharpe':>8}{'maxDD%':>8}{'DSR':>7}")
    singles = {"TSMOM only": dict(tsmom=1, xsmom=0, value=0, defen=0),
               "XS-momentum only": dict(tsmom=0, xsmom=1, value=0, defen=0),
               "Value only": dict(tsmom=0, xsmom=0, value=1, defen=0),
               "Defensive only": dict(tsmom=0, xsmom=0, value=0, defen=1)}
    COMBINED = dict(tsmom=1.0, xsmom=0.6, value=0.5, defen=0.4)
    for name, w in {**singles, "COMBINED (all 4)": COMBINED}.items():
        r = vol_target(strat_returns(panel, w), 0.10)
        m = metrics(r)
        if m:
            dsr = deflated_sharpe(r, 30)
            P(f"  {name:<26}{m['cagr']*100:>7.1f}{m['vol']*100:>7.1f}{m['sr']:>8.2f}"
              f"{m['dd']*100:>8.1f}{(dsr if dsr is not None else float('nan')):>7.2f}")

    # MAXIMISE RETURNS — combined book at rising vol targets
    P("\n" + "=" * 80)
    P("  MAXIMISE RETURNS — COMBINED book vol-targeted higher (return vs drawdown)")
    P("=" * 80)
    P(f"  {'vol target':<12}{'CAGR%':>8}{'real vol%':>10}{'Sharpe':>8}{'maxDD%':>9}{'final x':>9}")
    base = strat_returns(panel, COMBINED)
    for tgt in [0.10, 0.15, 0.20, 0.30, 0.40]:
        m = metrics(vol_target(base, tgt))
        if m:
            P(f"  {int(tgt*100)}%{'':<9}{m['cagr']*100:>8.1f}{m['vol']*100:>10.1f}{m['sr']:>8.2f}"
              f"{m['dd']*100:>9.1f}{m['eq']:>9.1f}")

    # per-asset-class contribution (combined weights, 10% vol)
    P("\n" + "=" * 80)
    P("  PER ASSET-CLASS (combined factors, 10% vol target)")
    P("=" * 80)
    for c in CLASSES:
        sub = {t: df[df.index.map(lambda i: CLS_OF.get(i)) == c] for t, df in panel.items()}
        sub = {t: d for t, d in sub.items() if len(d) >= 2}
        if not sub:
            continue
        r = vol_target(strat_returns(sub, COMBINED), 0.10)
        m = metrics(r)
        if m:
            P(f"  {c:<14}CAGR {m['cagr']*100:>6.1f}%  Sharpe {m['sr']:>5.2f}  maxDD {m['dd']*100:>6.1f}%")

    with open(os.path.join(HERE, "logs", "backtest_factors.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\n  Saved -> logs/backtest_factors.txt")


if __name__ == "__main__":
    run()
