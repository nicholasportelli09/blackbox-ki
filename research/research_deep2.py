"""
DEEP HUNT round 2 — zoom into the T1 discovery + finish T4/T5.

  T6  GOLD (and silver/platinum) REOPEN DRIFT: long the first H4 bar after the daily
      break (bars labelled 21:00/22:00 UTC = DST aliases). Exact net Sharpe at
      1.5/2/3bp per day, year-by-year consistency, and DST-alias-merged US-index
      session map for reference.
  T4' MACRO-CONDITIONAL GOLD (fixed dtypes).
  T5  FX weekend-gap fade.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data", "hist")
out = []
def P(s=""): out.append(s); print(s, flush=True)

def h4(inst):
    p = os.path.join(CACHE, f"{inst}_H4.csv")
    return pd.read_csv(p, parse_dates=["t"]).set_index("t") if os.path.exists(p) else None

def dlong(inst):
    p = os.path.join(CACHE, f"{inst}_Dlong.csv")
    return pd.read_csv(p, parse_dates=["t"]).set_index("t") if os.path.exists(p) else None

def srr(s, ppy=252):
    s = pd.Series(s).dropna()
    if len(s) < 40 or s.std() == 0: return 0.0, 0.0
    return float(s.mean()/s.std()*np.sqrt(ppy)), float(s.mean()*ppy)

def t6():
    P("="*88); P("T6  METALS REOPEN DRIFT — long first H4 bar after the daily break (21/22 UTC)"); P("="*88)
    for inst, base_cost in [("XAU_USD", 1.5), ("XAG_USD", 4.0), ("XPT_USD", 4.0)]:
        df = h4(inst)
        if df is None: continue
        r = df["close"]/df["open"] - 1
        hr = df.index.hour
        reopen = r[(hr == 21) | (hr == 22)]
        byday = reopen.groupby(reopen.index.date).sum()
        byday.index = pd.to_datetime(byday.index)
        sr_g, an_g = srr(byday)
        P(f"\n  {inst}  ({len(byday)} days, {byday.index[0].date()}..{byday.index[-1].date()})  GROSS ann {an_g*100:+.1f}%  SR {sr_g:.2f}")
        P(f"    {'cost/day':<10}{'net ann%':>9}{'net SR':>8}")
        for c_bps in [0, 1.5, 2, 3, base_cost] if base_cost not in (1.5,) else [0, 1, 1.5, 2, 3]:
            net = byday - c_bps/1e4
            sr_n, an_n = srr(net)
            P(f"    {c_bps}bp{'':<6}{an_n*100:>9.1f}{sr_n:>8.2f}")
        # year-by-year gross SR (consistency check)
        yr = byday.groupby(byday.index.year)
        line = "    yearly SR: " + "  ".join(f"{y}:{(g.mean()/g.std()*np.sqrt(252)) if g.std() else 0:+.1f}" for y, g in yr if len(g) > 60)
        P(line)
    # US indices DST-alias-merged map for reference
    P("\n  US indices — DST-merged session map (pooled SPX/NAS/US30, full sample, gross):")
    UNIV = ["SPX500_USD","NAS100_USD","US30_USD"]
    SESS = {"reopen(21/22)": (21,22), "night(01/02)": (1,2), "eu-am(05/06)": (5,6),
            "eu-pm/us-open(09/10)": (9,10), "us-am(13/14)": (13,14), "us-pm(17/18)": (17,18)}
    pool = []
    for i in UNIV:
        df = h4(i)
        if df is not None:
            r = df["close"]/df["open"]-1
            pool.append(pd.DataFrame({"r": r, "hr": df.index.hour}, index=df.index))
    allp = pd.concat(pool)
    mid = pd.Timestamp("2020-06-01")
    P(f"    {'session':<22}{'H1 SR':>7}{'H2 SR':>7}")
    for nm,(a,b) in SESS.items():
        s = allp[(allp["hr"]==a)|(allp["hr"]==b)]["r"]
        d = s.groupby(s.index.date).mean(); d.index = pd.to_datetime(d.index)
        s1,_ = srr(d[d.index<mid]); s2,_ = srr(d[d.index>=mid])
        P(f"    {nm:<22}{s1:>7.2f}{s2:>7.2f}")

def t4():
    P("\n"+"="*88); P("T4' GOLD trend gated by BOND regime"); P("="*88)
    g, b = dlong("XAU_USD"), dlong("USB10Y_USD")
    px = pd.DataFrame({"g": g["close"], "b": b["close"]}).dropna()
    rg = px["g"].pct_change()
    gtrend = (px["g"] > px["g"].rolling(200).mean()).shift(1).fillna(False).astype(bool)
    breg = (px["b"] > px["b"].rolling(100).mean()).shift(1).fillna(False).astype(bool)
    for lbl, mask in [("gold trend (uncond)", gtrend), ("trend & yields falling", gtrend & breg),
                      ("trend & yields rising", gtrend & (~breg))]:
        s = rg[mask]
        sr, an = srr(s)
        if len(s) > 100:
            mid = s.index[len(s)//2]
            s1,_ = srr(s[s.index<mid]); s2,_ = srr(s[s.index>=mid])
        else: s1=s2=0
        P(f"  {lbl:<26} days {len(s):>5}  ann {an*100:>6.1f}%  SR {sr:>5.2f}  halves {s1:.2f}/{s2:.2f}")

def t5():
    P("\n"+"="*88); P("T5  FX WEEKEND-GAP FADE (|gap|>0.15%)"); P("="*88)
    FX = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD","NZD_USD","EUR_AUD","GBP_AUD","AUD_NZD"]
    allr = []
    for i in FX:
        df = h4(i)
        if df is None: continue
        dow = pd.Series(df.index.dayofweek, index=df.index)
        first_of_week = (dow <= 1) & (dow.shift(1) >= 4)
        opens, closes = df["open"], df["close"]
        prev_close = closes.shift(1)
        gaps = (opens/prev_close - 1)[first_of_week.values]
        big = gaps[abs(gaps) > 0.0015]
        for ts, gp in big.items():
            loc = df.index.get_loc(ts)
            if loc + 6 >= len(df): continue
            ret = -np.sign(gp) * (closes.iloc[loc+5]/opens.iloc[loc] - 1) - 0.0001
            allr.append((ts, float(ret)))
    if len(allr) < 30:
        P(f"  only {len(allr)} events — insufficient"); return
    s = pd.Series(dict(allr)).sort_index()
    tpy = len(s)/((s.index[-1]-s.index[0]).days/365.25)
    mid = s.index[len(s)//2]
    f = lambda x: float(x.mean()/x.std()*np.sqrt(tpy)) if len(x)>10 and x.std() else 0
    P(f"  {len(s)} gaps ({tpy:.0f}/yr) | avg {s.mean()*100:+.3f}% | win {(s>0).mean()*100:.0f}% | SR {f(s):.2f} | halves {f(s[s.index<mid]):.2f}/{f(s[s.index>=mid]):.2f}")

if __name__ == "__main__":
    t6(); t4(); t5()
    with open(os.path.join(HERE, "logs", "research_deep2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\nSaved -> logs/research_deep2.txt")
