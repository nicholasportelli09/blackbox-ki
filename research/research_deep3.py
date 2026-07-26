"""
DEEP HUNT round 3 — build out the T4 discovery: the RATES-GATED METALS sleeve.

Mechanism: precious metals are zero-yield real assets -> they perform when yields
FALL (bond prices rise). Gate = USB10Y above its 100dMA. Test the same gate on each
metal (confirmatory cross-asset evidence), the combined inverse-vol sleeve with
vol-targeting, costs, and an equity variant for reference.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data", "hist")
out = []
def P(s=""): out.append(s); print(s, flush=True)

def dlong(inst):
    p = os.path.join(CACHE, f"{inst}_Dlong.csv")
    return pd.read_csv(p, parse_dates=["t"]).set_index("t") if os.path.exists(p) else None

def srr(s, ppy=252):
    s = pd.Series(s).dropna()
    if len(s) < 60 or s.std() == 0: return 0.0, 0.0, 0.0
    eq = (1+s).cumprod()
    dd = float((eq/eq.cummax()-1).min())
    return float(s.mean()/s.std()*np.sqrt(ppy)), float(s.mean()*ppy), dd

def halves(s):
    s = s.dropna()
    if len(s) < 200: return 0.0, 0.0
    mid = s.index[len(s)//2]
    a = s[s.index < mid]; b = s[s.index >= mid]
    f = lambda x: float(x.mean()/x.std()*np.sqrt(252)) if len(x) > 60 and x.std() else 0.0
    return f(a), f(b)

def run():
    bond = dlong("USB10Y_USD")["close"]
    breg = (bond > bond.rolling(100).mean()).shift(1)

    P("="*88)
    P("RATES-GATED METALS — long metal when (metal>200dMA) AND (USB10Y>100dMA)")
    P("="*88)
    P(f"  {'asset':<10}{'variant':<28}{'ann%':>7}{'SR':>6}{'maxDD%':>8}{'H1/H2':>12}")
    strat_rets = {}
    for m in ["XAU_USD", "XAG_USD", "XPT_USD"]:
        d = dlong(m)
        if d is None: continue
        c = d["close"]; r = c.pct_change()
        tr = (c > c.rolling(200).mean()).shift(1)
        idx = r.index.intersection(breg.index)
        rr, trm, bg = r[idx], tr[idx].fillna(False).astype(bool), breg[idx].fillna(False).astype(bool)
        for lbl, mask in [("trend only", trm), ("trend & yields falling", trm & bg)]:
            pos = mask.astype(float)
            # cost: 1.5bp per flip (regime changes only — few/yr)
            flips = pos.diff().abs().fillna(0)
            net = pos*rr - flips*0.00015
            sr, an, dd = srr(net)
            h1, h2 = halves(net)
            P(f"  {m[:3]:<10}{lbl:<28}{an*100:>7.1f}{sr:>6.2f}{dd*100:>8.1f}{f'{h1:.2f}/{h2:.2f}':>12}")
            if lbl.startswith("trend &"):
                strat_rets[m] = net

    # combined sleeve: inverse-vol across the 3 gated metals, vol-target 12%
    P("\n  COMBINED gated-metals sleeve (inverse-vol, vol-target 12%):")
    R = pd.DataFrame(strat_rets).fillna(0)
    vol = R.rolling(30).std()*np.sqrt(252)
    w = (1.0/vol.replace(0, np.nan)).fillna(0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port = (w.shift(1)*R).sum(axis=1)
    rv = port.rolling(30).std()*np.sqrt(252)
    lev = (0.12/rv).shift(1).clip(0, 3).fillna(0)
    pv = port*lev
    sr, an, dd = srr(pv); h1, h2 = halves(pv)
    P(f"    ann {an*100:+.1f}%  SR {sr:.2f}  maxDD {dd*100:.1f}%  halves {h1:.2f}/{h2:.2f}")
    # exposure stats
    days_on = (R.abs().sum(axis=1) > 0).mean()
    P(f"    time exposed: {days_on*100:.0f}% of days (in cash otherwise)")

    # correlation with the confirmed equity seasonal+trend edge
    P("\n  Correlation with existing edges:")
    try:
        idxs = ["SPX500_USD","NAS100_USD","US30_USD","DE30_EUR","UK100_GBP","JP225_USD","EU50_EUR","AU200_AUD"]
        rets, ons = {}, {}
        for i in idxs:
            d = dlong(i)
            if d is None: continue
            c = d["close"]; rets[i] = c.pct_change()
            ons[i] = (c > c.rolling(200).mean()).astype(float).shift(1)
        RR = pd.DataFrame(rets); ON = pd.DataFrame(ons)
        months = RR.index.month
        seas = pd.Series([1.0 if mm in {11,12,1,2,3,4} else 0.0 for mm in months], index=RR.index)
        eq_edge = (ON.mul(seas, axis=0)*RR).mean(axis=1).dropna()
        j = pd.DataFrame({"m": pv, "e": eq_edge}).dropna()
        P(f"    corr(gated-metals, equity seasonal+trend) = {j['m'].corr(j['e']):+.2f}")
    except Exception as e:
        P(f"    (corr calc failed: {e})")

    # equity variant for reference: NAS trend & yields falling
    P("\n  Reference — NAS100 trend gated by same bond regime:")
    d = dlong("NAS100_USD"); c = d["close"]; r = c.pct_change()
    tr = (c > c.rolling(200).mean()).shift(1)
    idx = r.index.intersection(breg.index)
    rr, trm, bg = r[idx], tr[idx].fillna(False).astype(bool), breg[idx].fillna(False).astype(bool)
    for lbl, mask in [("trend only", trm), ("trend & yields falling", trm & bg), ("trend & yields rising", trm & (~bg))]:
        s = rr[mask]; sr, an, _ = srr(s); h1, h2 = halves(s)
        P(f"    {lbl:<26} ann {an*100:>6.1f}%  SR {sr:>5.2f}  halves {h1:.2f}/{h2:.2f}")

    with open(os.path.join(HERE, "logs", "research_deep3.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\nSaved -> logs/research_deep3.txt")

if __name__ == "__main__":
    run()
