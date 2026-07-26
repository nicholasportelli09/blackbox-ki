"""
DEEP EDGE HUNT — five structurally-motivated, PRE-SPECIFIED tests on mechanisms we
have never tested (per the brain's rejected/confirmed ledger). Discipline:
  - selection decisions on HALF 1 only, validation on HALF 2 (true OOS)
  - harsh realistic OANDA costs charged per round trip
  - split-half Sharpe reported for everything

  T1  H4 SESSION ANATOMY (indices + gold): which 4-hour blocks of the 24h day carry
      return? (US cash-equity overnight anomaly, harvestable via ~23h CFDs. Our daily
      bars could never see this.) Select blocks on 2014-2020, validate 2020-2026.
  T2  PAIR / SPREAD MEAN-REVERSION: gold/silver, Brent/WTI, US 2s10s bond spread,
      SPX/Dow. z-score of 60d log-ratio; enter |z|>2, exit z=0 or 30d. Costs 6-10bps.
  T3  PANIC DIP-BUY in uptrend (indices): ret < -2*sigma20 AND close>200dMA -> long
      3 days. Pooled across 8 indices. Cost 1.5bps RT.
  T4  MACRO-CONDITIONAL GOLD: gold trend gated by bond regime (USB10Y>100dMA = yields
      falling). vs unconditional gold trend.
  T5  FX WEEKEND-GAP FADE: |Mon open - Fri close| > 0.15% -> fade, exit on fill or
      24h. Pooled 9 pairs. Cost 1bp RT.
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
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=["t"]).set_index("t")
    return df

def dlong(inst):
    p = os.path.join(CACHE, f"{inst}_Dlong.csv")
    if not os.path.exists(p): return None
    return pd.read_csv(p, parse_dates=["t"]).set_index("t")

def sr_ann(r, ppy):
    r = pd.Series(r).dropna()
    if len(r) < 40 or r.std() == 0: return 0.0, 0.0
    return float(r.mean()/r.std()*np.sqrt(ppy)), float(r.mean()*ppy)

# ---------------- T1: H4 session anatomy ----------------
def t1():
    P("="*88); P("T1  H4 SESSION ANATOMY — which 4h blocks carry the return (select H1, validate H2)"); P("="*88)
    UNIV = {"US indices": ["SPX500_USD","NAS100_USD","US30_USD"],
            "EU indices": ["DE30_EUR","EU50_EUR","UK100_GBP"],
            "Gold":       ["XAU_USD"]}
    results = {}
    for gname, insts in UNIV.items():
        blocks = {}   # hour -> list of (ts, ret) pooled across group
        for i in insts:
            df = h4(i)
            if df is None: continue
            r = (df["close"]/df["open"]-1)
            hr = df.index.hour
            for hh in sorted(set(hr)):
                blocks.setdefault(hh, []).append(pd.Series(r[hr==hh].values, index=df.index[hr==hh]))
        mid = pd.Timestamp("2020-06-01")
        P(f"\n  {gname}: per-block ann.ret% / Sharpe   (H1 = 2014-2020 | H2 = 2020-2026)")
        P(f"  {'blk(UTC)':<10}{'H1 ret%':>9}{'H1 SR':>7}{'H2 ret%':>9}{'H2 SR':>7}")
        sel = []
        for hh in sorted(blocks):
            s = pd.concat(blocks[hh]).sort_index()
            ppy = 252  # one obs per day per instrument, pooled -> ~252*n, use per-block daily mean
            byday = s.groupby(s.index.date).mean()
            byday.index = pd.to_datetime(byday.index)
            h1, h2 = byday[byday.index < mid], byday[byday.index >= mid]
            sr1, an1 = sr_ann(h1, 252); sr2, an2 = sr_ann(h2, 252)
            tag = ""
            if sr1 > 0.35:
                sel.append(hh); tag = "  <-- selected on H1"
            P(f"  {hh:02d}:00{'':<5}{an1*100:>9.1f}{sr1:>7.2f}{an2*100:>9.1f}{sr2:>7.2f}{tag}")
        # validate the H1-selected block-set on H2, net of 1 RT/day cost
        if sel:
            s_all = []
            for hh in sel: s_all.append(pd.concat(blocks[hh]).sort_index())
            s = pd.concat(s_all).sort_index()
            byday = s.groupby(s.index.date).sum() / max(len(UNIV[gname]),1)  # sum selected blocks per day (per-inst avg)
            byday.index = pd.to_datetime(byday.index)
            cost = 0.00015 if "Gold" not in gname else 0.0002   # 1.5-2 bps RT/day
            h2net = byday[byday.index >= mid] - cost
            h1net = byday[byday.index < mid] - cost
            sr2, an2 = sr_ann(h2net, 252); sr1, an1 = sr_ann(h1net, 252)
            results[gname] = (sel, sr1, an1, sr2, an2)
            P(f"  >> HOLD blocks {sel} net of {cost*1e4:.1f}bp/day:  H1 {an1*100:+.1f}%/SR {sr1:.2f}  |  OOS H2 {an2*100:+.1f}%/SR {sr2:.2f}")
    return results

# ---------------- T2: pair mean reversion ----------------
def t2():
    P("\n"+"="*88); P("T2  PAIR/SPREAD MEAN-REVERSION (z>2 entry, exit z=0 or 30d)"); P("="*88)
    PAIRS = [("XAU_USD","XAG_USD", 8), ("BCO_USD","WTICO_USD", 10),
             ("USB02Y_USD","USB10Y_USD", 6), ("SPX500_USD","US30_USD", 3)]
    P(f"  {'pair':<26}{'trades':>7}{'ann%':>7}{'SR':>6}{'H1 SR':>7}{'H2 SR':>7}{'verdict':>9}")
    res = {}
    for a, b, cost_bps in PAIRS:
        da, db = dlong(a), dlong(b)
        if da is None or db is None: continue
        px = pd.DataFrame({"a": da["close"], "b": db["close"]}).dropna()
        lr = np.log(px["a"]/px["b"])
        z = (lr - lr.rolling(60).mean())/lr.rolling(60).std()
        ra, rb = px["a"].pct_change(), px["b"].pct_change()
        pos = 0; entry_i = 0; rets = []; n_tr = 0
        zv = z.values; rav = ra.values; rbv = rb.values
        daily = np.zeros(len(px))
        for i in range(61, len(px)):
            if pos == 0:
                if zv[i-1] > 2:   pos = -1; entry_i = i; n_tr += 1   # short a, long b
                elif zv[i-1] < -2: pos = 1; entry_i = i; n_tr += 1   # long a, short b
                if pos != 0: daily[i] -= cost_bps/1e4
            else:
                daily[i] += pos*0.5*(rav[i]-rbv[i])
                if (pos == -1 and zv[i] < 0) or (pos == 1 and zv[i] > 0) or (i-entry_i > 30):
                    pos = 0
        r = pd.Series(daily, index=px.index)
        mid = px.index[len(px)//2]
        sr, an = sr_ann(r, 252); s1,_ = sr_ann(r[r.index<mid],252); s2,_ = sr_ann(r[r.index>=mid],252)
        verdict = "ROBUST" if s1>0.3 and s2>0.3 else ("weak" if sr>0.2 else "dead")
        res[f"{a[:3]}/{b[:3]}"] = (sr, s1, s2)
        P(f"  {a[:6]+' vs '+b[:6]:<26}{n_tr:>7}{an*100:>7.1f}{sr:>6.2f}{s1:>7.2f}{s2:>7.2f}{verdict:>9}")
    return res

# ---------------- T3: panic dip-buy ----------------
def t3():
    P("\n"+"="*88); P("T3  PANIC DIP-BUY in uptrend (ret<-2sig & >200dMA -> long 3d), pooled indices"); P("="*88)
    INDICES = ["SPX500_USD","NAS100_USD","US30_USD","DE30_EUR","UK100_GBP","JP225_USD","EU50_EUR","AU200_AUD"]
    allr = []
    for i in INDICES:
        d = dlong(i)
        if d is None: continue
        c = d["close"]; r = c.pct_change()
        sig = r.rolling(20).std(); ma = c.rolling(200).mean()
        trig = (r < -2*sig) & (c > ma)
        idx = np.where(trig.values)[0]
        last_exit = -1
        for j in idx:
            if j <= last_exit or j+3 >= len(c): continue
            ret3 = float(c.iloc[j+3]/c.iloc[j]-1) - 0.00015
            allr.append((c.index[j], ret3)); last_exit = j+3
    s = pd.Series(dict(allr)).sort_index()
    mid = s.index[len(s)//2]
    # ~len(s)/20yr trades per yr; annualize per-trade series by trades/yr
    tpy = len(s)/((s.index[-1]-s.index[0]).days/365.25)
    sr = float(s.mean()/s.std()*np.sqrt(tpy)) if s.std() else 0
    s1 = s[s.index<mid]; s2 = s[s.index>=mid]
    sr1 = float(s1.mean()/s1.std()*np.sqrt(tpy)) if s1.std() else 0
    sr2 = float(s2.mean()/s2.std()*np.sqrt(tpy)) if s2.std() else 0
    P(f"  {len(s)} trades ({tpy:.0f}/yr) | avg {s.mean()*100:+.2f}%/trade | win {(s>0).mean()*100:.0f}% | SR {sr:.2f} | halves {sr1:.2f}/{sr2:.2f}")
    return sr, sr1, sr2, s

# ---------------- T4: macro-conditional gold ----------------
def t4():
    P("\n"+"="*88); P("T4  GOLD trend gated by BOND regime (USB10Y>100dMA = falling yields)"); P("="*88)
    g, b = dlong("XAU_USD"), dlong("USB10Y_USD")
    px = pd.DataFrame({"g": g["close"], "b": b["close"]}).dropna()
    rg = px["g"].pct_change()
    gtrend = (px["g"] > px["g"].rolling(200).mean()).shift(1)
    bregime = (px["b"] > px["b"].rolling(100).mean()).shift(1)
    uncond = rg[gtrend.fillna(False)]
    cond = rg[(gtrend & bregime).fillna(False)]
    anti = rg[(gtrend & ~bregime).fillna(False)]
    for lbl, s in [("gold trend (uncond)", uncond), ("gold trend & yields falling", cond), ("gold trend & yields RISING", anti)]:
        sr, an = sr_ann(s, 252)
        mid = s.index[len(s)//2] if len(s)>100 else None
        s1,_ = sr_ann(s[s.index<mid],252) if mid is not None else (0,0)
        s2,_ = sr_ann(s[s.index>=mid],252) if mid is not None else (0,0)
        P(f"  {lbl:<30} days {len(s):>5}  ann {an*100:>6.1f}%  SR {sr:>5.2f}  halves {s1:.2f}/{s2:.2f}")

# ---------------- T5: FX weekend gap fade ----------------
def t5():
    P("\n"+"="*88); P("T5  FX WEEKEND-GAP FADE (|gap|>0.15%, exit fill/24h), pooled 9 pairs"); P("="*88)
    FX = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD","NZD_USD","EUR_AUD","GBP_AUD","AUD_NZD"]
    allr = []
    for i in FX:
        df = h4(i)
        if df is None: continue
        dow = df.index.dayofweek
        # Monday's first bar open vs previous close
        opens = df["open"]; closes = df["close"]
        prev_close = closes.shift(1)
        is_mon_first = (dow == 0) & (pd.Series(dow, index=df.index).shift(1) != 0).values
        gaps = (opens/prev_close - 1)[is_mon_first]
        big = gaps[abs(gaps) > 0.0015]
        for ts, gp in big.items():
            loc = df.index.get_loc(ts)
            if loc+6 >= len(df): continue
            # fade: position = -sign(gap); exit at close 6 bars (~24h) later
            ret = -np.sign(gp) * (closes.iloc[loc+5]/opens.iloc[loc] - 1) - 0.0001
            allr.append((ts, float(ret)))
    s = pd.Series(dict(allr)).sort_index()
    if len(s) < 30:
        P(f"  only {len(s)} events — insufficient"); return
    tpy = len(s)/((s.index[-1]-s.index[0]).days/365.25)
    mid = s.index[len(s)//2]
    sr = float(s.mean()/s.std()*np.sqrt(tpy)) if s.std() else 0
    s1 = s[s.index<mid]; s2 = s[s.index>=mid]
    sr1 = float(s1.mean()/s1.std()*np.sqrt(tpy)) if s1.std() else 0
    sr2 = float(s2.mean()/s2.std()*np.sqrt(tpy)) if s2.std() else 0
    P(f"  {len(s)} gaps ({tpy:.0f}/yr) | avg {s.mean()*100:+.3f}% | win {(s>0).mean()*100:.0f}% | SR {sr:.2f} | halves {sr1:.2f}/{sr2:.2f}")

if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t5()
    with open(os.path.join(HERE, "logs", "research_deep.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\nSaved -> logs/research_deep.txt")
