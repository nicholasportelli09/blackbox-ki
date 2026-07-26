"""
The honest test for 20%+ CAGR: crypto trend-following. Crypto is the one liquid,
retail-accessible market still inefficient + high-vol enough to plausibly support
quant-fund returns. Trend-following (long in uptrends, flat/cash in bear markets)
historically captured crypto's upside while sidestepping its -80% crashes.

Daily data via yfinance (~2015+). Tests buy-hold vs 200d-MA trend vs trend+vol-target,
equal-weight basket and BTC alone. Honest: CAGR, Sharpe, maxDD, split-half robustness,
net of ~15bps per position flip. Crypto risks (regime, the great run may not repeat) noted.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

PRODUCTS = ["BTC-USD","ETH-USD","LTC-USD","BCH-USD","ETC-USD","SOL-USD","ADA-USD",
            "XRP-USD","DOGE-USD","LINK-USD","AVAX-USD","DOT-USD","MATIC-USD"]
FLIP_COST = 0.0015         # 15bps per on/off flip
CACHE=os.path.join(os.path.dirname(__file__),"data","crypto"); os.makedirs(CACHE,exist_ok=True)
out=[]
def P(s=""): out.append(s); print(s, flush=True)

def _fetch_cb(prod):
    """Coinbase public daily candles, paginated 300/req from 2016, cached."""
    import urllib.request, json, time, datetime as dt
    path=os.path.join(CACHE,prod.replace("-","_")+".csv")
    if os.path.exists(path):
        return pd.read_csv(path,parse_dates=["t"]).set_index("t")["close"]
    rows={}; start=dt.datetime(2016,1,1); now=dt.datetime.utcnow()
    cur=start
    while cur<now:
        end=min(cur+dt.timedelta(days=290), now)
        u=(f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity=86400"
           f"&start={cur.isoformat()}Z&end={end.isoformat()}Z")
        try:
            req=urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"})
            d=json.load(urllib.request.urlopen(req,timeout=20))
            for c in d:  # [time, low, high, open, close, volume]
                rows[dt.datetime.utcfromtimestamp(c[0])]=c[4]
        except Exception:
            pass
        cur=end; time.sleep(0.25)
    if len(rows)<200:
        return None
    s=pd.Series(rows).sort_index(); s.index.name="t"; s.name="close"
    s.to_frame().to_csv(path)
    return s

def load():
    cols={}
    for p in PRODUCTS:
        s=_fetch_cb(p)
        if s is not None and len(s)>200:
            cols[p]=s
    df=pd.DataFrame(cols).sort_index()
    df.index=pd.to_datetime(df.index)
    return df.dropna(how="all")

def metrics(r, ppy=365):
    r=r.dropna()
    if len(r)<200: return None
    ann=r.mean()*ppy; vol=r.std()*np.sqrt(ppy); s=ann/vol if vol else 0
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min(); cg=eq.iloc[-1]**(ppy/len(r))-1
    return dict(cagr=cg, vol=vol, sr=s, dd=dd, n=len(r))

def trend_port(px, vol_target=None):
    """Equal-weight, long-only 200d-MA trend; flat (cash) when below. Net of flip cost."""
    rets=px.pct_change()
    sig=(px>px.rolling(200).mean()).astype(float).shift(1)   # decide on yesterday
    flips=sig.diff().abs().fillna(0)
    pos_ret = (sig*rets - flips*FLIP_COST)
    # equal-weight across coins that have data each day
    port = pos_ret.mean(axis=1)
    if vol_target:
        rv=port.rolling(30).std()*np.sqrt(365)
        lev=(vol_target/rv).shift(1).clip(0,3).fillna(0)
        port=port*lev
    return port.dropna()

def bh_port(px):
    return px.pct_change().mean(axis=1).dropna()

def line(lbl,r):
    m=metrics(r)
    if not m: P(f"  {lbl:<32} (insufficient)"); return
    h=len(r)//2; s1=metrics(r.iloc[:h]); s2=metrics(r.iloc[h:])
    rb="robust" if s1 and s2 and s1["sr"]>0 and s2["sr"]>0 else "fragile"
    P(f"  {lbl:<32}{m['cagr']*100:>8.1f}{m['vol']*100:>7.0f}{m['sr']:>7.2f}{m['dd']*100:>8.0f}   {rb}")

def run():
    P("Downloading crypto daily (yfinance)...")
    px=load()
    P(f"  {px.shape[1]} coins, {px.index[0].date()}..{px.index[-1].date()}\n")
    P("="*80)
    P(f"  {'strategy':<32}{'CAGR%':>8}{'vol%':>7}{'Sharpe':>7}{'maxDD%':>8}   robust")
    P("="*80)
    line("Buy-hold basket (equal-wt)", bh_port(px))
    line("Buy-hold BTC only", px["BTC-USD"].pct_change().dropna())
    line("TREND basket (200d MA)", trend_port(px))
    line("TREND basket + 40% vol-target", trend_port(px, 0.40))
    line("TREND basket + 60% vol-target", trend_port(px, 0.60))
    # BTC-only trend
    btc=px[["BTC-USD"]]
    line("TREND BTC only", trend_port(btc))
    P("\n  MAXIMISE RETURNS — trend basket levered to vol targets:")
    for t in [0.30,0.50,0.80]:
        m=metrics(trend_port(px,t))
        if m: P(f"    {int(t*100)}% vol -> CAGR {m['cagr']*100:.0f}%  Sharpe {m['sr']:.2f}  maxDD {m['dd']*100:.0f}%")
    with open(os.path.join(os.path.dirname(__file__),"logs","crypto_backtest.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\n  Saved -> logs/crypto_backtest.txt")

if __name__=="__main__":
    run()
