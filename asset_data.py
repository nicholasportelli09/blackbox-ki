"""
Multi-asset market data for the Regime Radar model suite.

Five asset classes, each with its own codename and its own real dataset:
    crypto      -> CATACLYSM   (Coinbase public API, free)
    equities    -> VANTAGE     (OANDA — global stock indices)
    bonds       -> BASTION     (OANDA — government bond futures / rates)
    commodities -> CRUCIBLE    (OANDA — metals, energy, agriculture)
    fx          -> MERIDIAN    (OANDA — major currency pairs)

Non-crypto classes use OANDA's daily candles (~20 years, 2006→), fetched and
cached by backtest_factors.fetch_daily(). Crypto uses the Coinbase cache.
"""
import os
import pandas as pd

HERE = os.path.dirname(__file__)

NAMES = {"crypto": "Cataclysm", "equities": "Vantage", "bonds": "Bastion",
         "commodities": "Crucible", "fx": "Meridian"}

LABELS = {"crypto": "Crypto", "equities": "Equities", "bonds": "Bonds & rates",
          "commodities": "Commodities", "fx": "FX / currencies"}

# OANDA instruments per class (20y daily history each)
UNIVERSE = {
    "equities":    ["SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR",
                    "UK100_GBP", "JP225_USD", "EU50_EUR", "AU200_AUD"],
    "bonds":       ["USB02Y_USD", "USB05Y_USD", "USB10Y_USD", "USB30Y_USD",
                    "DE10YB_EUR", "UK10YB_GBP"],
    "commodities": ["XAU_USD", "XAG_USD", "XPT_USD", "XPD_USD", "XCU_USD",
                    "WTICO_USD", "BCO_USD", "NATGAS_USD", "CORN_USD",
                    "WHEAT_USD", "SOYBN_USD", "SUGAR_USD"],
    "fx":          ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
                    "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"],
}

# the single instrument used for each class's live "today" call + friendly name
REPRESENTATIVE = {"crypto": "BTC-USD", "equities": "SPX500_USD",
                  "bonds": "USB10Y_USD", "commodities": "XAU_USD", "fx": "EUR_USD"}
REP_NAME = {"crypto": "Bitcoin", "equities": "S&P 500", "bonds": "US 10-year",
            "commodities": "Gold", "fx": "EUR/USD"}


def load_class(cls):
    """Daily-close DataFrame (one column per instrument) for one asset class."""
    if cls == "crypto":
        from crypto_backtest import load
        return load()
    import backtest_factors as bf
    cols = {}
    for inst in UNIVERSE.get(cls, []):
        try:
            df = bf.fetch_daily(inst)
            if df is not None and len(df) > 300:
                cols[inst] = df["close"]
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.to_datetime(px.index)
    return px.dropna(how="all")


if __name__ == "__main__":
    for cls in ["crypto"] + list(UNIVERSE):
        df = load_class(cls)
        if df is None or df.empty:
            print(f"{cls:<12} {NAMES[cls]:<10} NO DATA")
        else:
            print(f"{cls:<12} {NAMES[cls]:<10} {df.shape[1]} series, "
                  f"{df.index[0].date()} .. {df.index[-1].date()}, "
                  f"{int(df.notna().sum().sum()):,} obs")
