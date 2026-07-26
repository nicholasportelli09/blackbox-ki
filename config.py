import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
OANDA_BASE_URL = (
    "https://api-fxpractice.oanda.com"
    if OANDA_ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)

# Groq (AI features) — optional, features degrade gracefully without it
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Strategy
# 7 USD majors + 3 NON-USD HEDGE crosses re-added 2026-06-27.
# The JPY crosses stay cut (correlated to the dollar book via USD_JPY, weak edge).
# These 3 were re-validated with REALISTIC spreads (crosses_retest.py) and have a
# real net edge PLUS negative/low correlation to the USD book — they hedge the
# dollar over-concentration that majors-only created: EUR_AUD corr -0.57, GBP_AUD
# -0.43, AUD_NZD -0.10. Currency cap (+/-4 any ccy) still bounds concentration.
# USD_CHF DROPPED 2026-06-30 after the 12-year backtest: -0.009R expectancy,
# -54% drawdown at 1% risk — net-negative over a full cycle. Existing positions
# ride to managed exits (manage_open_trades handles all open trades; entries stop).
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_AUD", "GBP_AUD", "AUD_NZD",
]
INSTRUMENT  = "EUR_USD"       # fallback / single-pair references
GRANULARITY = "H4"
GRANULARITY_DAILY = "D"
LOOKBACK_CANDLES = 100
LOOKBACK_DAILY   = 60
PRED_LEN = 8

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Kronos directional threshold — forecast must point >X% to confirm
KRONOS_THRESHOLD = 0.0003    # 3 pips on EUR/USD

# Risk management
RISK_PER_TRADE     = 0.006   # 0.6%/trade (halved from 1.25% on 2026-07-20: FX is the ungated "adaptive lab" and the book's worst bleeder at -8.7%; user-agreed cut until the regime improves. History: 3% -> 1.5% -> 2% -> 1.25% -> 0.6%)
MAX_PORTFOLIO_RISK = 0.07    # total at-risk cap 7% NAV — keeps severe drawdowns clear of the 25% halt
MAX_OPEN_TRADES  = 10        # distinct pairs at once (portfolio-risk budget is the real constraint)
STOP_LOSS_PIPS   = 40
TAKE_PROFIT_PIPS = 100       # 2.5:1 reward/risk
MAX_DRAWDOWN_PCT = 0.25      # 25% drawdown halt (demo has more room)

# Kronos model — mini is 4.1M params, context 2048, much faster on CPU
KRONOS_MODEL = "NeoQuasar/Kronos-mini"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-2k"
KRONOS_PATH = os.getenv("KRONOS_PATH", "")  # optional local Kronos checkout; unused by the web app
