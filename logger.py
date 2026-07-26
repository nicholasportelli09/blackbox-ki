import csv
import os
import logging
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TRADE_LOG = os.path.join(LOG_DIR, "trades.csv")
BOT_LOG = os.path.join(LOG_DIR, "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BOT_LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bot")


def log_trade(action, instrument, units, entry_price, sl, tp, signal, balance, note=""):
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "action", "instrument", "units",
                        "entry_price", "sl", "tp", "signal", "balance", "note"])
        w.writerow([
            datetime.utcnow().isoformat(), action, instrument, units,
            entry_price, sl, tp, signal, balance, note,
        ])


def log_cycle(balance, peak_balance, signal, open_trades):
    log.info(
        f"Balance={balance:.2f} | Peak={peak_balance:.2f} | "
        f"Signal={signal:+d} | OpenTrades={len(open_trades)}"
    )
