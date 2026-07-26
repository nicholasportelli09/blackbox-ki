"""
Sentiment scanner daemon — refreshes logs/sentiment.json every 20 minutes.
Feeds the user app's AI-intelligence panel and the bots' shadow advisories.
"""
import time, traceback
from logger import log
from sentiment_scanner import run_scan

CYCLE_MIN = 20

def main():
    log.info("Sentiment scanner daemon starting (every %d min; settles user books after 21:05 UTC)" % CYCLE_MIN)
    while True:
        try:
            d = run_scan()
            snap = ", ".join(f"{a}:{v['score']:+.0f}" for a, v in d["assets"].items())
            log.info(f"sentiment scan: {d['n_scanned']} items | {snap}")
        except Exception:
            log.error("scanner cycle failed:")
            log.error(traceback.format_exc())
        # daily close-of-book settle for the user app's paper portfolios
        try:
            from user_engine import settle_users
            n = settle_users()
            if n:
                log.info(f"user books settled at daily close: {n}")
        except Exception:
            log.error("user settle failed:")
            log.error(traceback.format_exc())
        # daily heartbeat push — proves the machine is alive while unattended.
        # heartbeat.py is optional (not shipped); skip quietly if absent.
        try:
            from heartbeat import maybe_send
            hb = maybe_send()
            if hb:
                log.info(f"heartbeat sent: {hb[0]}")
        except ImportError:
            pass
        except Exception:
            log.error("heartbeat failed:")
            log.error(traceback.format_exc())
        # weekly guarded ML retrain (Sundays 22:00+ UTC; adopt only if OOS holds).
        # retrain_ml.py is optional (not shipped); skip quietly if absent.
        try:
            from retrain_ml import maybe_retrain
            r = maybe_retrain()
            if r:
                log.info("ML weekly retrain: " + "; ".join(f"{k}: {v}" for k, v in r.items()))
        except ImportError:
            pass
        except Exception:
            log.error("ML retrain failed:")
            log.error(traceback.format_exc())
        time.sleep(CYCLE_MIN * 60)

if __name__ == "__main__":
    main()
