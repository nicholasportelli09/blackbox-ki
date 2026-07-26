import requests
import time
import pandas as pd
from datetime import datetime, timezone
from config import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL


class OandaClient:
    MAX_RETRIES = 3
    RETRY_STATUSES = {401, 408, 429}  # transient on OANDA practice (~3% of calls, 2026-06)
    TIMEOUT = 15  # seconds

    def __init__(self, account_id=None):
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_KEY}",
            "Content-Type": "application/json",
        }
        self.base = OANDA_BASE_URL
        # account_id lets each sleeve trade its OWN sub-account; defaults to the
        # primary (FX) account from config when not specified.
        self.account = account_id or OANDA_ACCOUNT_ID

    def _request(self, method, url, **kwargs):
        """HTTP wrapper with timeout + retry/backoff.

        OANDA's practice API intermittently returns spurious 401s (~3% of
        calls observed 2026-06; the token is valid the other 97% of the time).
        Without retries a single flake raised straight out of the cycle,
        aborting stop management for ALL open trades that cycle. We retry
        transient statuses (401/408/429) plus 5xx and network errors; genuine
        4xx (bad order, 404) still fail fast.
        """
        kwargs.setdefault("timeout", self.TIMEOUT)
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                r = requests.request(method, url, headers=self.headers, **kwargs)
                r.raise_for_status()
                return r
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status is not None and status < 500 and status not in self.RETRY_STATUSES:
                    raise
                last_exc = e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exc = e
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
        raise last_exc

    def _get(self, path, params=None):
        return self._request("GET", f"{self.base}{path}", params=params).json()

    def _post(self, path, body):
        return self._request("POST", f"{self.base}{path}", json=body).json()

    # ── Data ──────────────────────────────────────────────────────────────

    def get_candles(self, instrument, granularity, count=500):
        data = self._get(
            f"/v3/instruments/{instrument}/candles",
            params={"granularity": granularity, "count": count, "price": "M"},
        )
        rows = []
        for c in data["candles"]:
            if c["complete"]:
                m = c["mid"]
                rows.append({
                    "timestamps": pd.Timestamp(c["time"]),
                    "open":   float(m["o"]),
                    "high":   float(m["h"]),
                    "low":    float(m["l"]),
                    "close":  float(m["c"]),
                    "volume": int(c["volume"]),
                    "amount": float(m["c"]) * int(c["volume"]),
                })
        df = pd.DataFrame(rows).set_index("timestamps")
        df.index = df.index.tz_localize(None)
        return df

    def get_candles_range(self, instrument, granularity, start, end):
        """Candles between two RFC3339 datetimes (for historical crisis windows).
        Keep ranges under ~5000 candles per call (OANDA cap)."""
        try:
            data = self._get(
                f"/v3/instruments/{instrument}/candles",
                params={"granularity": granularity, "from": start, "to": end, "price": "M"},
            )
        except Exception:
            return pd.DataFrame()
        rows = []
        for c in data.get("candles", []):
            if c["complete"]:
                m = c["mid"]
                rows.append({"timestamps": pd.Timestamp(c["time"]),
                             "open": float(m["o"]), "high": float(m["h"]),
                             "low": float(m["l"]), "close": float(m["c"]),
                             "volume": int(c["volume"])})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("timestamps")
        df.index = df.index.tz_localize(None)
        return df

    def get_account(self):
        return self._get(f"/v3/accounts/{self.account}/summary")["account"]

    def get_open_trades(self):
        return self._get(f"/v3/accounts/{self.account}/openTrades")["trades"]

    def get_open_orders(self):
        return self._get(f"/v3/accounts/{self.account}/pendingOrders")["orders"]

    def get_closed_trades(self, count=500):
        """
        Closed-trade history reconstructed from the TRANSACTION LEDGER.
        (OANDA's /trades?state=CLOSED endpoint silently omits most closed
        trades, discovered 2026-06-11 — the ledger is the source of truth.)
        Returns OANDA-trade-shaped dicts: id, instrument, price, openTime,
        closeTime, initialUnits, averageClosePrice, realizedPL.
        """
        # Pull the COMPLETE ledger via idrange. The /transactions LIST endpoint
        # type-filters and paginates incompletely — it silently returned only
        # 18 of 40 closed trades (found 2026-06-19). idrange 1..lastID is the
        # authoritative full ledger; OANDA caps each idrange call at 1000 ids.
        meta = self._get(f"/v3/accounts/{self.account}/transactions")
        last_id = int(meta.get("lastTransactionID", 0))
        opens, closes = {}, {}
        start = 1
        while start <= last_id:
            end = min(start + 999, last_id)
            txs = self._get(f"/v3/accounts/{self.account}/transactions/idrange",
                            params={"from": start, "to": end}).get("transactions", [])
            for t in txs:
                if t.get("type") != "ORDER_FILL":
                    continue
                to = t.get("tradeOpened")
                if to:
                    opens[to["tradeID"]] = {
                        "instrument": t.get("instrument", ""),
                        "price": float(to.get("price", t.get("price", 0))),
                        "openTime": t["time"],
                        "initialUnits": float(to.get("units", t.get("units", 0))),
                    }
                closed_list = list(t.get("tradesClosed") or [])
                if t.get("tradeReduced"):
                    closed_list.append(t["tradeReduced"])
                for tc in closed_list:
                    tid = tc["tradeID"]
                    rec = closes.setdefault(tid, {"realizedPL": 0.0})
                    rec["realizedPL"] += float(tc.get("realizedPL", 0))
                    rec["closeTime"] = t["time"]
                    rec["closePrice"] = float(tc.get("price", t.get("price", 0)))
            start = end + 1
        out = []
        for tid, c in closes.items():
            o = opens.get(tid)
            if not o:
                continue
            out.append({
                "id": tid,
                "instrument": o["instrument"],
                "price": str(o["price"]),
                "openTime": o["openTime"],
                "closeTime": c.get("closeTime", ""),
                "initialUnits": str(o["initialUnits"]),
                "averageClosePrice": str(c.get("closePrice", 0)),
                "realizedPL": str(c["realizedPL"]),
            })
        out.sort(key=lambda x: x["closeTime"])
        return out[-count:]

    def close_trade(self, trade_id):
        return self._request(
            "PUT", f"{self.base}/v3/accounts/{self.account}/trades/{trade_id}/close"
        ).json()

    def set_trade_stop(self, trade_id, instrument, price):
        """Modify the stop-loss on an open trade (used for breakeven/trailing moves)."""
        body = {"stopLoss": {"price": self._fmt_price(instrument, price), "timeInForce": "GTC"}}
        return self._request(
            "PUT", f"{self.base}/v3/accounts/{self.account}/trades/{trade_id}/orders",
            json=body,
        ).json()

    # ── Orders ────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_price(instrument: str, price: float) -> str:
        # Use the instrument's real display precision (gold=3, SPX500=1, Bund=3,
        # EUR/USD=5, USD/JPY=3). Orders are REJECTED if the precision is wrong.
        # Lazy import + fallback keeps FX behaviour byte-identical if specs are
        # unavailable (precision matches the old 3/5 rule for FX majors).
        try:
            from instruments import precision
            decimals = precision(instrument)
        except Exception:
            decimals = 3 if "JPY" in instrument else 5
        return f"{price:.{decimals}f}"

    def market_order(self, instrument, units, stop_loss_price=None, take_profit_price=None):
        order = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
        }
        if stop_loss_price:
            order["stopLossOnFill"] = {"price": self._fmt_price(instrument, stop_loss_price)}
        if take_profit_price:
            order["takeProfitOnFill"] = {"price": self._fmt_price(instrument, take_profit_price)}
        body = {"order": order}
        resp = self._post(f"/v3/accounts/{self.account}/orders", body)
        if "orderCancelTransaction" in resp:
            reason = resp["orderCancelTransaction"].get("reason", "CANCELLED")
            raise RuntimeError(f"order cancelled by OANDA: {reason}")
        return resp

    def margin_rates(self):
        """Per-instrument margin rates (cached)."""
        if not hasattr(self, "_margin_rates"):
            data = self._get(f"/v3/accounts/{self.account}/instruments")
            self._margin_rates = {i["name"]: float(i["marginRate"]) for i in data["instruments"]}
        return self._margin_rates

    def close_all_trades(self):
        trades = self.get_open_trades()
        results = []
        for t in trades:
            r = self._request(
                "PUT", f"{self.base}/v3/accounts/{self.account}/trades/{t['id']}/close"
            )
            results.append(r.json())
        return results

    def get_price(self, instrument):
        data = self._get(f"/v3/accounts/{self.account}/pricing", params={"instruments": instrument})
        p = data["prices"][0]
        return (float(p["asks"][0]["price"]) + float(p["bids"][0]["price"])) / 2
