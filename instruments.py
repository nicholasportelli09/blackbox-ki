"""
Instrument spec + sizing layer — the asset-agnostic foundation that lets the
Blackbox engine trade ANY OANDA instrument (FX, metals, indices, bonds, ...),
not just the FX majors the original code hardcoded.

Everything is driven by the LIVE specs OANDA returns for the account
(displayPrecision, pipLocation, marginRate, tradeUnitsPrecision) plus live
quote->home(CHF) conversion, so position sizing and price formatting are correct
for gold (3 dp), the S&P (1 dp), a Bund (3 dp) and EUR/USD (5 dp) alike.

Account currency is CHF. P&L for N units of an instrument quoted in QC, when price
moves by `dist` (in price units), is:  N * dist * (QC->CHF).  We invert that to
size a position to a target CHF risk.
"""
import time
from oanda_client import OandaClient

_specs = None
_specs_ts = 0.0
_conv = {}            # quote ccy -> (factor, timestamp)
HOME = "CHF"          # account currency


def _client():
    return OandaClient()


def specs(client=None, ttl=3600):
    """All tradeable instrument specs, cached for an hour."""
    global _specs, _specs_ts
    if _specs is not None and time.time() - _specs_ts < ttl:
        return _specs
    c = client or _client()
    data = c._get(f"/v3/accounts/{c.account}/instruments")
    out = {}
    for i in data["instruments"]:
        ploc = int(i["pipLocation"])
        out[i["name"]] = {
            "type": i["type"],                                  # CURRENCY / METAL / CFD
            "precision": int(i["displayPrecision"]),
            "pip_location": ploc,
            "pip": 10.0 ** ploc,
            "margin_rate": float(i["marginRate"]),
            "unit_precision": int(i.get("tradeUnitsPrecision", 0)),
            "quote": i["name"].split("_")[1],
            "display": i.get("displayName", i["name"]),
        }
    _specs = out
    _specs_ts = time.time()
    return out


def spec(inst):
    return specs().get(inst)


def pip_size(inst):
    s = specs().get(inst)
    return s["pip"] if s else 0.0001


def precision(inst):
    s = specs().get(inst)
    return s["precision"] if s else 5


def fmt_price(inst, price):
    """Format a price to the instrument's real display precision (orders are
    REJECTED otherwise — e.g. SPX500 wants 1 dp, not 5)."""
    return f"{price:.{precision(inst)}f}"


def home_factor(quote, client=None, ttl=300):
    """quote currency -> account currency (CHF) conversion, cached 5 min."""
    if quote == HOME:
        return 1.0
    hit = _conv.get(quote)
    if hit and time.time() - hit[1] < ttl:
        return hit[0]
    c = client or _client()
    f = None
    for pair, inv in ((f"{quote}_{HOME}", False), (f"{HOME}_{quote}", True)):
        try:
            px = c.get_price(pair)
            f = (1.0 / px) if inv else px
            break
        except Exception:
            continue
    if f is None:
        f = _conv.get(quote, (1.0, 0))[0]   # last known, else 1.0
    _conv[quote] = (f, time.time())
    return f


def home_value_per_unit(inst, client=None):
    """CHF value of a 1.0-price move on 1 unit of `inst` (= QC->CHF factor)."""
    s = specs().get(inst)
    quote = s["quote"] if s else "USD"
    return home_factor(quote, client)


def round_units(inst, units):
    s = specs().get(inst)
    up = s["unit_precision"] if s else 0
    if up <= 0:
        return int(units)
    return round(float(units), up)


def position_size(balance_chf, inst, sl_price_distance, risk_pct, client=None):
    """Units sized so that hitting the stop loses ~risk_pct of balance (CHF).

    loss_chf(N) = N * sl_price_distance * home_value_per_unit
    -> N = risk_chf / (sl_price_distance * home_value_per_unit)
    """
    if sl_price_distance is None or sl_price_distance <= 0:
        return 0
    vpu = home_value_per_unit(inst, client)
    if vpu <= 0:
        return 0
    risk_chf = balance_chf * risk_pct
    units = risk_chf / (sl_price_distance * vpu)
    return round_units(inst, units)


def margin_per_unit(inst, price, client=None):
    """CHF margin to hold 1 unit at `price` (notional * marginRate, in CHF)."""
    s = specs().get(inst)
    if not s:
        return None
    return price * s["margin_rate"] * home_value_per_unit(inst, client)


def cap_units_by_margin(inst, units, price, margin_avail_chf, nav_chf=None,
                        buffer=0.7, per_trade_nav=0.12, client=None):
    """Shrink an order so its margin fits ~70% of free margin AND a per-trade NAV
    budget (default 12% NAV). Same discipline the FX bot uses, generalised."""
    mpu = margin_per_unit(inst, price, client)
    if not mpu or mpu <= 0:
        return int(units)
    budget = buffer * margin_avail_chf
    if nav_chf:
        budget = min(budget, per_trade_nav * nav_chf)
    cap = int(budget / mpu)
    if abs(units) > cap:
        return cap if units > 0 else -cap
    return int(units) if specs()[inst]["unit_precision"] == 0 else round_units(inst, units)


if __name__ == "__main__":
    # quick self-test (read-only): print specs + a sample sizing for each sleeve
    c = _client()
    acct = c.get_account()
    bal = float(acct["balance"])
    print(f"account balance {bal:,.0f} {acct.get('currency', HOME)}")
    sp = specs(c)
    samples = ["EUR_USD", "USD_JPY", "XAU_USD", "XAG_USD",
               "SPX500_USD", "NAS100_USD", "DE30_EUR", "UK100_GBP",
               "USB10Y_USD", "DE10YB_EUR", "UK10YB_GBP"]
    print(f"\n{'inst':<12}{'type':<9}{'prec':>5}{'pip':>9}{'mgn':>6}{'uprec':>6}"
          f"{'QC->CHF':>9}{'~units@1% (SL=2*pip*100)':>26}")
    for i in samples:
        s = sp.get(i)
        if not s:
            print(f"{i:<12} (not tradeable)"); continue
        f = home_factor(s["quote"], c)
        sl_dist = 100 * s["pip"]                       # nominal 100-pip stop
        u = position_size(bal, i, sl_dist, 0.01, c)
        print(f"{i:<12}{s['type']:<9}{s['precision']:>5}{s['pip']:>9.4g}"
              f"{s['margin_rate']:>6.2f}{s['unit_precision']:>6}{f:>9.4f}{u:>26,}")
