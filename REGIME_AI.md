# Regime Radar — our own machine-learning model suite

*Critical reflection and documentation for the KI Challenge 2026.*

This document describes the part of Blackbox that is **genuinely our own AI**: a
suite of **five neural networks we designed, trained and evaluated ourselves** —
one per asset class. Every other "AI" feature in the app *uses* a pre-trained
large language model through an API; Regime Radar is the opposite — we built these
from raw data up. It exists so the project's methods come from the field of
machine learning, as the challenge requires.

Each model has a codename:

| Codename | Asset class | Data |
|---|---|---|
| **Cataclysm** | Crypto | 13 coins, Coinbase |
| **Vantage** | Equities | 8 global stock indices, OANDA |
| **Bastion** | Bonds & rates | 6 government-bond futures, OANDA |
| **Crucible** | Commodities | 12 metals/energy/agri, OANDA |
| **Meridian** | FX / currencies | 10 major pairs, OANDA |

---

## 1. The problem

Markets move between **calm** periods and **turbulent** ones (high volatility).
Turbulence is when investors get hurt. If a model can see turbulence coming even
slightly better than chance, that is useful — you can lower risk *before* the
storm, not after.

**Task.** For each asset class, a binary classification: *will the next 10 trading
days be turbulent (high realised volatility) or calm?* — predicted only from what
is knowable today.

We chose volatility rather than price *direction* deliberately. Predicting up/down
is near-impossible (markets are near-efficient), so such a model would honestly be
a coin-flip. Volatility, by contrast, **clusters** — so it is genuinely, if
imperfectly, learnable. Choosing a learnable target is itself part of doing honest
machine learning.

## 2. The data

- **~20 years of real daily prices** (2006→2026) for equities, bonds, commodities
  and FX, from **OANDA**; **~10 years** for crypto (2016→) from **Coinbase**.
- Pooling every instrument in a class gives large datasets — **over 220,000
  labelled examples in total** (crypto 29.6k, equities 40.7k, bonds 30.4k,
  commodities 63.3k, FX 57.0k).

## 3. Method (identical for every model)

**Features (15, all past-only).** For each day we compute inputs describing the
market's recent behaviour, using *only* information available up to that day:
multi-window returns (5/10/20/60d), realised volatility (5/10/20/30d), distance
from moving averages (20/50/200d), RSI, and trailing drawdown (30/60d). Preventing
future information from leaking into the features is the single most important
thing to get right in financial ML.

**Label.** For day *t*, the realised volatility over the **future** window
*[t+1 … t+10]*, labelled **turbulent** if it lands in the top third of the
training era's volatility. The threshold is fit on **training data only**.

**Split (by time, never shuffled).** Train = oldest ~60 %, validation = next
~15 %, test = **most-recent ~25 %, untouched**. Shuffling would let a forecaster
peek at the future.

**Models.** A **neural network** (multi-layer perceptron, hidden layers 32→16) and
a **random forest** as a cross-check. Features standardised; classes balanced; the
decision threshold tuned on the **validation** set. The better model on validation
is deployed per class.

## 4. Results — honestly, out-of-sample

Measured on each class's untouched recent years. AUC runs from 0.50 (a coin-flip)
to 1.00 (perfect) and is the right metric under class imbalance.

| Model | Asset class | Test AUC | Naive baseline AUC | Edge |
|---|---|---|---|---|
| **Bastion** | Bonds & rates | **0.93** | 0.91 | +0.024 |
| **Crucible** | Commodities | **0.85** | 0.81 | +0.034 |
| **Vantage** | Equities | **0.82** | 0.77 | +0.045 |
| **Meridian** | FX | **0.81** | 0.74 | +0.068 |
| **Cataclysm** | Crypto | **0.70** | 0.65 | +0.055 |

**Two honest readings of this table.**

1. **Every model beats its baselines.** The "naive baseline" is a strong rule —
   *"volatility is already high, so expect turbulence"* (persistence). Beating it
   is the real test, because volatility clustering is genuinely powerful. Every one
   of our models beats both that rule and a coin-flip. The **edge** column is the
   honest measure of what the machine learning *adds* on top of persistence.

2. **The edge is real but modest, and the high absolute AUCs are mostly the market,
   not us.** Bonds score 0.93 — but persistence alone already scores 0.91, so most
   of that number is just "calm markets stay calm." The largest genuine machine-
   learning contribution is in **FX (+0.068)** and **crypto (+0.055)**. We say this
   plainly rather than claiming a 0.93 as our achievement.

## 5. The cross-asset finding

Training the same method on five markets reveals a clear pattern:

> **The steadier the market, the more predictable its turbulence.** Bond/rate
> turbulence is the most forecastable (AUC 0.93); crypto — the wildest, most
> headline-driven market — is the least (0.70).

This makes economic sense: bond volatility is driven by slow, telegraphed forces
(central-bank policy, the rate cycle), whereas crypto turbulence is often triggered
by sudden, unforecastable news. Discovering this by running one honest method
across five datasets is exactly the kind of result a cross-asset study should
produce.

## 6. Error analysis — where the models fail

- **They miss sudden, news-driven spikes.** A shock with no footprint in past
  prices (a hack, a surprise announcement) cannot be seen coming from price
  features alone. This hits crypto hardest.
- **They raise false alarms** in choppy-but-not-turbulent stretches. For a *risk*
  tool this is the safer error (de-risking unnecessarily beats missing a storm),
  but it is a real cost.
- **Regime shift over time** (clearest in crypto): the training era was more
  volatile than the recent test era, so a model trained on a wilder past is
  slightly miscalibrated for a calmer present.

## 7. Limitations & honest caveats

- **A modest edge, not a crystal ball.** Even the best model is imperfect. We
  present these as *risk signals*, never trade triggers — they do **not** drive any
  buying or selling in the app; the validated strategies do that.
- **Volatility ≠ direction.** The models say how bumpy the ride will be, not which
  way prices go.
- **High baselines.** Because volatility is so persistent, the naive baseline is
  hard to beat; our contribution is the incremental edge, which we report honestly.

## 8. What we would do next

- Add non-price features (volume, the app's own news-sentiment scores, cross-asset
  signals) — turbulence often shows up in volume and mood first.
- Try a sequence model (LSTM / temporal CNN) that reads the recent path directly.
- Recalibrate on rolling windows so each model adapts as its volatility regime
  shifts.

## 9. How to run the AI

```bash
pip install -r requirements.txt      # includes scikit-learn
python3 regime_ai.py train           # trains all five models, evaluates, saves + reports
python3 regime_ai.py predict         # prints today's regime call for each asset class
```

Training needs the OANDA practice key in `.env` for the non-crypto data (crypto is
free from Coinbase). Models are saved to `ml/regime_<class>.pkl`; the full report
is in `logs/regime_report.txt` / `.json`. In the web app, the five live predictions
and the honest cross-asset scorecard appear on the **Intelligence** page
("Regime Radar"), with the method on the **AI Inside** page.

## 10. On honesty

Everything here is reported as measured. We show the baselines each model must
beat, we distinguish the market's own predictability from our model's added edge,
and we show where the models fail. Five models that honestly beat their baselines
on real, unseen data — and that we can explain, compare and critique — are worth
far more than one impressive number we could not defend.
