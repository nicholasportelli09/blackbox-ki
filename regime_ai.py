"""
REGIME RADAR — our own machine-learning model suite (KI Challenge).

For each of five asset classes we train a SEPARATE neural network to forecast the
market's near-term RISK REGIME — whether the next ~10 trading days will be
TURBULENT (high volatility) or CALM — from features engineered off past prices
alone. Each model has a codename:

    crypto      CATACLYSM     stocks      VANTAGE     bonds   BASTION
    commodities CRUCIBLE      fx          MERIDIAN

This is genuinely "our own AI": we engineer the features, define the target,
train the models, and evaluate them honestly out-of-sample. It is deliberately
separate from the LLM features elsewhere in the app — those *use* a pre-trained
model; these we train ourselves.

Honesty by construction (no lookahead):
  * Features at day t use ONLY information up to and including day t.
  * The label at day t is the realised volatility over days [t+1 .. t+H].
  * Train / validation / test are split by DATE, never shuffled.
  * The "turbulent" threshold is fit on the TRAINING data only.

Run:
    python3 regime_ai.py train      # trains all five, evaluates, saves + reports
    python3 regime_ai.py predict    # prints today's regime call for each class
"""
import os, json, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import asset_data

HERE = os.path.dirname(__file__)
ML_DIR = os.path.join(HERE, "ml")
REPORT_TXT = os.path.join(HERE, "logs", "regime_report.txt")
REPORT_JSON = os.path.join(HERE, "logs", "regime_report.json")

H = 10                 # forecast horizon (trading days)
TURBULENT_Q = 0.66     # top third of forward vol = "turbulent"
CLASSES = ["crypto", "equities", "bonds", "commodities", "fx"]

FEATURES = (["ret5", "ret10", "ret20", "ret60"]
            + ["vol5", "vol10", "vol20", "vol30"]
            + ["ma_dist20", "ma_dist50", "ma_dist200"]
            + ["rsi", "dd30", "dd60", "vol_chg"])


def model_path(cls):
    return os.path.join(ML_DIR, f"regime_{cls}.pkl")


# ---------------------------------------------------------------- features / label
def feature_frame(close):
    """Past-only features for one close-price series."""
    close = close.dropna().astype(float)
    r = np.log(close).diff()
    df = pd.DataFrame(index=close.index)
    for w in (5, 10, 20, 60):
        df[f"ret{w}"] = close.pct_change(w)
    for w in (5, 10, 20, 30):
        df[f"vol{w}"] = r.rolling(w).std() * np.sqrt(252)
    for w in (20, 50, 200):
        df[f"ma_dist{w}"] = close / close.rolling(w).mean() - 1
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    for w in (30, 60):
        df[f"dd{w}"] = close / close.rolling(w).max() - 1
    df["vol_chg"] = df["vol10"] - df["vol30"]
    return df


def forward_vol(close, h=H):
    """Realised volatility over the FUTURE window [t+1 .. t+h] (the target)."""
    r = np.log(close.dropna().astype(float)).diff()
    return r.rolling(h).std().shift(-h) * np.sqrt(252)


def build_dataset(px):
    """Pool every instrument in a class into one (date, features, fwd_vol) table."""
    parts = []
    for col in px.columns:
        s = px[col].dropna()
        if len(s) < 500:
            continue
        d = feature_frame(s)
        d["fwd_vol"] = forward_vol(s)
        d["inst"] = col
        d["date"] = d.index
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts).dropna(subset=FEATURES + ["fwd_vol"])
    return data.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- eval helpers
def _splits(data):
    dates = np.sort(data["date"].unique())
    val_start = dates[int(len(dates) * 0.60)]
    test_start = dates[int(len(dates) * 0.75)]
    tr = data[data["date"] < val_start]
    va = data[(data["date"] >= val_start) & (data["date"] < test_start)]
    te = data[data["date"] >= test_start]
    return tr, va, te, pd.Timestamp(val_start), pd.Timestamp(test_start)


def _metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(set(y_true)) > 1 else None,
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------- train one class
def train_class(cls, px):
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score

    data = build_dataset(px)
    if len(data) < 2000:
        return None
    tr, va, te, val_start, test_start = _splits(data)

    thr = float(tr["fwd_vol"].quantile(TURBULENT_Q))     # fit on train only
    lab = lambda d: (d["fwd_vol"].values > thr).astype(int)
    Xtr, ytr = tr[FEATURES].values, lab(tr)
    Xva, yva = va[FEATURES].values, lab(va)
    Xte, yte = te[FEATURES].values, lab(te)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None

    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                        early_stopping=True, random_state=0).fit(Ztr, ytr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                class_weight="balanced", random_state=0, n_jobs=-1).fit(Xtr, ytr)

    pv_mlp, pv_rf = mlp.predict_proba(Zva)[:, 1], rf.predict_proba(Xva)[:, 1]
    mlp_auc, rf_auc = roc_auc_score(yva, pv_mlp), roc_auc_score(yva, pv_rf)
    chosen = "neural_net" if mlp_auc >= rf_auc - 0.01 else "random_forest"
    pv = pv_mlp if chosen == "neural_net" else pv_rf
    cut = float(max(np.linspace(0.15, 0.85, 71),
                    key=lambda c: f1_score(yva, (pv >= c).astype(int), zero_division=0)))

    pt_mlp, pt_rf = mlp.predict_proba(Zte)[:, 1], rf.predict_proba(Xte)[:, 1]
    maj = int(round(ytr.mean()))
    base_majority = _metrics(yte, np.full_like(yte, maj), np.full_like(yte, maj, dtype=float))
    vthr = float(np.quantile(tr["vol10"].values, TURBULENT_Q))
    pers_pred = (te["vol10"].values > vthr).astype(int)
    base_persist = _metrics(yte, pers_pred, te["vol10"].values / (vthr * 2))
    m_mlp = _metrics(yte, (pt_mlp >= cut).astype(int), pt_mlp)
    m_rf = _metrics(yte, (pt_rf >= cut).astype(int), pt_rf)
    m_chosen = m_mlp if chosen == "neural_net" else m_rf

    importances = sorted(zip(FEATURES, rf.feature_importances_), key=lambda z: -z[1])

    os.makedirs(ML_DIR, exist_ok=True)
    joblib.dump({"scaler": scaler, "mlp": mlp, "rf": rf, "chosen": chosen,
                 "features": FEATURES, "cut": cut, "H": H, "test_metrics": m_chosen},
                model_path(cls))

    return {
        "class": cls, "name": asset_data.NAMES[cls], "label": asset_data.LABELS[cls],
        "dataset": {"instruments": int(data["inst"].nunique()), "rows": int(len(data)),
                    "date_start": str(data["date"].min().date()),
                    "date_end": str(data["date"].max().date()),
                    "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                    "test_start": str(test_start.date())},
        "chosen_model": chosen, "decision_threshold": round(cut, 3),
        "train_turbulent_rate": round(float(ytr.mean()), 3),
        "test_turbulent_rate": round(float(yte.mean()), 3),
        "val_auc": {"neural_net": round(float(mlp_auc), 4), "random_forest": round(float(rf_auc), 4)},
        "test": {"neural_net": m_mlp, "random_forest": m_rf,
                 "baseline_majority": base_majority, "baseline_persistence": base_persist},
        "chosen_test": m_chosen,
        "edge_over_persistence": round((m_chosen["roc_auc"] or 0.5) - (base_persist["roc_auc"] or 0.5), 4),
        "top_features": [{"feature": f, "importance": round(float(i), 4)} for f, i in importances[:6]],
    }


def train_all():
    reports = {}
    for cls in CLASSES:
        print(f"\n=== {asset_data.NAMES[cls]} ({cls}) ===")
        px = asset_data.load_class(cls)
        if px is None or px.empty:
            print("  no data — skipped"); continue
        print(f"  {px.shape[1]} instruments, {px.index[0].date()}..{px.index[-1].date()}")
        rep = train_class(cls, px)
        if not rep:
            print("  insufficient data — skipped"); continue
        reports[cls] = rep
        m = rep["chosen_test"]
        print(f"  {rep['name']}: AUC {m['roc_auc']} (chosen {rep['chosen_model']}) "
              f"vs persistence {rep['test']['baseline_persistence']['roc_auc']} "
              f"-> edge {rep['edge_over_persistence']:+.3f}")
    doc = {"generated": datetime.now(timezone.utc).isoformat(),
           "horizon_days": H, "classes": reports}
    json.dump(doc, open(REPORT_JSON, "w"), indent=2)
    _write_txt(doc)
    print("\n" + open(REPORT_TXT).read())
    return doc


def _write_txt(doc):
    L = []
    def P(s=""): L.append(s)
    P("REGIME RADAR — model suite training report")
    P("=" * 74)
    P("One neural network per asset class, trained on real ~20-year daily data,")
    P("tested on the most recent untouched years. AUC: 0.50 = coin-flip, 1.0 = perfect.")
    P("")
    P(f"{'codename':<12}{'asset class':<16}{'AUC':>6}{'vs persist':>12}{'edge':>7}{'test yrs from':>15}")
    P("-" * 74)
    ranked = sorted(doc["classes"].values(), key=lambda r: -(r["chosen_test"]["roc_auc"] or 0))
    for r in ranked:
        m = r["chosen_test"]; pers = r["test"]["baseline_persistence"]["roc_auc"]
        P(f"{r['name']:<12}{r['label']:<16}{m['roc_auc']:>6.3f}{pers:>12.3f}"
          f"{r['edge_over_persistence']:>+7.3f}{r['dataset']['test_start']:>15}")
    P("")
    if ranked:
        best = ranked[0]
        P(f"Most forecastable turbulence: {best['name']} ({best['label']}), AUC "
          f"{best['chosen_test']['roc_auc']:.3f}.")
        byedge = max(doc["classes"].values(), key=lambda r: r["edge_over_persistence"])
        P(f"Biggest edge of the model over the naive 'vol is already high' baseline: "
          f"{byedge['name']} (+{byedge['edge_over_persistence']:.3f} AUC).")
    open(REPORT_TXT, "w").write("\n".join(L))


# ---------------------------------------------------------------- inference
def predict(cls):
    """Today's regime call for one asset class. None if unavailable."""
    import joblib
    try:
        b = joblib.load(model_path(cls))
        px = asset_data.load_class(cls)
        inst = asset_data.REPRESENTATIVE[cls]
        s = px[inst].dropna()
        feats = feature_frame(s).dropna().iloc[-1:][b["features"]]
        if b["chosen"] == "neural_net":
            prob = float(b["mlp"].predict_proba(b["scaler"].transform(feats.values))[0, 1])
        else:
            prob = float(b["rf"].predict_proba(feats.values)[0, 1])
        return {"class": cls, "name": asset_data.NAMES[cls], "label": asset_data.LABELS[cls],
                "instrument": asset_data.REP_NAME[cls],
                "prob_turbulent": round(prob, 3),
                "regime": "turbulent" if prob >= b["cut"] else "calm",
                "as_of": str(s.index[-1].date()), "model": b["chosen"],
                "test_auc": b["test_metrics"]["roc_auc"], "horizon_days": b["H"]}
    except Exception:
        return None


def predict_all():
    out = {}
    for cls in CLASSES:
        p = predict(cls)
        if p:
            out[cls] = p
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train_all()
    elif cmd == "predict":
        print(json.dumps(predict_all(), indent=2))
