"""
trade_logic.py - all the 'brains' behind the dashboard (the API just calls these functions).

TWO DATA SOURCES
  source="db"     : the newest real candles in your PostgreSQL database (see db_data.py). Live-style signals.
  source="replay" : steps through the TEST period (data the model never trained on), one candle at a time.
                    Honest for demos: every signal shown is a genuine out-of-sample prediction.

Verdict = traffic light built from four honest checks:
  1. Direction   : this tool only considers BUYING (spot, long-only). A 'down' lean means stay out.
  2. Fees        : is the typical move over the next ~15 candles bigger than the round-trip fee?
  3. Market      : is the market unusually quiet?
  4. Confidence  : how does this prediction rank against the model's usual confidence?
"""
import functools
import os
import time

import joblib
import numpy as np
import pandas as pd

import backtest as bt
from model_utils import FEATURE_COLUMNS, load_split

FEE = 0.001          # 0.10% per side
HORIZON = 15         # candles
STOP_MULT = 2.0
TARGET_MULT = 3.0

GLOSSARY = {
    "rsi": "RSI: how fast price has been rising or falling lately (0-100). Above 70 = risen a lot, below 30 = fallen a lot.",
    "macd": "MACD: compares a fast and a slow average of price. Fast above slow suggests upward momentum.",
    "sma/ema": "Moving averages: the average price over recent candles. Price above the average = short-term uptrend.",
    "bollinger": "Bollinger Bands: a band around the average showing the normal price range. Near the top edge = stretched upward.",
    "volume": "Volume: how much is being traded. Higher than usual = more activity behind the move.",
    "shap": "SHAP: shows how much each indicator pushed the model towards 'up' or 'down' for this prediction.",
    "stop-loss": "Stop-loss: a price where you exit to cap your loss if the trade goes wrong.",
    "take-profit": "Take-profit: a price where you exit to lock in a gain.",
    "position size": "Position size: how much you buy, chosen so that a stop-out loses only a set % of your account.",
    "fees": "Fees: charged when you buy AND when you sell (0.10% each way here), so a round trip costs 0.20%.",
    "volatility": "Volatility: how much the price typically moves. In our tests, whether the NEXT move is big or small was much easier to predict (AUC 0.64-0.70) than whether it is up or down (about 0.52).",
    "auc": "AUC: how well a model ranks 'up' above 'down'. 0.5 = coin flip, 1.0 = perfect. Ours is about 0.52.",
}

RESULT_FILES = {
    "model_comparison_val": "model_comparison_val.csv",
    "model_comparison_test": "test_model_comparison.csv",
    "backtest": "backtest_test.csv",
    "backtest_vs_random": "backtest_vs_random.csv",
    "fee_sensitivity": "fee_sensitivity_test.csv",
    "lstm": "lstm_comparison.csv",
    "weighting": "weighting_val.csv",
    "risk_of_ruin": "risk_of_ruin.csv",
    "forward_auc": "forward_test_auc.csv",
    "forward_backtest": "forward_test_backtest.csv",
}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) or np.isinf(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


@functools.lru_cache(maxsize=None)
def get_bundle(symbol, interval):
    path = os.path.join("models", f"{symbol}_{interval}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run: python save_model.py")
    return joblib.load(path)


@functools.lru_cache(maxsize=None)
def _get_data_replay(symbol, interval):
    b = get_bundle(symbol, interval)
    df = load_split(symbol, interval, "test").reset_index(drop=True)
    r = df["return"].to_numpy()
    return dict(
        source="replay",
        df=df,
        r=r,
        vol=bt.recent_vol(r),
        probs=b["model"].predict_proba(df[b["features"]])[:, 1],
        price=100 * np.cumprod(1 + r),     # price INDEX starting at 100 (not a real price)
        times=None,
    )


_db_cache = {}


def _get_data_db(symbol, interval, ttl=20):
    """Newest real candles from PostgreSQL (cached for `ttl` seconds so the dashboard can refresh often)."""
    key = (symbol, interval)
    hit = _db_cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    import db_data
    b = get_bundle(symbol, interval)
    df = db_data.compute_features(db_data.load_candles(symbol, interval))
    df = df.dropna(subset=b["features"]).reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} usable candles for {symbol} {interval} - need at least 60.")
    r = df["return"].to_numpy()
    times = [int(t.timestamp()) for t in df["open_time"]]
    data = dict(
        source="db",
        df=df,
        r=r,
        vol=bt.recent_vol(r),
        probs=b["model"].predict_proba(df[b["features"]])[:, 1],
        price=df["close"].to_numpy(),
        times=times,
    )
    _db_cache[key] = (time.time(), data)
    return data


def get_data(symbol, interval, source="db"):
    if source == "db":
        return _get_data_db(symbol, interval)
    if source == "replay":
        return _get_data_replay(symbol, interval)
    raise ValueError("source must be 'db' or 'replay'")


def status():
    """Data freshness per coin/timeframe from the database."""
    import db_data
    return db_data.status()


@functools.lru_cache(maxsize=None)
def get_explainer(symbol, interval):
    import shap
    return shap.TreeExplainer(get_bundle(symbol, interval)["model"])


def _resolve(symbol, interval, index, source="db"):
    d = get_data(symbol, interval, source)
    n = len(d["df"])
    i = index if index >= 0 else n + index
    if not 0 <= i < n:
        raise ValueError(f"index must be between {-n} and {n - 1}")
    return d, i, n


def _shap_row(symbol, interval, row_df):
    sv = get_explainer(symbol, interval).shap_values(row_df)
    if isinstance(sv, list):
        sv = sv[-1]
    return np.asarray(sv)[0]


def signal(symbol, interval, index=-1, source="db"):
    b = get_bundle(symbol, interval)
    d, i, n = _resolve(symbol, interval, index, source)
    p = float(d["probs"][i])
    vol = float(d["vol"][i]) if not np.isnan(d["vol"][i]) else None
    lean = "up" if p >= 0.5 else "down"

    pct = int(min(100, np.searchsorted(b["val_prob_quantiles"], p)))
    rank = pct if lean == "up" else 100 - pct          # how extreme vs the model's usual predictions (50-100)
    rank = max(rank, 50)
    conf = "strong (for this model)" if rank >= 90 else "moderate" if rank >= 75 else "weak"

    reasons, red, amber = [], False, False
    if lean == "down":
        red = True
        reasons.append(dict(level="red", text="The model leans DOWN. This tool only looks at buying, so the safe call is to stay out."))
    else:
        reasons.append(dict(level="ok", text="The model leans slightly up."))

    if vol is None:
        amber = True
        reasons.append(dict(level="amber", text="Not enough recent data to judge volatility yet."))
        fee_ratio = None
    else:
        typical_move = vol * np.sqrt(HORIZON)
        fee_ratio = typical_move / (2 * FEE)
        if fee_ratio < 1:
            red = True
            reasons.append(dict(level="red", text=f"Fees would eat the move: a typical move over the next {HORIZON} candles is about {typical_move:.2%}, but a round trip costs {2 * FEE:.2%}."))
        elif fee_ratio < 2:
            amber = True
            reasons.append(dict(level="amber", text=f"Tight after fees: a typical move is about {typical_move:.2%} against {2 * FEE:.2%} round-trip fees."))
        else:
            reasons.append(dict(level="ok", text=f"A typical move ({typical_move:.2%}) is comfortably bigger than round-trip fees ({2 * FEE:.2%})."))
        if vol <= b["train_vol_quantiles"][30]:
            amber = True
            reasons.append(dict(level="amber", text="The market is unusually quiet right now. Quiet markets have smaller moves, so fees matter even more."))

    if rank < 75:
        amber = True
        reasons.append(dict(level="amber", text="The model is not very confident compared with its usual predictions."))
    else:
        reasons.append(dict(level="ok", text=f"Confidence is {conf} compared with the model's usual predictions."))

    candle_time, age_hours = None, None
    if d["times"] is not None:
        candle_time = pd.Timestamp(d["times"][i], unit="s", tz="UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - candle_time).total_seconds() / 3600
        if i == len(d["times"]) - 1 and age_hours > 1:
            amber = True
            reasons.append(dict(level="amber", text=f"This is the newest candle in the database, but it is {age_hours:.0f} hours old. The data feed is not current, so do not treat this as a live signal."))

    color ="red" if red else "amber" if amber else "green"
    headline = {
        "red": "Skip this trade.",
        "amber": "Not convincing. Be careful.",
        "green": "No red flags found, but the signal is still weak.",
    }[color]

    row_df = d["df"].iloc[[i]][b["features"]]
    sv = _shap_row(symbol, interval, row_df)
    row = row_df.iloc[0]
    order = np.argsort(-np.abs(sv))[:5]
    drivers = [dict(feature=b["features"][j], value=_clean(row.iloc[j]), shap=_clean(sv[j]),
                    pushes="up" if sv[j] > 0 else "down") for j in order]

    explanation = None
    try:
        from explanations import explain_prediction
    except Exception:
        explain_prediction = None
    if explain_prediction is not None:
        shap_named = pd.Series(sv, index=b["features"])
        typ = b["typical"]
        # explanations.py may index SHAP values / typical values by position OR by name: try both styles
        attempts = [(row, sv, typ), (row, sv, typ.to_numpy()), (row.to_numpy(), sv, typ.to_numpy()),
                    (row, shap_named, typ)]
        for r_, s_, t_ in attempts:
            try:
                res = explain_prediction(r_, s_, p, t_, top_n=3)
            except Exception:
                continue
            if res is not None and str(res).strip():
                explanation = res if isinstance(res, str) else "\n".join(map(str, res))
                break
    if not explanation:
        top = drivers[:3]
        explanation = "Main drivers: " + "; ".join(f"{x['feature']} pushes {x['pushes']}" for x in top) + "."

    return dict(symbol=symbol, interval=interval, source=d["source"], index=i, n_steps=n,
                candle_time=str(candle_time) if candle_time is not None else None,
                hours_old=_clean(age_hours), price=_clean(d["price"][i]), price_is_index=d["source"] == "replay",
                probability_up=round(p, 4), lean=lean, confidence=conf, confidence_rank=rank,
                verdict=color, headline=headline, reasons=reasons,
                fee_ratio=_clean(fee_ratio), recent_volatility=_clean(vol),
                drivers=drivers, explanation=explanation,
                disclaimer="Short-term direction is barely predictable (AUC about 0.52). This is decision support, not advice.")


def series(symbol, interval, end=-1, n=200, source="db"):
    d, i, total = _resolve(symbol, interval, end, source)
    lo = max(0, i - n + 1)
    df = d["df"].iloc[lo:i + 1]
    out = dict(symbol=symbol, interval=interval, source=d["source"], start=lo, end=i,
               price_is_index=d["source"] == "replay")
    out["price_index"] = [_clean(x) for x in d["price"][lo:i + 1]]
    if d["times"] is not None:      # real candles for a candlestick chart
        out["candles"] = [dict(time=d["times"][k], open=_clean(df["open"].iloc[k - lo]),
                               high=_clean(df["high"].iloc[k - lo]), low=_clean(df["low"].iloc[k - lo]),
                               close=_clean(df["close"].iloc[k - lo])) for k in range(lo, i + 1)]
    for col in ("rsi", "macd_hist_pct", "bb_position", "relative_volume"):
        out[col] = [_clean(x) for x in df[col].to_numpy()]
    out["probability_up"] = [_clean(x) for x in d["probs"][lo:i + 1]]
    return out


def risk_plan(symbol, interval, index, balance, risk_pct, entry=None, source="db"):
    d, i, _ = _resolve(symbol, interval, index, source)
    vol = d["vol"][i]
    if np.isnan(vol):
        raise ValueError("Not enough recent data at this point to size a trade.")
    if balance <= 0 or not 0 < risk_pct <= 1:
        raise ValueError("balance must be positive and risk_pct between 0 and 1 (e.g. 0.01 for 1%).")
    entry = float(entry) if entry else float(d["price"][i])
    v = vol * np.sqrt(HORIZON)
    sf, tf = max(STOP_MULT * v, 1e-4), TARGET_MULT * v
    stop, target = entry * (1 - sf), entry * (1 + tf)
    notional = min(balance * risk_pct / sf, balance / (1 + FEE))
    fees = notional * 2 * FEE
    max_loss = notional * sf + fees
    gain = notional * tf - fees
    rr = gain / max_loss if max_loss > 0 else None

    warnings = []
    if risk_pct > 0.02:
        warnings.append("You are risking more than 2% of your account on one trade. Our simulation shows risking 5-10% per trade makes ruin very likely.")
    if notional >= balance / (1 + FEE) - 1e-9:
        warnings.append("The position is capped at your whole balance (no leverage), so you are risking less than you asked for.")
    if tf < 2 * 2 * FEE:
        warnings.append("The target move is only a few times the round-trip fee, so fees take a big share of any gain.")
    if rr is not None and rr < 1.2:
        warnings.append("Reward is not much bigger than risk after fees.")
    return dict(entry=round(entry, 4), stop=round(stop, 4), target=round(target, 4),
                stop_distance_pct=round(sf * 100, 3), target_distance_pct=round(tf * 100, 3),
                position_value=round(notional, 2), units=round(notional / entry, 6),
                round_trip_fees=round(fees, 2), max_loss=round(max_loss, 2),
                potential_gain=round(gain, 2), risk_reward_after_fees=_clean(rr),
                warnings=warnings,
                note="Stop and target scale with recent volatility (stop 2x, target 3x, over 15 candles).")


def results(name):
    if name not in RESULT_FILES:
        raise KeyError(f"unknown result '{name}'. Options: {', '.join(RESULT_FILES)}")
    path = os.path.join("results", RESULT_FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the script that creates it first")
    df = pd.read_csv(path)
    return [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]
