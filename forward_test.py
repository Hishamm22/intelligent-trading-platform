"""
forward_test.py - do the saved models still work on data they have NEVER seen?

WHY: the models were trained on the TRAIN period only, and the TEST period was set aside, but we
did look at the test results while choosing what to report. The candles that arrived AFTER the
test period (kept in PostgreSQL by live_feed.py) were never used for any decision. Running the saved
models on them, with NO retraining, is the cleanest out-of-sample check we have.

WHAT IT DOES (nothing is retrained and no existing result file is changed):
  1. For each coin/timeframe it reads where your TEST period ended (datasets/<coin>_<tf>_test.csv).
  2. It loads every candle AFTER that moment from PostgreSQL and builds the same 12 features.
  3. It scores them with the saved model in models/ and reports AUC with a 95% range
     (same block-bootstrap as the rest of the project).
  4. For 5m it repeats the backtest from run_backtests.py: the strategy settings are chosen on the
     VALIDATION period exactly as before, then applied ONCE to the forward period, and compared
     with buy-and-hold and 200 random-entry traders. It also shows the no-fee result.

Run:   python forward_test.py
Saves: results/forward_test_auc.csv and results/forward_test_backtest.csv
Needs: your datasets/ folder, models/ folder and PostgreSQL running. Do NOT re-run build_datasets.py first.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import backtest as bt
import db_data
from model_utils import FEATURE_COLUMNS, bootstrap_auc_ci, load_split

warnings.filterwarnings("ignore")
SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
BT_INTERVAL = "5m"
CONF_QUANTILES = [0.50, 0.70, 0.85, 0.95]     # same grid as run_backtests.py
QUIET_QUANTILES = [0.0, 0.3, 0.5]
MODES = ["fixed", "trailing"]
MIN_ROWS = 500
os.makedirs("results", exist_ok=True)


def forward_frame(symbol, interval, loader):
    """Features for every candle AFTER the end of the TEST period (with their true next-candle label)."""
    test = load_split(symbol, interval, "test")
    cutoff = pd.to_datetime(test["open_time"].max(), utc=True)
    raw = loader(symbol, interval, 10 ** 9)                       # ALL candles, so indicators have a warm-up
    df = db_data.compute_features(raw)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df["flat_next"] = df["close"].shift(-1) == df["close"]
    df = df.iloc[:-1]                                             # newest candle has no 'next' close yet
    df = df[df["open_time"] > cutoff].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df, cutoff


def choose_config(symbol, bundle):
    """Pick the strategy settings on VALIDATION only - identical procedure to run_backtests.py."""
    tr, va = load_split(symbol, BT_INTERVAL, "train"), load_split(symbol, BT_INTERVAL, "val")
    p_va = bundle["model"].predict_proba(va[bundle["features"]])[:, 1]
    r_va = va["return"].to_numpy()
    v_va = bt.recent_vol(r_va)
    tvol = bt.recent_vol(tr["return"].to_numpy())
    best, cfg = None, None
    for cq in CONF_QUANTILES:
        thr = float(np.quantile(p_va, cq))
        for qq in QUIET_QUANTILES:
            mv = float(np.nanquantile(tvol, qq))
            for mode in MODES:
                res = bt.run_backtest(r_va, p_va, v_va, BT_INTERVAL, threshold=thr, min_vol=mv, mode=mode)
                score = res["sharpe"] if res["n_trades"] >= 30 and not np.isnan(res["sharpe"]) else -1e9
                if best is None or score > best:
                    best, cfg = score, dict(threshold=thr, min_vol=mv, mode=mode)
    return cfg


def main(loader=None):
    loader = loader or db_data.load_candles
    try:
        orig = pd.read_csv("results/test_model_comparison.csv")
        orig = orig[orig["model"] == "XGBoost"].set_index(["symbol", "interval"])["roc_auc"]
    except Exception:
        orig = pd.Series(dtype=float)

    auc_rows, bt_rows, frames = [], [], {}
    for sym in SYMBOLS:
        for itv in INTERVALS:
            bundle = joblib.load(os.path.join("models", f"{sym}_{itv}.joblib"))
            df, cutoff = forward_frame(sym, itv, loader)
            if len(df) < MIN_ROWS:
                print(f"{sym} {itv}: only {len(df)} forward candles - skipped")
                continue
            prob = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
            y = df["target"].to_numpy()
            auc = roc_auc_score(y, prob)
            lo, hi = bootstrap_auc_ci(y, prob)
            auc_rows.append(dict(symbol=sym, interval=itv, forward_start=str(df["open_time"].iloc[0]),
                                 forward_end=str(df["open_time"].iloc[-1]), n_candles=len(df),
                                 days=round((df["open_time"].iloc[-1] - df["open_time"].iloc[0]).total_seconds() / 86400, 1),
                                 up_rate=round(float(y.mean()), 4), forward_auc=round(auc, 4),
                                 auc_low=round(lo, 4), auc_high=round(hi, 4),
                                 original_test_auc=round(float(orig.get((sym, itv), np.nan)), 4),
                                 ci_excludes_0_5=bool(lo > 0.5 or hi < 0.5)))
            print(f"{sym} {itv}: {len(df):,} candles, AUC {auc:.4f} ({lo:.4f}-{hi:.4f})")
            if itv == BT_INTERVAL:
                frames[sym] = (bundle, df, prob)

    aucs = pd.DataFrame(auc_rows)
    aucs.to_csv("results/forward_test_auc.csv", index=False)
    print("\n=== FORWARD TEST: AUC on data the models never saw ===")
    print(aucs.to_string(index=False))

    for sym, (bundle, df, p) in frames.items():
        cfg = choose_config(sym, bundle)
        r = df["return"].to_numpy()
        v = bt.recent_vol(r)
        naive = dict(threshold=0.5, min_vol=0.0, mode="fixed")
        for label, c, fee in (("naive (trade every up-signal, fixed stop)", naive, bt.FEE),
                              ("chosen on validation", cfg, bt.FEE),
                              ("chosen on validation, NO fees", cfg, 0.0)):
            res = bt.run_backtest(r, p, v, BT_INTERVAL, fee=fee, **c)
            bt_rows.append(dict(symbol=sym, strategy=label, **res))
        bh = bt.buy_and_hold(r, BT_INTERVAL)
        bt_rows.append(dict(symbol=sym, strategy="buy and hold", **bh))
        n_sig = int(((p >= cfg["threshold"]) & (v >= cfg["min_vol"]) & ~np.isnan(v)).sum())
        rnd = bt.random_baseline(r, v, n_sig, interval=BT_INTERVAL, mode=cfg["mode"])
        rr = np.array([x["total_return"] for x in rnd])
        chosen = [x for x in bt_rows if x["symbol"] == sym and x["strategy"] == "chosen on validation"][0]
        bt_rows.append(dict(symbol=sym, strategy="random entries (median of 200)", total_return=float(np.median(rr)),
                            n_trades=n_sig))
        print(f"{sym}: chosen {chosen['total_return']:+.2%} | buy&hold {bh['total_return']:+.2%} | "
              f"random median {np.median(rr):+.2%} | model beats {(rr < chosen['total_return']).mean():.0%} of random runs")

    out = pd.DataFrame(bt_rows)
    cols = ["symbol", "strategy", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate", "profit_factor"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv("results/forward_test_backtest.csv", index=False)
    print("\n=== FORWARD TEST: 5m backtest (settings chosen on validation, applied once) ===")
    print(out.round(4).to_string(index=False))
    print("\nSaved results/forward_test_auc.csv and results/forward_test_backtest.csv")
    print("Report it honestly whatever it shows: AUC near 0.5 or a negative return are valid findings.")


if __name__ == "__main__":
    main()
