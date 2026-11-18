"""
run_backtests.py - Week 10: backtest the model's signals.

Honest procedure (so the result is defensible):
  1. Train XGBoost on the TRAIN period only.
  2. Try 24 strategy settings on the VALIDATION period and pick the best by Sharpe ratio.
  3. Run ONLY that chosen setting (and the 'naive' one for comparison) on the TEST period, once.
  4. Compare with buy-and-hold and with 200 random-entry traders using the same exit rules.

Run:  python run_backtests.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import backtest as bt
from model_utils import FEATURE_COLUMNS, load_split

warnings.filterwarnings("ignore")

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVAL = "5m"
SAMPLE_WEIGHTS = None   # set to "calm" after weighting_test.py if it helped (see that script)
CONF_QUANTILES = [0.50, 0.70, 0.85, 0.95]   # act only on the most confident X% of val predictions
QUIET_QUANTILES = [0.0, 0.3, 0.5]           # skip the quietest X% of markets
MODES = ["fixed", "trailing"]
os.makedirs("results", exist_ok=True)


def train_model(tr):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
                      n_jobs=-1, random_state=42)
    w = None
    if SAMPLE_WEIGHTS == "calm":
        from weighting_test import make_weights
        w = make_weights(tr, "calm_balanced")
    m.fit(tr[FEATURE_COLUMNS], tr["target"], sample_weight=w)
    return m


def prepare(df, model):
    r = df["return"].to_numpy()
    return r, model.predict_proba(df[FEATURE_COLUMNS])[:, 1], bt.recent_vol(r)


rows_val, rows_test, rows_cmp = [], [], []
for sym in SYMBOLS:
    print(f"\n=== {sym} {INTERVAL} ===")
    tr, va, te = (load_split(sym, INTERVAL, s) for s in ("train", "val", "test"))
    model = train_model(tr)
    r_va, p_va, v_va = prepare(va, model)
    r_te, p_te, v_te = prepare(te, model)
    train_vol = bt.recent_vol(tr["return"].to_numpy())

    # ---- 1) choose settings on VALIDATION only
    best, best_cfg = None, None
    for cq in CONF_QUANTILES:
        thr = float(np.quantile(p_va, cq))
        for qq in QUIET_QUANTILES:
            mv = float(np.nanquantile(train_vol, qq))
            for mode in MODES:
                res = bt.run_backtest(r_va, p_va, v_va, INTERVAL, threshold=thr, min_vol=mv, mode=mode)
                rows_val.append(dict(symbol=sym, conf_q=cq, quiet_q=qq, mode=mode, threshold=thr, **res))
                score = res["sharpe"] if res["n_trades"] >= 30 and not np.isnan(res["sharpe"]) else -1e9
                if best is None or score > best:
                    best, best_cfg = score, dict(conf_q=cq, quiet_q=qq, mode=mode, threshold=thr, min_vol=mv)
    print("chosen on validation:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_cfg.items()})

    # ---- 2) evaluate ONCE on TEST: chosen config vs naive config vs buy&hold vs random
    naive_cfg = dict(threshold=0.5, min_vol=0.0, mode="fixed")
    chosen_cfg = dict(threshold=best_cfg["threshold"], min_vol=best_cfg["min_vol"], mode=best_cfg["mode"])
    out = {}
    for label, cfg in (("naive (trade every up-signal, fixed stop)", naive_cfg),
                       ("chosen on validation", chosen_cfg)):
        res = bt.run_backtest(r_te, p_te, v_te, INTERVAL, **cfg)
        out[label] = res
        rows_test.append(dict(symbol=sym, strategy=label, **res))
    bh = bt.buy_and_hold(r_te, INTERVAL)
    rows_test.append(dict(symbol=sym, strategy="buy and hold", **bh))

    n_sig = int(((p_te >= chosen_cfg["threshold"]) & (v_te >= chosen_cfg["min_vol"]) & ~np.isnan(v_te)).sum())
    rnd = bt.random_baseline(r_te, v_te, n_sig, **{"interval": INTERVAL, "mode": chosen_cfg["mode"]})
    rr = np.array([x["total_return"] for x in rnd])
    rs = np.array([x["sharpe"] for x in rnd])
    chosen = out["chosen on validation"]
    pct = float((rr < chosen["total_return"]).mean())
    rows_cmp.append(dict(symbol=sym, model_return=chosen["total_return"], buyhold_return=bh["total_return"],
                         random_median_return=float(np.median(rr)), random_5pct=float(np.percentile(rr, 5)),
                         random_95pct=float(np.percentile(rr, 95)),
                         model_beats_pct_of_random=pct, model_sharpe=chosen["sharpe"],
                         random_median_sharpe=float(np.nanmedian(rs))))

    show = pd.DataFrame([dict(strategy=k, **v) for k, v in out.items()] + [dict(strategy="buy and hold", **bh)])
    cols = ["strategy", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate",
            "avg_loss", "worst_trade", "profit_factor"]
    print(show[cols].round(4).to_string(index=False))
    print(f"random-entry traders (200 runs, same exits): median return {np.median(rr):.4f}, "
          f"5-95% range {np.percentile(rr, 5):.4f} to {np.percentile(rr, 95):.4f}; "
          f"model beats {pct:.0%} of them")
    flag = "YES" if chosen["sharpe"] >= 2.0 else "no"
    print(f"Sharpe >= 2.0 (supervisor's target)?  {flag}   (model {chosen['sharpe']:.2f}, buy&hold {bh['sharpe']:.2f})")

pd.DataFrame(rows_val).to_csv("results/backtest_val_grid.csv", index=False)
pd.DataFrame(rows_test).to_csv("results/backtest_test.csv", index=False)
pd.DataFrame(rows_cmp).to_csv("results/backtest_vs_random.csv", index=False)
print("\nSaved: results/backtest_val_grid.csv, backtest_test.csv, backtest_vs_random.csv")
