"""
run_fee_sensitivity.py - how much of the loss is just trading fees?

Same honest procedure as run_backtests.py (settings picked on VALIDATION, evaluated once on TEST),
repeated at different fee levels per side:
  0.10%   standard Binance spot fee (what we assumed so far)
  0.075%  with the BNB fee discount
  0.02%   a low maker-style fee
  0.00%   NO fees = the model's raw (gross) edge

Run:  python run_fee_sensitivity.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import backtest as bt
from model_utils import FEATURE_COLUMNS, load_split

warnings.filterwarnings("ignore")
SYMBOLS, INTERVAL = ["BTCUSDT", "XRPUSDT", "SOLUSDT"], "5m"
FEES = [0.001, 0.00075, 0.0002, 0.0]
CONF_Q, QUIET_Q, MODES = [0.50, 0.70, 0.85, 0.95], [0.0, 0.3, 0.5], ["fixed", "trailing"]
os.makedirs("results", exist_ok=True)

rows = []
for sym in SYMBOLS:
    tr, va, te = (load_split(sym, INTERVAL, s) for s in ("train", "val", "test"))
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
                      n_jobs=-1, random_state=42).fit(tr[FEATURE_COLUMNS], tr["target"])
    r_va, r_te = va["return"].to_numpy(), te["return"].to_numpy()
    p_va, p_te = (m.predict_proba(d[FEATURE_COLUMNS])[:, 1] for d in (va, te))
    v_va, v_te = bt.recent_vol(r_va), bt.recent_vol(r_te)
    tvol = bt.recent_vol(tr["return"].to_numpy())
    for fee in FEES:
        best, cfg = None, None
        for cq in CONF_Q:
            thr = float(np.quantile(p_va, cq))
            for qq in QUIET_Q:
                mv = float(np.nanquantile(tvol, qq))
                for mode in MODES:
                    res = bt.run_backtest(r_va, p_va, v_va, INTERVAL, threshold=thr, min_vol=mv, mode=mode, fee=fee)
                    sc = res["sharpe"] if res["n_trades"] >= 30 and not np.isnan(res["sharpe"]) else -1e9
                    if best is None or sc > best:
                        best, cfg = sc, dict(threshold=thr, min_vol=mv, mode=mode)
        res = bt.run_backtest(r_te, p_te, v_te, INTERVAL, fee=fee, **cfg)
        bh = bt.buy_and_hold(r_te, INTERVAL, fee=fee)
        rows.append(dict(symbol=sym, fee_per_side=fee, mode=cfg["mode"], n_trades=res["n_trades"],
                         win_rate=res["win_rate"], model_return=res["total_return"], model_sharpe=res["sharpe"],
                         buyhold_return=bh["total_return"], buyhold_sharpe=bh["sharpe"]))

out = pd.DataFrame(rows)
out.to_csv("results/fee_sensitivity_test.csv", index=False)
print(out.round(4).to_string(index=False))
print("\nModel total return on TEST by fee level:")
print(out.pivot(index="symbol", columns="fee_per_side", values="model_return").round(4).to_string())
print("\nIf the 0.00 column is still about zero or negative, the model has no gross edge - fees are not the only problem.")
print("Saved results/fee_sensitivity_test.csv")
