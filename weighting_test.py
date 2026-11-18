"""
weighting_test.py - does 'balancing' the training data (sample weights) help?

Your supervisor's idea: the market is often calm, and calm candles are mostly noise
(and many are flat, which our target labels as 0). We test 4 ways of weighting
the TRAINING rows and compare on the VALIDATION set:

  none            every row counts the same (what we did so far)
  class_balanced  'up' and 'down' rows carry equal total weight
  calm_balanced   calm-market rows and active-market rows carry equal total weight
  move_weighted   rows followed by a bigger price move count more (tiny moves = noise)

Run:  python weighting_test.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def make_weights(df, scheme):
    from backtest import recent_vol
    n = len(df)
    if scheme == "none":
        return np.ones(n)
    y = df["target"].to_numpy()
    if scheme == "class_balanced":
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
        return w
    if scheme == "calm_balanced":
        v = recent_vol(df["return"].to_numpy())
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        calm = v <= np.median(v)
        w = np.where(calm, 0.5 / max(calm.mean(), 1e-9), 0.5 / max(1 - calm.mean(), 1e-9))
        return w
    if scheme == "move_weighted":
        nxt = np.abs(df["return"].shift(-1).to_numpy())
        med = np.nanmedian(nxt[nxt > 0]) if (nxt > 0).any() else 1.0
        w = np.clip(np.nan_to_num(nxt, nan=med) / med, 0.1, 3.0)
        return w
    raise ValueError(scheme)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, roc_auc_score
    from xgboost import XGBClassifier
    from model_utils import FEATURE_COLUMNS, load_split

    SCHEMES = ["none", "class_balanced", "calm_balanced", "move_weighted"]
    rows = []
    for sym in ["BTCUSDT", "XRPUSDT", "SOLUSDT"]:
        tr, va = load_split(sym, "5m", "train"), load_split(sym, "5m", "val")
        base = max(va["target"].mean(), 1 - va["target"].mean())
        for sc in SCHEMES:
            m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
                              n_jobs=-1, random_state=42)
            m.fit(tr[FEATURE_COLUMNS], tr["target"], sample_weight=make_weights(tr, sc))
            p = m.predict_proba(va[FEATURE_COLUMNS])[:, 1]
            rows.append(dict(symbol=sym, scheme=sc, baseline_accuracy=base,
                             accuracy=accuracy_score(va["target"], p > 0.5),
                             roc_auc=roc_auc_score(va["target"], p)))
            print(f"{sym} {sc:15s} AUC {rows[-1]['roc_auc']:.4f}")
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/weighting_val.csv", index=False)
    print("\nMean over the 3 coins (validation, 5m):")
    print(out.groupby("scheme")[["accuracy", "roc_auc"]].mean().round(4).to_string())
    print("\nRule: only call a scheme better if it beats 'none' on average AND on most coins.")
    print("Saved results/weighting_val.csv")
