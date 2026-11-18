"""
save_model.py - train XGBoost ONCE per coin/timeframe and save it, so the API can load it instantly.

Run:  python save_model.py        (about a minute; creates a models/ folder with 9 files)

Each saved file holds: the trained model, the feature list, the 'typical' (median) value of every
feature, and reference numbers (how confident the model usually is, how volatile the market usually is)
so the dashboard can say things like "this is a weak signal" or "the market is quiet".
Trained on the TRAIN period only. Nothing from validation/test is used to fit it.
"""
import os
import warnings

import joblib
import numpy as np
from xgboost import XGBClassifier

import backtest as bt
from model_utils import FEATURE_COLUMNS, load_split

warnings.filterwarnings("ignore")
SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
os.makedirs("models", exist_ok=True)

for sym in SYMBOLS:
    for itv in INTERVALS:
        tr, va = load_split(sym, itv, "train"), load_split(sym, itv, "val")
        model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
                              n_jobs=-1, random_state=42)
        model.fit(tr[FEATURE_COLUMNS], tr["target"])
        p_val = model.predict_proba(va[FEATURE_COLUMNS])[:, 1]
        tvol = bt.recent_vol(tr["return"].to_numpy())
        tvol = tvol[~np.isnan(tvol)]
        bundle = dict(
            model=model,
            features=FEATURE_COLUMNS,
            typical=tr[FEATURE_COLUMNS].median(),
            val_prob_quantiles=np.quantile(p_val, np.linspace(0, 1, 101)),
            train_vol_quantiles=np.quantile(tvol, np.linspace(0, 1, 101)),
            train_rows=len(tr),
        )
        joblib.dump(bundle, f"models/{sym}_{itv}.joblib")
        print(f"saved models/{sym}_{itv}.joblib  (trained on {len(tr):,} rows)")

print("\nDone. Next: python -m uvicorn api:app --reload")
