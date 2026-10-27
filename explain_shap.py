import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from model_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_split

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]

os.makedirs("results/shap", exist_ok=True)


def make_xgb():
    # Must match the settings used in train_models.py
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
        n_jobs=-1, random_state=42)


importance = {}

for symbol in SYMBOLS:
    for interval in INTERVALS:
        train = load_split(symbol, interval, "train")
        val = load_split(symbol, interval, "val")

        model = make_xgb()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(val[FEATURE_COLUMNS])

        importance[f"{symbol} {interval}"] = np.abs(shap_values).mean(axis=0)

        if symbol == "BTCUSDT" and interval == "1m":
            shap.summary_plot(shap_values, val[FEATURE_COLUMNS], show=False)
            plt.tight_layout()
            plt.savefig("results/shap/summary_BTCUSDT_1m.png", dpi=150)
            plt.close()

            prob = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
            i = int(np.argmax(prob))
            contrib = pd.Series(shap_values[i], index=FEATURE_COLUMNS)
            contrib = contrib.sort_values(key=np.abs, ascending=False)
            print(f"\nExample: BTCUSDT 1m candle at {val['open_time'].iloc[i]}, "
                  f"P(up) = {prob[i]:.3f}")
            for name, value in contrib.head(5).items():
                side = "toward UP" if value > 0 else "toward NOT-UP"
                print(f"  {name} = {val[name].iloc[i]:.4f} -> pushes {side} ({value:+.3f})")

imp = pd.DataFrame(importance, index=FEATURE_COLUMNS)
share = (imp / imp.sum() * 100).round(1)
share["average"] = share.mean(axis=1).round(1)
share = share.sort_values("average", ascending=False)
print("\nShare of total SHAP importance (%), validation sets:")
print(share.to_string())
share.to_csv("results/shap_importance_val.csv")