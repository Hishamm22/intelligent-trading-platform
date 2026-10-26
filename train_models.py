import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from model_utils import (FEATURE_COLUMNS, TARGET_COLUMN, load_split,
                         evaluate, bootstrap_auc_ci)

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]

os.makedirs("results", exist_ok=True)


def make_models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=100, max_features="sqrt",
            n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
            n_jobs=-1, random_state=42),
    }


rows = []

for symbol in SYMBOLS:
    for interval in INTERVALS:
        train = load_split(symbol, interval, "train")
        val = load_split(symbol, interval, "val")

        X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN].to_numpy()
        flat_val = val["flat_next"].to_numpy(dtype=bool)

        majority = int(y_train.mean() >= 0.5)
        baseline_acc = (y_val == majority).mean()
        baseline_moving = (y_val[~flat_val] == majority).mean()

        for name, model in make_models().items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_val)[:, 1]
            pred = (prob >= 0.5).astype(int)

            result = evaluate(y_val, pred, prob, flat_val)
            low, high = bootstrap_auc_ci(y_val, prob)
            result.update({
                "symbol": symbol, "interval": interval, "model": name,
                "baseline_accuracy": baseline_acc,
                "baseline_up_vs_down": baseline_moving,
                "auc_low": low, "auc_high": high,
            })
            rows.append(result)
            print(f"{symbol} {interval} {name}: AUC {result['roc_auc']:.4f} "
                  f"(95% range {low:.4f} to {high:.4f})")

results = pd.DataFrame(rows)
cols = ["symbol", "interval", "model", "baseline_accuracy", "accuracy",
        "baseline_up_vs_down", "accuracy_up_vs_down", "precision", "recall",
        "f1", "roc_auc", "auc_low", "auc_high"]
results = results[cols].round(4)
print()
print(results.to_string(index=False))
results.to_csv("results/model_comparison_val.csv", index=False)