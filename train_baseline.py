import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from model_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_split, evaluate

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]

os.makedirs("results", exist_ok=True)
rows = []

for symbol in SYMBOLS:
    for interval in INTERVALS:
        train = load_split(symbol, interval, "train")
        val = load_split(symbol, interval, "val")

        X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN].to_numpy()
        flat_val = val["flat_next"].to_numpy(dtype=bool)

        # Baseline: always predict the majority class seen in TRAINING
        majority = int(y_train.mean() >= 0.5)
        baseline_acc = (y_val == majority).mean()
        baseline_acc_moving = (y_val[~flat_val] == majority).mean()

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        result = evaluate(y_val, pred, prob, flat_val)
        result.update({"symbol": symbol, "interval": interval, "model": "LogisticRegression",
                       "baseline_accuracy": baseline_acc,
                       "baseline_up_vs_down": baseline_acc_moving})
        rows.append(result)

results = pd.DataFrame(rows)
cols = ["symbol", "interval", "model", "baseline_accuracy", "accuracy", "baseline_up_vs_down",
        "accuracy_up_vs_down", "precision", "recall", "f1", "roc_auc"]
results = results[cols].round(4)
print(results.to_string(index=False))
results.to_csv("results/baseline_val_results.csv", index=False)