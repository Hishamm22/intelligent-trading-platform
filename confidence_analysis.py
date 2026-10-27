import pandas as pd
from xgboost import XGBClassifier
from model_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_split

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
rows = []

for symbol in SYMBOLS:
    for interval in INTERVALS:
        train = load_split(symbol, interval, "train")
        val = load_split(symbol, interval, "val").copy()

        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
            n_jobs=-1, random_state=42)
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        val["prob"] = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
        val["pred"] = (val["prob"] >= 0.5).astype(int)
        val["confidence"] = (val["prob"] - 0.5).abs()

        moving = val[~val["flat_next"]].copy()
        moving["bucket"] = pd.qcut(moving["confidence"], 5, labels=[1, 2, 3, 4, 5])
        moving["correct"] = (moving["pred"] == moving[TARGET_COLUMN]).astype(int)

        summary = moving.groupby("bucket", observed=True)["correct"].agg(["mean", "count"])
        for bucket, r in summary.iterrows():
            rows.append({"symbol": symbol, "interval": interval, "bucket": int(bucket),
                         "accuracy": round(r["mean"], 4), "n": int(r["count"])})

out = pd.DataFrame(rows)
table = out.pivot_table(index=["symbol", "interval"], columns="bucket", values="accuracy")
print(table.to_string())
out.to_csv("results/confidence_buckets_val.csv", index=False)