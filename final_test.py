import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from model_utils import (FEATURE_COLUMNS, TARGET_COLUMN, load_split,
                         evaluate, bootstrap_auc_ci)

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
CUT_QUANTILE = 0.80

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


def acc_and_margin(correct, mask):
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan, 0
    p = correct[mask].mean()
    return p, 1.96 * np.sqrt(p * (1 - p) / n), n


main_rows, conf_rows, brier_rows, cal_bins = [], [], [], []

for symbol in SYMBOLS:
    for interval in INTERVALS:
        train = load_split(symbol, interval, "train")
        val = load_split(symbol, interval, "val")
        test = load_split(symbol, interval, "test")

        X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
        X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN].to_numpy()
        flat_test = test["flat_next"].to_numpy(dtype=bool)
        moving = ~flat_test

        majority = int(y_train.mean() >= 0.5)
        baseline_acc = (y_test == majority).mean()
        baseline_moving = (y_test[moving] == majority).mean()

        for name, model in make_models().items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)

            result = evaluate(y_test, pred, prob, flat_test)
            low, high = bootstrap_auc_ci(y_test, prob)
            result.update({"symbol": symbol, "interval": interval, "model": name,
                           "baseline_accuracy": baseline_acc,
                           "baseline_up_vs_down": baseline_moving,
                           "auc_low": low, "auc_high": high})
            main_rows.append(result)
            print(f"{symbol} {interval} {name}: AUC {result['roc_auc']:.4f}")

            if name != "XGBoost":
                continue

            # Confidence cut-off: fixed on VALIDATION, applied unchanged to TEST
            val_prob = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
            cut = np.quantile(np.abs(val_prob - 0.5), CUT_QUANTILE)
            is_high = np.abs(prob - 0.5) >= cut
            correct = (pred == y_test)
            acc_h, mar_h, n_h = acc_and_margin(correct, is_high & moving)
            acc_r, mar_r, n_r = acc_and_margin(correct, ~is_high & moving)
            conf_rows.append({"symbol": symbol, "interval": interval, "cutoff": cut,
                              "n_high": n_h, "acc_high": acc_h, "margin_high": mar_h,
                              "n_rest": n_r, "acc_rest": acc_r, "margin_rest": mar_r})

            # Calibration: do predicted probabilities match what actually happened?
            brier_model = brier_score_loss(y_test, prob)
            brier_const = brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))
            bin_id = pd.qcut(prob, 5, labels=False, duplicates="drop")
            cal = (pd.DataFrame({"prob": prob, "y": y_test, "bin": bin_id})
                   .groupby("bin")
                   .agg(mean_pred=("prob", "mean"), actual=("y", "mean"), n=("y", "size")))
            ece = (cal["n"] / cal["n"].sum() * (cal["mean_pred"] - cal["actual"]).abs()).sum()
            brier_rows.append({"symbol": symbol, "interval": interval,
                               "brier_model": brier_model, "brier_constant": brier_const,
                               "calibration_gap": ece})
            cal["symbol"], cal["interval"] = symbol, interval
            cal_bins.append(cal.reset_index())

main_cols = ["symbol", "interval", "model", "baseline_accuracy", "accuracy",
             "baseline_up_vs_down", "accuracy_up_vs_down", "precision", "recall",
             "f1", "roc_auc", "auc_low", "auc_high"]
main = pd.DataFrame(main_rows)[main_cols].round(4)
conf = pd.DataFrame(conf_rows).round(4)
brier = pd.DataFrame(brier_rows).round(4)

print("\n=== TEST SET: all models ===")
print(main.to_string(index=False))
print("\n=== TEST SET: XGBoost, confidence cut-off fixed on validation (top 20%) ===")
print(conf.to_string(index=False))
print("\n=== TEST SET: XGBoost calibration ===")
print(brier.to_string(index=False))

main.to_csv("results/test_model_comparison.csv", index=False)
conf.to_csv("results/test_confidence_cutoff.csv", index=False)
brier.to_csv("results/test_calibration_summary.csv", index=False)
pd.concat(cal_bins).round(4).to_csv("results/test_calibration_bins.csv", index=False)