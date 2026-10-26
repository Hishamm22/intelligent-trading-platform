import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

FEATURE_COLUMNS = [
    "return", "rsi",
    "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "price_vs_sma_short", "price_vs_sma_long", "ema_spread",
    "bb_width", "bb_position",
    "volume_change", "relative_volume",
]
TARGET_COLUMN = "target"


def load_split(symbol, interval, split):
    df = pd.read_csv(f"datasets/{symbol}_{interval}_{split}.csv", parse_dates=["open_time"])

    # Mark candles whose NEXT close is unchanged (must happen before any row is dropped)
    df["flat_next"] = df["close"].shift(-1) == df["close"]

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    dropped = before - len(df)
    if dropped:
        print(f"  {symbol} {interval} {split}: dropped {dropped} rows with infinite/missing features")

    return df.reset_index(drop=True)


def evaluate(y_true, y_pred, y_prob, flat_next):
    moving = ~flat_next
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy_up_vs_down": accuracy_score(y_true[moving], y_pred[moving]),
    }

def bootstrap_auc_ci(y_true, y_prob, n_boot=300, block=100, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample_true = y_true[idx]
        if sample_true.min() == sample_true.max():
            continue
        aucs.append(roc_auc_score(sample_true, y_prob[idx]))
    return np.percentile(aucs, [2.5, 97.5])