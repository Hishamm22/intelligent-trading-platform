import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from model_utils import load_split

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
WINDOW = 20

rows = []
for symbol in SYMBOLS:
    for interval in INTERVALS:
        val = load_split(symbol, interval, "val").copy()

        val["next_abs_return"] = (val["close"].shift(-1) / val["close"] - 1).abs()
        val["recent_vol"] = val["return"].rolling(WINDOW).std()
        val = val.dropna(subset=["next_abs_return", "recent_vol"])

        high_next = (val["next_abs_return"] > val["next_abs_return"].median()).astype(int)

        rho, _ = spearmanr(val["recent_vol"], val["next_abs_return"])
        auc = roc_auc_score(high_next, val["recent_vol"])
        rows.append({"symbol": symbol, "interval": interval,
                     "spearman": round(rho, 3), "auc_high_vol": round(auc, 4), "n": len(val)})

print(pd.DataFrame(rows).to_string(index=False))