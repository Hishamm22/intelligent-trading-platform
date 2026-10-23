import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from features import build_feature_set

load_dotenv()

SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

os.makedirs("datasets", exist_ok=True)

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="trading_platform",
    user="postgres",
    password=os.getenv("POSTGRES_PASSWORD")
)

def chronological_split(df, train_frac, val_frac):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

if __name__ == "__main__":
    query = """
        SELECT open_time, open, high, low, close, volume
        FROM candles
        WHERE symbol = %s AND interval = %s
        ORDER BY open_time
    """

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            df = pd.read_sql(query, conn, params=(symbol, interval))
            features = build_feature_set(df)
            features = features.dropna().reset_index(drop=True)

            train, val, test = chronological_split(features, TRAIN_FRAC, VAL_FRAC)

            for name, part in [("train", train), ("val", val), ("test", test)]:
                part.to_csv(f"datasets/{symbol}_{interval}_{name}.csv", index=False)

            print(f"{symbol} {interval}: {len(train)} train / {len(val)} val / {len(test)} test "
                  f"| up-rate train {train['target'].mean():.3f}, test {test['target'].mean():.3f}")

    conn.close()