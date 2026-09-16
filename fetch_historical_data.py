import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Public market data endpoint - no API key needed
BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
INTERVALS = ["1m", "3m", "5m"]
DAYS_BACK = 30

def fetch_klines(symbol, interval, start_time, end_time, limit=1000):
    all_candles = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if not data:
            break

        all_candles.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.2)

    return all_candles

def klines_to_dataframe(klines):
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df[["open_time", "open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=DAYS_BACK)).timestamp() * 1000)

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"Fetching {symbol} - {interval}...")
            raw_klines = fetch_klines(symbol, interval, start_time, end_time)
            df = klines_to_dataframe(raw_klines)
            filename = f"data_{symbol}_{interval}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} rows to {filename}")