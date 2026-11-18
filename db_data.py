"""
db_data.py - reads REAL candles from your PostgreSQL database and turns them into the same
indicator features the models were trained on.

It uses the very same indicator functions from your features.py, so live features match
training exactly. (build_feature_set() drops the newest candle because it needs the 'next' close
for the target; here we keep the newest candle - that's the one we want to predict from.)
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import features as F

load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trading_platform",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def load_candles(symbol, interval, limit=1500):
    """Newest `limit` candles for one coin/timeframe, oldest first, prices as floats."""
    sql = """SELECT open_time, open, high, low, close, volume
             FROM candles WHERE symbol = %s AND interval = %s
             ORDER BY open_time DESC LIMIT %s"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(sql, (symbol, interval, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f"No candles in the database for {symbol} {interval}.")
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"])
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    else:
        df["open_time"] = df["open_time"].dt.tz_convert("UTC")
    return df.sort_values("open_time").reset_index(drop=True)


def compute_features(raw):
    """Same indicators as features.build_feature_set, minus the target, keeping the newest candle."""
    df = raw.copy()
    df["return"] = df["close"].pct_change()
    df["rsi"] = F.calculate_rsi(df)
    macd, macd_signal, macd_hist = F.calculate_macd(df)
    sma_short, sma_long, ema_short, ema_long = F.calculate_moving_averages(df)
    bb_mid, bb_upper, bb_lower = F.calculate_bollinger_bands(df)
    vol_change, vol_sma, rel_vol = F.calculate_volume_features(df)

    df["macd_pct"] = macd / df["close"]
    df["macd_signal_pct"] = macd_signal / df["close"]
    df["macd_hist_pct"] = macd_hist / df["close"]
    df["price_vs_sma_short"] = df["close"] / sma_short - 1
    df["price_vs_sma_long"] = df["close"] / sma_long - 1
    df["ema_spread"] = (ema_short - ema_long) / df["close"]
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    df["volume_change"] = vol_change
    df["relative_volume"] = rel_vol

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def status():
    """Row counts and first/last candle for every coin/timeframe (data freshness check)."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time)
                       FROM candles GROUP BY symbol, interval ORDER BY symbol, interval""")
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    out = []
    for sym, itv, n, first, last in rows:
        last = pd.Timestamp(last)
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        age_h = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600
        out.append(dict(symbol=sym, interval=itv, candles=int(n), first=str(first), last=str(last),
                        hours_since_last_candle=round(age_h, 1)))
    return out
