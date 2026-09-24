import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trading_platform",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD")
}

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
INTERVALS = ["1m", "3m", "5m"]

def insert_dataframe(conn, df, symbol, interval):
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO candles (symbol, interval, open_time, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, open_time) DO NOTHING
    """

    rows_inserted = 0
    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            symbol,
            interval,
            row["open_time"],
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"]
        ))
        rows_inserted += cursor.rowcount

    conn.commit()
    cursor.close()
    return rows_inserted

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected to database successfully.")

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            filename = f"data_{symbol}_{interval}.csv"
            print(f"Loading {filename}...")

            df = pd.read_csv(filename, parse_dates=["open_time"])
            inserted = insert_dataframe(conn, df, symbol, interval)

            print(f"Inserted {inserted} new rows from {filename}")

    conn.close()
    print("Migration complete.")