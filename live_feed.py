import os
import json
import websocket
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

SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt"]
INTERVALS = ["1m", "3m", "5m"]

conn = psycopg2.connect(**DB_CONFIG)
print("Connected to database.")

def insert_candle(symbol, interval, kline):
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO candles (symbol, interval, open_time, open, high, low, close, volume)
        VALUES (%s, %s, to_timestamp(%s / 1000.0), %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, interval, open_time) DO NOTHING
    """

    cursor.execute(insert_query, (
        symbol.upper(),
        interval,
        kline["t"],
        kline["o"],
        kline["h"],
        kline["l"],
        kline["c"],
        kline["v"]
    ))

    conn.commit()
    cursor.close()

def on_message(ws, message):
    data = json.loads(message)
    payload = data["data"]
    kline = payload["k"]

    is_closed = kline["x"]

    if is_closed:
        symbol = payload["s"].lower()
        interval = kline["i"]
        insert_candle(symbol, interval, kline)
        print(f"Saved closed candle: {symbol.upper()} {interval} at {kline['t']}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed.")

def on_open(ws):
    print("WebSocket connection opened. Listening for live candles...")

if __name__ == "__main__":
    streams = "/".join([f"{symbol}@kline_{interval}" for symbol in SYMBOLS for interval in INTERVALS])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()