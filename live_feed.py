"""
live_feed.py - keeps your PostgreSQL 'candles' table up to date with Binance.

    python live_feed.py          keep running: check Binance every 30 seconds (stop with Ctrl+C)
    python live_feed.py --once   catch up once, then stop (good for testing)

What it does, for each coin (BTCUSDT, XRPUSDT, SOLUSDT) and each timeframe (1m, 3m, 5m):
  1. finds the newest candle already in your database,
  2. asks Binance for every candle since a little BEFORE that point,
  3. keeps only COMPLETED candles (a candle that is still forming is ignored),
  4. saves them. If a candle is already stored it is overwritten with the final values,
     which repairs any unfinished candle that was saved earlier.
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg2
import requests

from db_data import DB_CONFIG   # same database settings as the rest of the project

BASE_URL = "https://api.binance.com/api/v3/klines"   # public data, no API key needed
SYMBOLS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
INTERVALS = ["1m", "3m", "5m"]
INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000}
RECHECK_CANDLES = 20        # re-read this many stored candles each time (repairs unfinished ones)
POLL_SECONDS = 30           # how often to check Binance when running continuously
PAGE_SIZE = 1000            # Binance returns at most 1000 candles per request

UPSERT = """
    INSERT INTO candles (symbol, interval, open_time, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
        close = EXCLUDED.close, volume = EXCLUDED.volume
"""


def now_ms():
    return int(time.time() * 1000)


def to_ms(dt):
    """Database time -> milliseconds since 1970 (treat a time with no zone as UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def column_has_timezone(cur):
    """Is candles.open_time 'timestamp with time zone'? Decides how we write new times."""
    cur.execute("""SELECT data_type FROM information_schema.columns
                   WHERE table_name = 'candles' AND column_name = 'open_time'""")
    row = cur.fetchone()
    return bool(row) and "with time zone" in row[0]


def make_time(ms, with_tz):
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt if with_tz else dt.replace(tzinfo=None)


def fetch_closed_candles(session, symbol, interval, start_ms):
    """All COMPLETED candles from start_ms until now, as lists [open_ms, o, h, l, c, volume]."""
    step = INTERVAL_MS[interval]
    out, cursor_ms = [], start_ms
    while True:
        r = session.get(BASE_URL, params={"symbol": symbol, "interval": interval,
                                          "startTime": cursor_ms, "limit": PAGE_SIZE}, timeout=15)
        data = r.json()
        if not isinstance(data, list):                      # Binance sent an error message
            raise RuntimeError(f"Binance said: {data}")
        if not data:
            break
        out.extend(data)
        cursor_ms = data[-1][0] + 1
        if len(data) < PAGE_SIZE:
            break
        time.sleep(0.2)
    now = now_ms()
    return [k for k in out if k[0] + step <= now]          # drop the still-forming candle


def update_one(conn, session, symbol, interval, with_tz):
    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM candles WHERE symbol = %s AND interval = %s",
                (symbol, interval))
    newest = cur.fetchone()[0]
    step = INTERVAL_MS[interval]
    if newest is None:
        start_ms = now_ms() - 24 * 3600 * 1000              # empty table: start with the last day
        newest_ms = 0
    else:
        newest_ms = to_ms(newest)
        start_ms = newest_ms - RECHECK_CANDLES * step
    candles = fetch_closed_candles(session, symbol, interval, start_ms)
    for k in candles:
        cur.execute(UPSERT, (symbol, interval, make_time(k[0], with_tz),
                             float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
    conn.commit()
    cur.close()
    return sum(1 for k in candles if k[0] > newest_ms), len(candles)


def run_once(conn, session, with_tz):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            try:
                new, checked = update_one(conn, session, symbol, interval, with_tz)
                if new:
                    print(f"[{stamp} UTC] {symbol} {interval}: {new} new candle(s) saved")
            except Exception as e:                           # keep running if one request fails
                conn.rollback()
                print(f"[{stamp} UTC] {symbol} {interval}: problem - {e}")


if __name__ == "__main__":
    once = "--once" in sys.argv
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    with_tz = column_has_timezone(cur)
    cur.close()
    print("Connected to the database. Live feed started"
          + (" (single catch-up run)." if once else f" - checking every {POLL_SECONDS} s. Press Ctrl+C to stop."))
    session = requests.Session()
    try:
        run_once(conn, session, with_tz)
        if once:
            print("Catch-up finished.")
        else:
            while True:
                time.sleep(POLL_SECONDS)
                run_once(conn, session, with_tz)
    except KeyboardInterrupt:
        print("\nLive feed stopped.")
    finally:
        conn.close()
