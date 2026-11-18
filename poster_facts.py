import glob
import pandas as pd
import psycopg2
from db_data import DB_CONFIG

# 1) Ask the database how much data we really have
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("""SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time)
               FROM candles GROUP BY symbol, interval ORDER BY symbol, interval""")
for row in cur.fetchall():
    print(row)
cur.execute("SELECT COUNT(*) FROM candles")
print("TOTAL CANDLES:", cur.fetchone()[0])
conn.close()

# 2) Print every results file in full so I can check the poster's numbers
for path in sorted(glob.glob("results/*.csv")):
    print("\n=====", path, "=====")
    print(pd.read_csv(path).to_string())