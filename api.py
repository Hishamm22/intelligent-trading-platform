"""
api.py - the FastAPI backend. Start it with:   python -m uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs in your browser to try every endpoint.

source=db      -> newest real candles from your PostgreSQL database (default)
source=replay  -> the held-out TEST period, stepped through candle by candle (out-of-sample demo)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import trade_logic as tl

app = FastAPI(title="Intelligent Trading Platform API",
              description="Explainable trade checker: signal, verdict, explanation, risk and results.")

# lets the React dashboard (a different port) call this API from the browser
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Symbol = Query("BTCUSDT", pattern="^(BTCUSDT|XRPUSDT|SOLUSDT)$")
Interval = Query("5m", pattern="^(1m|3m|5m)$")
Source = Query("db", pattern="^(db|replay)$", description="db = real candles from PostgreSQL; replay = held-out test period")


def _call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        name = type(e).__name__
        if "psycopg2" in type(e).__module__:
            raise HTTPException(status_code=503, detail=f"Database problem ({name}): {str(e)[:200]}. "
                                "Is PostgreSQL running, and is POSTGRES_PASSWORD set in .env?")
        raise HTTPException(status_code=500, detail=f"{name}: {str(e)[:300]}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    """How fresh is the data in the database?"""
    return _call(tl.status)


@app.get("/signal")
def signal(symbol: str = Symbol, interval: str = Interval,
           index: int = Query(-1, description="Candle position; -1 = newest"), source: str = Source):
    return _call(tl.signal, symbol, interval, index, source)


@app.get("/series")
def series(symbol: str = Symbol, interval: str = Interval, end: int = -1,
           n: int = Query(200, ge=20, le=1000), source: str = Source):
    return _call(tl.series, symbol, interval, end, n, source)


@app.get("/risk")
def risk(symbol: str = Symbol, interval: str = Interval, index: int = -1,
         balance: float = Query(1000, gt=0), risk_pct: float = Query(0.01, gt=0, le=1),
         entry: float | None = Query(None, description="Entry price; default = current price"),
         source: str = Source):
    return _call(tl.risk_plan, symbol, interval, index, balance, risk_pct, entry, source)


@app.get("/results/{name}")
def results(name: str):
    return _call(tl.results, name)


@app.get("/glossary")
def glossary():
    return tl.GLOSSARY
