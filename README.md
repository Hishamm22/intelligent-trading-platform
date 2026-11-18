# Intelligent Trading Platform with Real-Time Market Analysis and Risk Management

Final-year project, BEng Computer Science, Anglia Ruskin University (Hisham Ahmad, SID 2318781; supervisor Dr Vitaliy Milke).

An **explainable** crypto trade-checker. For BTCUSDT, XRPUSDT and SOLUSDT on 1m / 3m / 5m candles it:

- predicts the chance that the next candle closes higher (XGBoost on 12 scale-free technical indicators),
- explains each prediction in plain English with SHAP,
- gives a red / amber / green verdict that also checks fees, quiet markets and model confidence,
- sizes a trade with a volatility-based stop-loss and take-profit (risk management),
- lets you practise in a paper-trading simulator (pretend money only; no real orders are ever sent).

**Honest headline result:** the models have a small but real predictive edge (test AUC about 0.51 to 0.535; on 47 further days of data they had never seen, forward AUC 0.511 to 0.542, every 95% range above 0.5). That edge is **not large enough to trade profitably after 0.10% fees per side**: all backtests lose money after fees. This is a research and decision-support tool, not a money machine, and nothing here is financial advice.

## How it fits together

```
Binance public API --> live_feed.py --> PostgreSQL (candles) --> api.py (FastAPI) --> dashboard.html
                                                |                    |
                                       build_datasets.py       trade_logic.py (features, models, SHAP, risk)
                                       train / test scripts    models/*.joblib (saved XGBoost)
                                                |
                                            results/*.csv
```

Only public market data is used. No personal data is collected and no API key is needed.

## Requirements

- Python 3.12, PostgreSQL 15+ (running on localhost:5432), Windows or Linux
- A PostgreSQL database called `trading_platform` containing a `candles` table:

```sql
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    interval  TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
    close DOUBLE PRECISION, volume DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, open_time)
);
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a file called `.env` next to the code (it is git-ignored, never commit it):

```
POSTGRES_PASSWORD=your-database-password
```

## Run the dashboard (everyday use)

Open two terminals in the project folder (with the venv activated):

```powershell
python live_feed.py                       # terminal 1: keeps the database up to date (every 30 s)
python -m uvicorn api:app --reload        # terminal 2: starts the API and serves the dashboard
```

Then open **http://127.0.0.1:8000/** in a browser. The interactive API documentation is at http://127.0.0.1:8000/docs.

The dashboard has two data modes: **Live (database)** and **Replay**, which steps through the held-out test period one candle at a time. Replay is the honest way to demo, because every signal shown is a genuine out-of-sample prediction.

## Reproducing the experiments

Run these in order. **Warning:** `build_datasets.py` re-splits whatever is in the database, which now contains more data than the original experiments used, so re-running it changes every result. The published numbers come from the original split.

| Step | Script | Output |
|---|---|---|
| 1 | `fetch_historical_data.py` | `data_<coin>_<tf>.csv` (public Binance candles) |
| 2 | `migrate_to_postgres.py` | loads the CSVs into the `candles` table |
| 3 | `build_datasets.py` | `datasets/` (chronological 70% / 15% / 15% split) |
| 4 | `train_baseline.py`, `train_models.py` | validation results (`results/*.csv`) |
| 5 | `final_test.py` | test-set results, confidence and calibration |
| 6 | `explain_shap.py`, `explanations.py` | SHAP importance and plain-English explanations |
| 7 | `run_backtests.py`, `run_fee_sensitivity.py`, `risk_of_ruin.py` | fee-aware backtests (0.10% per side), random-entry baselines, risk of ruin |
| 8 | `lstm_test.py`, `weighting_test.py`, `volatility_test.py` | extra checks (LSTM, sample weighting, volatility predictability) |
| 9 | `save_model.py` | `models/*.joblib` used by the API |
| 10 | `forward_test.py` | `results/forward_test_*.csv`: saved models on candles after the test period |

## Main files

- `live_feed.py`: closed-candle-only upsert from Binance into PostgreSQL
- `db_data.py`: reads candles and builds the same features as training
- `features.py`: RSI, MACD, moving averages, Bollinger Bands, volume features
- `trade_logic.py`: signal, verdict, series, risk plan, results
- `api.py`: FastAPI endpoints `/signal`, `/series`, `/risk`, `/results/{name}`, `/glossary`, `/status`, `/health` and the dashboard at `/`
- `backtest.py`, `risk.py`, `model_utils.py`: backtesting, position sizing, evaluation helpers
- `dashboard.html`: the single-page dashboard (no build step)

## Method notes

- Chronological split, never shuffled; the test period is used once.
- 12 scale-free features, so models are not tied to a particular price level.
- Strategy settings are chosen on the validation period only, then applied once to test or forward data.
- Results are compared with buy-and-hold and 200 random-entry traders using the same exit rules.
- Stops and targets are checked at candle closes in the backtests (a stated simplification); the paper-trading simulator uses candle highs and lows and assumes the stop hits first if both are touched.

## Limitations

Short history (months, not years), three coins, spot long-only, simplified fee model, no slippage or market impact, and results from one market regime. The edge is small, so conclusions about profitability should not be generalised.
