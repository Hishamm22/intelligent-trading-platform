"""
backtest.py - the backtesting + paper-trading logic (a module; you do not run it directly).

Long-only spot trading, one position at a time, fees charged on BOTH sides.
Prices are rebuilt from the 'return' column, so stops/targets are checked at each
candle CLOSE (no intra-candle highs/lows) - a stated simplification.

Two exit styles
  "fixed"    : stop-loss + take-profit set at entry (scaled to recent volatility),
               plus a maximum holding time.
  "trailing" : NO fixed target. The stop starts below entry and moves UP as the
               price makes new highs (it never moves down). Once the trade is up by
               one stop-distance, the stop is also floored at break-even (+ fees),
               so a winning trade cannot turn into a loser ("stay in the profit zone").
"""
import numpy as np

FEE = 0.001  # 0.10% per side


def periods_per_year(interval):
    minutes = {"1m": 1, "3m": 3, "5m": 5}[interval]
    return 365 * 24 * 60 / minutes


def recent_vol(returns, window=20):
    """Mean absolute return of the last `window` candles (known at the close of each candle)."""
    import pandas as pd
    return pd.Series(returns).abs().rolling(window).mean().to_numpy()


def _metrics(equity, strat_rets, trade_rets, ppy, in_market):
    equity = np.asarray(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min())
    sd = strat_rets.std()
    sharpe = float(strat_rets.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    tr = np.asarray(trade_rets)
    wins, losses = tr[tr > 0], tr[tr <= 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("nan")
    return dict(
        total_return=float(equity[-1] / equity[0] - 1) if equity[0] else float("nan"),
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=int(len(tr)),
        win_rate=float((tr > 0).mean()) if len(tr) else float("nan"),
        avg_win=float(wins.mean()) if len(wins) else float("nan"),
        avg_loss=float(losses.mean()) if len(losses) else float("nan"),
        worst_trade=float(tr.min()) if len(tr) else float("nan"),
        profit_factor=pf,
        time_in_market=float(in_market),
    )


def run_backtest(returns, probs, vol, interval="5m", threshold=0.5, min_vol=0.0,
                 mode="fixed", risk_pct=0.01, horizon=15, stop_mult=2.0,
                 target_mult=3.0, trail_mult=2.0, max_hold=60, fee=FEE, entries=None):
    """
    returns : per-candle pct change (array)        probs : model P(up next candle) (array)
    vol     : recent_vol() array                   entries: optional boolean array that overrides the rule
    Entry rule: probs >= threshold AND vol >= min_vol (i.e. skip quiet markets).
    Position size: risk `risk_pct` of the account on the stop distance, never more than the account (no leverage).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if entries is None:
        entries = (np.asarray(probs) >= threshold) & (np.asarray(vol) >= min_vol)
    entries = entries & ~np.isnan(vol)

    cash, pos = 1.0, None          # pos = dict(value, price, peak, age, sf, tf, notional)
    eq = np.empty(n)
    trades = []
    in_mkt = 0

    for t in range(n):
        exited_now = False
        if pos is not None:
            in_mkt += 1
            pos["price"] *= 1 + r[t]
            pos["value"] *= 1 + r[t]
            pos["peak"] = max(pos["peak"], pos["price"])
            pos["age"] += 1
            p = pos["price"]
            if mode == "fixed":
                stop = 1 - pos["sf"]
                hit = p <= stop or p >= 1 + pos["tf"] or pos["age"] >= horizon
            else:
                stop = pos["peak"] * (1 - pos["sf"])
                if pos["peak"] >= 1 + pos["sf"]:               # up by one stop-distance -> floor at break-even
                    stop = max(stop, 1 + 2 * fee)
                hit = p <= stop or pos["age"] >= max_hold
            if hit or t == n - 1:
                cash += pos["value"] * (1 - fee)
                trades.append(p * (1 - fee) / (1 + fee) - 1)   # net return on the traded amount
                pos, exited_now = None, True
        if pos is None and not exited_now and entries[t] and t < n - 1:
            v = vol[t] * np.sqrt(horizon)
            sf = max((stop_mult if mode == "fixed" else trail_mult) * v, 1e-4)
            tf = target_mult * v
            equity_now = cash
            notional = min(equity_now * risk_pct / sf, equity_now / (1 + fee))
            cash -= notional * (1 + fee)
            pos = dict(value=notional, price=1.0, peak=1.0, age=0, sf=sf, tf=tf)
        eq[t] = cash + (pos["value"] if pos else 0.0)

    start = np.concatenate([[1.0], eq[:-1]])
    strat = eq / start - 1
    return _metrics(np.concatenate([[1.0], eq]), strat, trades, periods_per_year(interval), in_mkt / n)


def buy_and_hold(returns, interval="5m", fee=FEE):
    r = np.asarray(returns, dtype=float)
    eq = (1 - fee) * np.cumprod(1 + r)
    eq[-1] *= (1 - fee)
    full = np.concatenate([[1.0], eq])
    out = _metrics(full, r, [eq[-1] - 1], periods_per_year(interval), 1.0)
    return out


def random_baseline(returns, vol, n_entries, n_runs=200, seed=42, **kw):
    """Same exit rules, but entries chosen at random with the same overall frequency."""
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(vol)
    p = min(1.0, n_entries / max(valid.sum(), 1))
    res = []
    for _ in range(n_runs):
        mask = (rng.random(len(vol)) < p) & valid
        res.append(run_backtest(returns, None, vol, entries=mask, **kw))
    return res
