import os
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

N_TRADES = 200
N_RUNS = 5000
RUIN_LEVEL = 0.5  # "ruin" = balance ever falls below half its starting value
RISK_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10]

SCENARIOS = {
    "Fair coin, no fees": (0.50, 1.00),
    "No edge, with fees": (0.40, 0.65),
    "Break-even for fees": (0.61, 0.65),
}


def simulate(p, ratio, risk_pct, n_trades=N_TRADES, n_runs=N_RUNS, seed=42):
    rng = np.random.default_rng(seed)
    wins = rng.random((n_runs, n_trades)) < p
    multiplier = np.where(wins, 1 + risk_pct * ratio, 1 - risk_pct)
    paths = np.cumprod(multiplier, axis=1)

    peak = np.maximum.accumulate(paths, axis=1)
    drawdown = 1 - paths / peak

    return {
        "median_final": np.median(paths[:, -1]),
        "p_below_start": (paths[:, -1] < 1).mean(),
        "p_ruin": (paths.min(axis=1) < RUIN_LEVEL).mean(),
        "median_max_drawdown": np.median(drawdown.max(axis=1)),
    }


rows = []
for name, (p, ratio) in SCENARIOS.items():
    for risk in RISK_LEVELS:
        result = simulate(p, ratio, risk)
        row = {"scenario": name, "risk_per_trade": f"{risk:.1%}"}
        row.update({k: round(float(v), 3) for k, v in result.items()})
        rows.append(row)

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("results/risk_of_ruin.csv", index=False)