import math

FEE_RATE = 0.001  # 0.1% per side: a placeholder, to be confirmed against the exchange's fee schedule


def volatility_levels(entry, recent_vol, horizon=15, stop_mult=2.0, target_mult=3.0):
    expected_move = recent_vol * math.sqrt(horizon) * entry
    stop = entry - stop_mult * expected_move
    target = entry + target_mult * expected_move
    return stop, target


def position_size(balance, risk_pct, entry, stop, fee_rate=FEE_RATE):
    if not 0 < risk_pct <= 1:
        raise ValueError("risk_pct must be between 0 and 1, e.g. 0.01 for 1%")
    if stop >= entry:
        raise ValueError("for a long trade the stop must be below the entry")

    loss_per_unit = (entry - stop) + fee_rate * (entry + stop)
    units = balance * risk_pct / loss_per_unit

    max_units = balance / (entry * (1 + fee_rate))
    capped = units > max_units
    units = min(units, max_units)

    return {
        "units": units,
        "cost": units * entry,
        "loss_if_stopped": units * loss_per_unit,
        "loss_pct_of_balance": units * loss_per_unit / balance,
        "capped_by_balance": capped,
    }


def risk_reward(entry, stop, target, fee_rate=FEE_RATE):
    loss = (entry - stop) + fee_rate * (entry + stop)
    gain = (target - entry) - fee_rate * (entry + target)
    return {"loss_per_unit": loss, "gain_per_unit": gain,
            "ratio": gain / loss if loss > 0 else float("nan")}


if __name__ == "__main__":
    entry, balance = 84000.0, 10000.0
    recent_vol = 0.0005
    stop, target = volatility_levels(entry, recent_vol)
    print(f"entry {entry:,.0f}  stop {stop:,.0f}  target {target:,.0f}")
    print(position_size(balance, 0.01, entry, stop))
    print(risk_reward(entry, stop, target))