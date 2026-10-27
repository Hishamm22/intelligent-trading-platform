import numpy as np
import shap
from xgboost import XGBClassifier
from model_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_split


def describe(name, value, typical):
    if name == "return":
        return f"the last candle moved {value:+.2%}"
    if name == "rsi":
        if value < 30:
            return f"RSI is low ({value:.0f}), which signals recent selling"
        if value > 70:
            return f"RSI is high ({value:.0f}), which signals recent buying"
        return f"RSI is neutral ({value:.0f})"
    if name in ("macd_pct", "macd_signal_pct", "macd_hist_pct"):
        label = {"macd_pct": "MACD line", "macd_signal_pct": "MACD signal line",
                 "macd_hist_pct": "MACD histogram"}[name]
        return f"the {label} is {'positive' if value > 0 else 'negative'}"
    if name == "price_vs_sma_short":
        side = "above" if value > 0 else "below"
        return f"price is {abs(value):.2%} {side} its 10-candle average"
    if name == "price_vs_sma_long":
        side = "above" if value > 0 else "below"
        return f"price is {abs(value):.2%} {side} its 50-candle average"
    if name == "ema_spread":
        side = "above" if value > 0 else "below"
        return f"the short-term trend is {side} the longer-term trend"
    if name == "bb_width":
        level = "higher" if value > typical else "lower"
        return f"volatility (Bollinger Band width) is {level} than usual"
    if name == "bb_position":
        if value < 0:
            return "price is below the lower Bollinger Band"
        if value > 1:
            return "price is above the upper Bollinger Band"
        return f"price sits {value:.0%} of the way up the Bollinger Bands"
    if name == "relative_volume":
        return f"volume is {value:.1f}x its recent average"
    if name == "volume_change":
        return f"volume changed {value:+.0%} from the previous candle"
    return name


def explain_prediction(row, shap_row, prob, typical, top_n=3):
    direction = "UP" if prob >= 0.5 else "NOT UP"
    lines = [f"Prediction: next candle {direction} (P(up) = {prob:.1%})."]
    lines.append("Main reasons:")
    order = np.argsort(-np.abs(shap_row))[:top_n]
    for j in order:
        name = FEATURE_COLUMNS[j]
        push = "toward UP" if shap_row[j] > 0 else "toward NOT UP"
        lines.append(f"  - {describe(name, row[name], typical[name])} (pushes {push})")
    lines.append("Reliability: weak. Short-term direction is hard to predict, "
                 "so treat this as one input, not advice.")
    return "\n".join(lines)


if __name__ == "__main__":
    symbol, interval = "BTCUSDT", "1m"
    train = load_split(symbol, interval, "train")
    val = load_split(symbol, interval, "val")

    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, eval_metric="logloss",
        n_jobs=-1, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    typical = train[FEATURE_COLUMNS].median()
    prob = model.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    shap_values = shap.TreeExplainer(model).shap_values(val[FEATURE_COLUMNS])

    for i in [int(np.argmax(prob)), int(np.argmin(prob)), len(val) // 2]:
        print(f"\n{symbol} {interval} candle at {val['open_time'].iloc[i]}")
        print(explain_prediction(val.iloc[i], shap_values[i], prob[i], typical))