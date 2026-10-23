import pandas as pd

def calculate_rsi(df, period=14, column="close"):
    delta = df[column].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9, column="close"):
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def calculate_moving_averages(df, short_window=10, long_window=50, column="close"):
    sma_short = df[column].rolling(window=short_window).mean()
    sma_long = df[column].rolling(window=long_window).mean()
    ema_short = df[column].ewm(span=short_window, adjust=False).mean()
    ema_long = df[column].ewm(span=long_window, adjust=False).mean()

    return sma_short, sma_long, ema_short, ema_long

def calculate_bollinger_bands(df, window=20, num_std=2, column="close"):
    sma = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)

    return sma, upper_band, lower_band

def calculate_volume_features(df, window=20):
    volume_change = df["volume"].pct_change()
    volume_sma = df["volume"].rolling(window=window).mean()
    relative_volume = df["volume"] / volume_sma

    return volume_change, volume_sma, relative_volume







# combines the features in one rather than calling all of them seperate.

def build_feature_set(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["rsi"] = calculate_rsi(df)
    macd, macd_signal, macd_hist = calculate_macd(df)

    sma_short, sma_long, ema_short, ema_long = calculate_moving_averages(df)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(df)
    vol_change, vol_sma, rel_vol = calculate_volume_features(df)

    # Scale-free versions of the price-level indicators
    df["macd_pct"] = macd / df["close"]
    df["macd_signal_pct"] = macd_signal / df["close"]
    df["macd_hist_pct"] = macd_hist / df["close"]
    df["price_vs_sma_short"] = df["close"] / sma_short - 1
    df["price_vs_sma_long"] = df["close"] / sma_long - 1
    df["ema_spread"] = (ema_short - ema_long) / df["close"]
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    df["volume_change"] = vol_change
    df["relative_volume"] = rel_vol

    # Target: is the NEXT candle's close higher than this candle's close?
    next_close = df["close"].shift(-1)
    df["target"] = (next_close > df["close"]).astype(int)
    df = df[next_close.notna()]

    return df