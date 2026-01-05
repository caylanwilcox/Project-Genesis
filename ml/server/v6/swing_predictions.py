"""
V6 SWING Model Predictions

Core prediction logic for V6 swing trade models.
"""

import numpy as np
import pandas as pd
from ..models.store import swing_v6_models, swing_3d_models, swing_1d_models


def calculate_swing_features(daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> dict:
    """
    Calculate features for swing trade prediction.

    Args:
        daily_df: Daily OHLCV DataFrame (needs at least 30 rows)
        weekly_df: Weekly OHLCV DataFrame (optional)

    Returns:
        Dictionary of features or None if insufficient data
    """
    if len(daily_df) < 30:
        return None

    features = {}
    idx = len(daily_df) - 1  # Last row

    # Ensure column names are lowercase
    if 'Close' in daily_df.columns:
        daily_df = daily_df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

    current = daily_df.iloc[idx]
    features['current_price'] = current['close']

    # =================
    # DAILY FEATURES
    # =================

    # Recent returns
    for days in [1, 3, 5, 10, 20]:
        if idx >= days:
            past_close = daily_df.iloc[idx - days]['close']
            features[f'return_{days}d'] = (current['close'] - past_close) / past_close
        else:
            features[f'return_{days}d'] = 0

    # Moving averages
    close = daily_df['close'].values
    for window in [5, 10, 20, 50]:
        if idx >= window:
            sma = np.mean(close[idx-window:idx])
            features[f'dist_from_sma_{window}'] = (current['close'] - sma) / sma
            features[f'above_sma_{window}'] = 1 if current['close'] > sma else 0
        else:
            features[f'dist_from_sma_{window}'] = 0
            features[f'above_sma_{window}'] = 0

    # Trend strength
    if idx >= 50:
        sma_5 = np.mean(close[idx-5:idx])
        sma_20 = np.mean(close[idx-20:idx])
        sma_50 = np.mean(close[idx-50:idx])
        features['sma_alignment'] = 1 if (sma_5 > sma_20 > sma_50) else (-1 if sma_5 < sma_20 < sma_50 else 0)
    else:
        features['sma_alignment'] = 0

    # Volatility
    if idx >= 20:
        returns = pd.Series(close).pct_change().dropna()[-20:]
        features['volatility_20d'] = float(returns.std())
        features['volatility_10d'] = float(pd.Series(close).pct_change().dropna()[-10:].std())
    else:
        features['volatility_20d'] = 0
        features['volatility_10d'] = 0

    # ATR
    if idx >= 14:
        tr_list = []
        for i in range(idx-14, idx):
            high = daily_df.iloc[i]['high']
            low = daily_df.iloc[i]['low']
            prev_close = daily_df.iloc[i-1]['close'] if i > 0 else daily_df.iloc[i]['open']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        features['atr_14'] = np.mean(tr_list)
        features['atr_pct'] = features['atr_14'] / current['close']
    else:
        features['atr_14'] = 0
        features['atr_pct'] = 0

    # RSI
    if idx >= 14:
        changes = pd.Series(close).diff().dropna()[-14:]
        gains = float(changes[changes > 0].sum())
        losses = float(abs(changes[changes < 0].sum()))
        if losses > 0:
            rs = gains / losses
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 100
    else:
        features['rsi_14'] = 50

    # Higher highs / Lower lows
    high = daily_df['high'].values
    low = daily_df['low'].values
    if idx >= 10:
        recent_high = np.max(high[idx-5:idx])
        prior_high = np.max(high[idx-10:idx-5])
        recent_low = np.min(low[idx-5:idx])
        prior_low = np.min(low[idx-10:idx-5])
        features['higher_high'] = 1 if recent_high > prior_high else 0
        features['higher_low'] = 1 if recent_low > prior_low else 0
        features['lower_high'] = 1 if recent_high < prior_high else 0
        features['lower_low'] = 1 if recent_low < prior_low else 0
    else:
        features['higher_high'] = 0
        features['higher_low'] = 0
        features['lower_high'] = 0
        features['lower_low'] = 0

    # Volume
    if 'volume' in daily_df.columns and idx >= 20:
        volume = daily_df['volume'].values
        avg_vol = np.mean(volume[idx-20:idx])
        features['volume_ratio'] = current['volume'] / avg_vol if avg_vol > 0 else 1
        features['volume_trend'] = np.mean(volume[idx-5:idx]) / avg_vol if avg_vol > 0 else 1
    else:
        features['volume_ratio'] = 1
        features['volume_trend'] = 1

    # Candle patterns
    body = current['close'] - current['open']
    total_range = current['high'] - current['low']
    features['body_to_range'] = abs(body) / total_range if total_range > 0 else 0
    features['is_bullish'] = 1 if body > 0 else 0
    features['upper_wick'] = (current['high'] - max(current['open'], current['close'])) / total_range if total_range > 0 else 0
    features['lower_wick'] = (min(current['open'], current['close']) - current['low']) / total_range if total_range > 0 else 0

    # Consecutive days
    consec_up = 0
    consec_down = 0
    for i in range(1, min(6, idx)):
        if daily_df.iloc[idx-i]['close'] > daily_df.iloc[idx-i]['open']:
            if consec_down == 0:
                consec_up += 1
            else:
                break
        else:
            if consec_up == 0:
                consec_down += 1
            else:
                break
    features['consec_up'] = consec_up
    features['consec_down'] = consec_down

    # Mean reversion
    features['mean_reversion'] = -features['return_5d']

    # =================
    # WEEKLY FEATURES
    # =================
    current_date = daily_df.index[idx] if hasattr(daily_df.index[idx], 'date') else pd.Timestamp(daily_df.index[idx])

    if weekly_df is not None and len(weekly_df) >= 4:
        # Ensure column names are lowercase
        if 'Close' in weekly_df.columns:
            weekly_df = weekly_df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })

        weekly_close = weekly_df['close'].values
        week_idx = len(weekly_df) - 1

        # Weekly returns
        for weeks in [1, 2, 4]:
            if week_idx >= weeks:
                past_close = weekly_df.iloc[week_idx - weeks]['close']
                features[f'weekly_return_{weeks}w'] = (weekly_close[-1] - past_close) / past_close
            else:
                features[f'weekly_return_{weeks}w'] = 0

        # Weekly SMA
        if week_idx >= 4:
            weekly_sma_4 = np.mean(weekly_close[week_idx-4:week_idx])
            features['weekly_dist_from_sma_4'] = (weekly_close[-1] - weekly_sma_4) / weekly_sma_4
            features['weekly_above_sma_4'] = 1 if weekly_close[-1] > weekly_sma_4 else 0
        else:
            features['weekly_dist_from_sma_4'] = 0
            features['weekly_above_sma_4'] = 0

        # Weekly trend
        weekly_current = weekly_df.iloc[-1]
        features['weekly_bullish'] = 1 if weekly_current['close'] > weekly_current['open'] else 0

        # Weekly RSI
        if week_idx >= 4:
            changes = pd.Series(weekly_close).diff().dropna()[-4:]
            gains = float(changes[changes > 0].sum())
            losses = float(abs(changes[changes < 0].sum()))
            if losses > 0:
                rs = gains / losses
                features['weekly_rsi'] = 100 - (100 / (1 + rs))
            else:
                features['weekly_rsi'] = 100
        else:
            features['weekly_rsi'] = 50
    else:
        features['weekly_return_1w'] = 0
        features['weekly_return_2w'] = 0
        features['weekly_return_4w'] = 0
        features['weekly_dist_from_sma_4'] = 0
        features['weekly_above_sma_4'] = 0
        features['weekly_bullish'] = 0
        features['weekly_rsi'] = 50

    # =================
    # TIME FEATURES
    # =================
    if hasattr(current_date, 'dayofweek'):
        features['day_of_week'] = current_date.dayofweek
        features['is_monday'] = 1 if current_date.dayofweek == 0 else 0
        features['is_friday'] = 1 if current_date.dayofweek == 4 else 0
        features['month'] = current_date.month
        features['week_of_year'] = current_date.isocalendar()[1]
    else:
        features['day_of_week'] = 0
        features['is_monday'] = 0
        features['is_friday'] = 0
        features['month'] = 1
        features['week_of_year'] = 1

    return features


def get_v6_swing_prediction(ticker: str, daily_df: pd.DataFrame,
                            weekly_df: pd.DataFrame = None) -> tuple:
    """Get swing prediction from V6 SWING model

    Args:
        ticker: Stock symbol (SPY, QQQ, IWM)
        daily_df: Daily OHLCV DataFrame
        weekly_df: Weekly OHLCV DataFrame (optional)

    Returns:
        Tuple of (prob_5d, prob_10d) or (None, None) if unavailable
    """
    ticker = ticker.upper()

    if ticker not in swing_v6_models:
        return None, None

    model_data = swing_v6_models[ticker]
    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']

    # Calculate features
    features = calculate_swing_features(daily_df, weekly_df)
    if features is None:
        return None, None

    # Build feature array
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale
    X_scaled = scaler.transform(X)

    # 5-day prediction
    models_5d = model_data['models_5d']
    weights_5d = model_data['weights_5d']
    prob_5d = sum(m.predict_proba(X_scaled)[0][1] * weights_5d[name] for name, m in models_5d.items())

    # 10-day prediction
    models_10d = model_data['models_10d']
    weights_10d = model_data['weights_10d']
    prob_10d = sum(m.predict_proba(X_scaled)[0][1] * weights_10d[name] for name, m in models_10d.items())

    return prob_5d, prob_10d


def get_3d_swing_prediction(ticker: str, daily_df: pd.DataFrame,
                            weekly_df: pd.DataFrame = None) -> float:
    """Get 3-day swing prediction

    Args:
        ticker: Stock symbol (SPY, QQQ, IWM)
        daily_df: Daily OHLCV DataFrame
        weekly_df: Weekly OHLCV DataFrame (optional)

    Returns:
        prob_3d or None if unavailable
    """
    ticker = ticker.upper()

    if ticker not in swing_3d_models:
        return None

    model_data = swing_3d_models[ticker]
    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']

    # Calculate features
    features = calculate_swing_features(daily_df, weekly_df)
    if features is None:
        return None

    # Build feature array
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale
    X_scaled = scaler.transform(X)

    # 3-day prediction
    models_3d = model_data['models_3d']
    weights_3d = model_data['weights_3d']
    prob_3d = sum(m.predict_proba(X_scaled)[0][1] * weights_3d[name] for name, m in models_3d.items())

    return prob_3d


def calculate_1d_features(daily_df: pd.DataFrame, weekly_df: pd.DataFrame = None) -> dict:
    """Calculate features for 1-day prediction"""
    if len(daily_df) < 20:
        return None

    df = daily_df.copy()
    latest = df.iloc[-1]

    features = {}

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    features['rsi_14'] = rsi_14.iloc[-1]

    gain5 = delta.clip(lower=0).rolling(5).mean()
    loss5 = (-delta.clip(upper=0)).rolling(5).mean()
    rs5 = gain5 / (loss5 + 1e-10)
    rsi_5 = 100 - (100 / (1 + rs5))
    features['rsi_5'] = rsi_5.iloc[-1]

    # Price changes
    features['pct_1d'] = df['close'].pct_change().iloc[-1]
    features['pct_2d'] = df['close'].pct_change(2).iloc[-1]
    features['pct_3d'] = df['close'].pct_change(3).iloc[-1]
    features['pct_5d'] = df['close'].pct_change(5).iloc[-1]

    # Streak features
    df['bullish_day'] = (df['close'] > df['open']).astype(int)

    # Calculate streaks
    up_streak = 0
    down_streak = 0
    for i in range(len(df) - 1, -1, -1):
        if df['bullish_day'].iloc[i] == 1:
            if down_streak == 0:
                up_streak += 1
            else:
                break
        else:
            if up_streak == 0:
                down_streak += 1
            else:
                break

    features['up_streak'] = up_streak
    features['down_streak'] = down_streak

    # Same direction
    features['same_dir_2d'] = int(df['bullish_day'].iloc[-1] == df['bullish_day'].iloc[-2])
    features['same_dir_3d'] = int(df['bullish_day'].iloc[-1] == df['bullish_day'].iloc[-2] == df['bullish_day'].iloc[-3])

    # Range features
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    features['daily_range'] = df['daily_range'].iloc[-1]
    avg_range_5d = df['daily_range'].rolling(5).mean().iloc[-1]
    features['range_vs_avg'] = features['daily_range'] / (avg_range_5d + 1e-10)

    # Shadows
    features['upper_shadow'] = (latest['high'] - max(latest['open'], latest['close'])) / latest['close']
    features['lower_shadow'] = (min(latest['open'], latest['close']) - latest['low']) / latest['close']

    # Position in range
    high_5d = df['high'].rolling(5).max().iloc[-1]
    low_5d = df['low'].rolling(5).min().iloc[-1]
    features['pct_from_high_5d'] = (latest['close'] - high_5d) / high_5d
    features['pct_from_low_5d'] = (latest['close'] - low_5d) / low_5d
    features['range_position_5d'] = (latest['close'] - low_5d) / (high_5d - low_5d + 0.001)

    high_10d = df['high'].rolling(10).max().iloc[-1]
    low_10d = df['low'].rolling(10).min().iloc[-1]
    features['range_position_10d'] = (latest['close'] - low_10d) / (high_10d - low_10d + 0.001)

    # Volume
    avg_vol_20 = df['volume'].rolling(20).mean().iloc[-1]
    features['volume_ratio'] = latest['volume'] / (avg_vol_20 + 1)
    avg_vol_3 = df['volume'].rolling(3).mean().iloc[-1]
    avg_vol_10 = df['volume'].rolling(10).mean().iloc[-1]
    features['volume_trend'] = avg_vol_3 / (avg_vol_10 + 1)

    # Weekly context
    if weekly_df is not None and len(weekly_df) >= 2:
        features['weekly_bullish'] = int(weekly_df['close'].iloc[-1] > weekly_df['open'].iloc[-1])
        features['prior_week_bullish'] = int(weekly_df['close'].iloc[-2] > weekly_df['open'].iloc[-2])
    else:
        features['weekly_bullish'] = 0
        features['prior_week_bullish'] = 0

    # Day of week
    if 'date' in df.columns or 'timestamp' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        day_of_week = pd.to_datetime(df[date_col]).iloc[-1].dayofweek
    else:
        day_of_week = 2  # Default to mid-week
    features['is_monday'] = int(day_of_week == 0)
    features['is_friday'] = int(day_of_week == 4)

    # Gap
    if len(df) >= 2:
        features['gap'] = (latest['open'] - df['close'].iloc[-2]) / df['close'].iloc[-2]
    else:
        features['gap'] = 0

    return features


def get_1d_swing_prediction(ticker: str, daily_df: pd.DataFrame,
                            weekly_df: pd.DataFrame = None) -> float:
    """Get 1-day swing prediction

    Args:
        ticker: Stock symbol (SPY, QQQ, IWM)
        daily_df: Daily OHLCV DataFrame
        weekly_df: Weekly OHLCV DataFrame (optional)

    Returns:
        prob_1d or None if unavailable
    """
    ticker = ticker.upper()

    if ticker not in swing_1d_models:
        return None

    model_data = swing_1d_models[ticker]
    feature_cols = model_data['feature_cols']
    scaler = model_data['scaler']

    # Calculate features
    features = calculate_1d_features(daily_df, weekly_df)
    if features is None:
        return None

    # Build feature array
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale
    X_scaled = scaler.transform(X)

    # 1-day prediction
    models_1d = model_data['models_1d']
    weights_1d = model_data['weights_1d']
    prob_1d = sum(m.predict_proba(X_scaled)[0][1] * weights_1d[name] for name, m in models_1d.items())

    return prob_1d
