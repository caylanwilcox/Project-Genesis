"""
Daily Direction Prediction Model

Predicts:
1. Daily direction (bullish/bearish/neutral)
2. Predicted price range
3. Best FVG type to trade today
4. Overall confidence

Features used:
- Previous day's OHLCV
- Technical indicators (RSI, MACD, ATR, SMAs)
- Day of week
- Recent momentum
- Volatility regime
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pickle
import os
import json

# Try to import yfinance for historical data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")


def fetch_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical daily data from Yahoo Finance"""
    if not HAS_YFINANCE:
        raise ImportError("yfinance required. Run: pip install yfinance")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and features for daily prediction"""

    # Price changes
    df['daily_return'] = df['close'].pct_change() * 100
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)

    # Momentum (sum of last 3 days returns)
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)

    # Volatility
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)

    # Range
    df['daily_range'] = ((df['high'] - df['low']) / df['close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)

    # Gap from previous close
    df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100

    # RSI (14-day)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['prev_rsi'] = df['rsi_14'].shift(1)

    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Price vs MAs (shifted to use previous day's values)
    df['price_vs_sma5'] = ((df['close'].shift(1) - df['sma_5'].shift(1)) / df['sma_5'].shift(1)) * 100
    df['price_vs_sma20'] = ((df['close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100

    # SMA crossovers
    df['sma5_vs_sma20'] = ((df['sma_5'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['prev_macd_hist'] = df['macd_histogram'].shift(1)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    df['prev_atr_pct'] = df['atr_pct'].shift(1)

    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['prev_volume_ratio'] = df['volume_ratio'].shift(1)

    # Day of week (0=Monday, 4=Friday)
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

    # Is Monday/Friday
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Target: Next day's direction
    # 1 = bullish (>0.2% gain), 0 = bearish (<-0.2%), 0.5 = neutral
    df['target'] = (df['daily_return'] > 0.2).astype(int)

    # Also create a 3-class target
    df['target_3class'] = df['daily_return'].apply(
        lambda x: 2 if x > 0.3 else (0 if x < -0.3 else 1)
    )

    return df


def train_daily_model(ticker: str = 'SPY'):
    """Train daily direction prediction model"""

    print(f"\n{'='*60}")
    print(f"Training Daily Direction Model for {ticker}")
    print('='*60)

    # Fetch data
    print("\nFetching historical data...")
    df = fetch_historical_data(ticker, period="2y")
    print(f"  Got {len(df)} days of data")

    # Calculate features
    print("Calculating features...")
    df = calculate_features(df)

    # Feature columns
    feature_cols = [
        'prev_return', 'prev_2_return', 'prev_3_return',
        'momentum_3d', 'momentum_5d',
        'volatility_5d', 'volatility_10d',
        'prev_range', 'avg_range_5d',
        'gap',
        'prev_rsi',
        'price_vs_sma5', 'price_vs_sma20', 'price_vs_sma50',
        'sma5_vs_sma20',
        'prev_macd_hist',
        'prev_atr_pct',
        'prev_volume_ratio',
        'day_of_week', 'is_monday', 'is_friday'
    ]

    # Drop rows with NaN
    df_clean = df.dropna(subset=feature_cols + ['target'])
    print(f"  {len(df_clean)} samples after cleaning")

    # Split: train on 2023-2024, test on 2025
    train_mask = pd.to_datetime(df_clean['date']) < '2025-01-01'
    test_mask = pd.to_datetime(df_clean['date']) >= '2025-01-01'

    X_train = df_clean[train_mask][feature_cols]
    y_train = df_clean[train_mask]['target']
    X_test = df_clean[test_mask][feature_cols]
    y_test = df_clean[test_mask]['target']

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Bullish days (train): {y_train.mean():.1%}")
    print(f"  Bullish days (test): {y_test.mean():.1%}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print("\nTraining ensemble model...")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_acc = rf.score(X_test_scaled, y_test)
    print(f"  Random Forest: {rf_acc:.1%}")

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_acc = gb.score(X_test_scaled, y_test)
    print(f"  Gradient Boosting: {gb_acc:.1%}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    print(f"  Logistic Regression: {lr_acc:.1%}")

    # Ensemble prediction
    weights = {'rf': 0.4, 'gb': 0.35, 'lr': 0.25}

    y_pred_proba = (
        rf.predict_proba(X_test_scaled)[:, 1] * weights['rf'] +
        gb.predict_proba(X_test_scaled)[:, 1] * weights['gb'] +
        lr.predict_proba(X_test_scaled)[:, 1] * weights['lr']
    )
    y_pred = (y_pred_proba >= 0.5).astype(int)
    ensemble_acc = (y_pred == y_test).mean()
    print(f"  Ensemble: {ensemble_acc:.1%}")

    # High confidence predictions
    high_conf_mask = (y_pred_proba >= 0.65) | (y_pred_proba <= 0.35)
    if high_conf_mask.sum() > 0:
        high_conf_acc = (y_pred[high_conf_mask] == y_test.values[high_conf_mask]).mean()
        print(f"\n  High confidence predictions: {high_conf_mask.sum()}")
        print(f"  High confidence accuracy: {high_conf_acc:.1%}")

    # Feature importance
    print("\nTop features (Random Forest):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # Save model
    model_data = {
        'models': {
            'rf': rf,
            'gb': gb,
            'lr': lr
        },
        'weights': weights,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': float(ensemble_acc),
            'rf_accuracy': float(rf_acc),
            'gb_accuracy': float(gb_acc),
            'lr_accuracy': float(lr_acc),
            'high_conf_accuracy': float(high_conf_acc) if high_conf_mask.sum() > 0 else 0,
            'high_conf_count': int(high_conf_mask.sum()),
            'bullish_rate_train': float(y_train.mean()),
            'bullish_rate_test': float(y_test.mean()),
        },
        'feature_importance': importance.to_dict('records'),
        'ticker': ticker,
        'version': 'daily_v1',
        'trained_at': datetime.now().isoformat(),
    }

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f'{ticker.lower()}_daily_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✓ Model saved to {model_path}")

    return model_data


def get_daily_prediction(ticker: str, features: dict = None) -> dict:
    """Get daily direction prediction for a ticker"""

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, f'{ticker.lower()}_daily_model.pkl')

    if not os.path.exists(model_path):
        return {'error': f'No daily model for {ticker}'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # If no features provided, fetch latest data
    if features is None:
        df = fetch_historical_data(ticker, period="3mo")
        df = calculate_features(df)
        latest = df.iloc[-1]

        features = {col: latest[col] for col in model_data['feature_cols']}

    # Build feature vector
    X = pd.DataFrame([features])[model_data['feature_cols']]
    X_scaled = model_data['scaler'].transform(X)

    # Get predictions from each model
    rf_prob = model_data['models']['rf'].predict_proba(X_scaled)[0][1]
    gb_prob = model_data['models']['gb'].predict_proba(X_scaled)[0][1]
    lr_prob = model_data['models']['lr'].predict_proba(X_scaled)[0][1]

    # Ensemble
    weights = model_data['weights']
    bullish_prob = (
        rf_prob * weights['rf'] +
        gb_prob * weights['gb'] +
        lr_prob * weights['lr']
    )

    # Direction
    if bullish_prob >= 0.6:
        direction = 'BULLISH'
        direction_emoji = '🟢'
    elif bullish_prob <= 0.4:
        direction = 'BEARISH'
        direction_emoji = '🔴'
    else:
        direction = 'NEUTRAL'
        direction_emoji = '🟡'

    # Confidence
    confidence = abs(bullish_prob - 0.5) * 2  # 0 to 1 scale
    if confidence >= 0.5:
        conf_tier = 'HIGH'
    elif confidence >= 0.25:
        conf_tier = 'MEDIUM'
    else:
        conf_tier = 'LOW'

    # FVG recommendation
    if bullish_prob >= 0.55:
        fvg_recommendation = 'BULLISH FVGs'
        fvg_avoid = 'bearish setups'
    elif bullish_prob <= 0.45:
        fvg_recommendation = 'BEARISH FVGs'
        fvg_avoid = 'bullish setups'
    else:
        fvg_recommendation = 'EITHER'
        fvg_avoid = 'low-confidence setups'

    return {
        'ticker': ticker,
        'direction': direction,
        'direction_emoji': direction_emoji,
        'bullish_probability': round(bullish_prob, 3),
        'bearish_probability': round(1 - bullish_prob, 3),
        'confidence': round(confidence, 3),
        'confidence_tier': conf_tier,
        'fvg_recommendation': fvg_recommendation,
        'fvg_avoid': fvg_avoid,
        'model_accuracy': model_data['metrics']['accuracy'],
        'model_version': model_data['version'],
        'generated_at': datetime.now().isoformat(),
    }


def get_morning_briefing(tickers: list = ['SPY', 'QQQ', 'IWM']) -> dict:
    """Generate morning briefing for multiple tickers"""

    briefing = {
        'generated_at': datetime.now().isoformat(),
        'market_day': datetime.now().strftime('%A, %B %d, %Y'),
        'tickers': {},
        'overall_bias': None,
        'best_opportunity': None,
    }

    bullish_count = 0
    best_conf = 0
    best_ticker = None

    for ticker in tickers:
        try:
            pred = get_daily_prediction(ticker)
            if 'error' not in pred:
                briefing['tickers'][ticker] = pred

                if pred['direction'] == 'BULLISH':
                    bullish_count += 1
                elif pred['direction'] == 'BEARISH':
                    bullish_count -= 1

                if pred['confidence'] > best_conf:
                    best_conf = pred['confidence']
                    best_ticker = ticker
        except Exception as e:
            briefing['tickers'][ticker] = {'error': str(e)}

    # Overall market bias
    if bullish_count >= 2:
        briefing['overall_bias'] = 'BULLISH'
        briefing['overall_emoji'] = '🟢'
    elif bullish_count <= -2:
        briefing['overall_bias'] = 'BEARISH'
        briefing['overall_emoji'] = '🔴'
    else:
        briefing['overall_bias'] = 'MIXED'
        briefing['overall_emoji'] = '🟡'

    # Best opportunity
    if best_ticker and best_conf > 0.3:
        briefing['best_opportunity'] = {
            'ticker': best_ticker,
            'confidence': best_conf,
            'direction': briefing['tickers'][best_ticker]['direction'],
        }

    return briefing


if __name__ == '__main__':
    # Train models for all tickers
    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            train_daily_model(ticker)
        except Exception as e:
            print(f"Error training {ticker}: {e}")

    # Test morning briefing
    print("\n" + "="*60)
    print("MORNING BRIEFING TEST")
    print("="*60)

    briefing = get_morning_briefing()
    print(f"\n{briefing['market_day']}")
    print(f"Overall Bias: {briefing['overall_emoji']} {briefing['overall_bias']}")

    print("\nTicker Predictions:")
    for ticker, pred in briefing['tickers'].items():
        if 'error' not in pred:
            print(f"  {ticker}: {pred['direction_emoji']} {pred['direction']} ({pred['bullish_probability']:.0%} bullish)")
            print(f"         Trade: {pred['fvg_recommendation']}, Avoid: {pred['fvg_avoid']}")

    if briefing['best_opportunity']:
        print(f"\nBest Opportunity: {briefing['best_opportunity']['ticker']}")
