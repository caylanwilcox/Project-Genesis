"""
FVG Multi-Ticker Prediction Server (Improved v2)

Flask server that loads trained ensemble models for each ticker
(SPY, QQQ, IWM) and provides predictions via HTTP API.

Model: Ensemble (XGBoost + Random Forest + Logistic Regression)
Features: 39 (23 original + 16 engineered)

Supports:
- Per-ticker predictions using ticker-specific models
- Fallback to combined model if ticker model not available
- Batch predictions for multiple FVGs
- Confidence-based filtering (>=70% recommended)
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Model storage
models = {}  # ticker -> model_data
combined_model = None
daily_models = {}  # ticker -> daily model data
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Supported tickers
SUPPORTED_TICKERS = ['SPY', 'QQQ', 'IWM']

# Try to import yfinance for daily predictions
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Categorical encoding mappings
CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}


def load_models():
    """Load all available models on startup"""
    global models, combined_model, daily_models

    print("Loading ML models (Improved v2)...")

    # Load per-ticker models
    for ticker in SUPPORTED_TICKERS:
        model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_fvg_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[ticker] = pickle.load(f)

                m = models[ticker]['metrics']
                version = models[ticker].get('version', 'unknown')
                print(f"  ✓ {ticker} model loaded")
                print(f"      Version: {version}")
                print(f"      Accuracy: {m['accuracy']:.1%}")
                print(f"      High Conf WR: {m.get('high_conf_win_rate', 0):.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} model failed to load: {e}")

    # Load combined model
    combined_path = os.path.join(MODELS_DIR, 'combined_fvg_model.pkl')
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'rb') as f:
                combined_model = pickle.load(f)
            m = combined_model['metrics']
            print(f"  ✓ Combined model loaded (accuracy: {m['accuracy']:.1%})")
        except Exception as e:
            print(f"  ✗ Combined model failed to load: {e}")

    # Load daily direction models
    print("\nLoading Daily Direction models...")
    for ticker in SUPPORTED_TICKERS:
        daily_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
        if os.path.exists(daily_path):
            try:
                with open(daily_path, 'rb') as f:
                    daily_models[ticker] = pickle.load(f)
                m = daily_models[ticker]['metrics']
                print(f"  ✓ {ticker} daily model loaded (accuracy: {m['accuracy']:.1%})")
            except Exception as e:
                print(f"  ✗ {ticker} daily model failed to load: {e}")

    total_models = len(models) + (1 if combined_model else 0) + len(daily_models)
    print(f"\nTotal models loaded: {total_models}")
    return total_models > 0


def get_model_for_ticker(ticker):
    """Get the best model for a given ticker"""
    ticker_upper = ticker.upper() if ticker else None

    if ticker_upper and ticker_upper in models:
        return models[ticker_upper], ticker_upper

    if combined_model:
        return combined_model, 'COMBINED'

    return None, None


def engineer_features(data):
    """Engineer new features from input data"""
    features = {}

    # Get base values
    gap_size_pct = data.get('gap_size_pct', 0) or 0
    atr_14 = data.get('atr_14', 1) or 1
    sma_20 = data.get('sma_20', 1) or 1
    rsi_14 = data.get('rsi_14', 50) or 50
    macd = data.get('macd', 0) or 0
    macd_histogram = data.get('macd_histogram', 0) or 0
    bb_bandwidth = data.get('bb_bandwidth', 1) or 1
    volume_ratio = data.get('volume_ratio', 1) or 1
    price_vs_sma20 = data.get('price_vs_sma20', 0) or 0
    price_vs_sma50 = data.get('price_vs_sma50', 0) or 0
    hour_of_day = data.get('hour_of_day', 12) or 12
    day_of_week = data.get('day_of_week', 2) or 2
    fvg_type = str(data.get('fvg_type', 'unknown')).lower()
    market_structure = str(data.get('market_structure', 'unknown')).lower()

    # Engineered features
    features['gap_to_atr'] = gap_size_pct / (atr_14 / sma_20 * 100 + 0.001)
    features['trend_alignment'] = (
        (1 if price_vs_sma20 > 0 else 0) +
        (1 if price_vs_sma50 > 0 else 0) +
        (1 if macd > 0 else 0)
    ) / 3
    features['rsi_momentum'] = abs(rsi_14 - 50)
    features['volume_spike'] = 1 if volume_ratio > 1.5 else 0
    features['volatility_squeeze'] = 1 if bb_bandwidth < 2.0 else 0  # Approximate median
    features['macd_strength'] = abs(macd_histogram) / (atr_14 + 0.001)
    features['price_extension'] = abs(price_vs_sma20) / (bb_bandwidth + 0.001)
    features['is_market_open'] = 1 if 9 <= hour_of_day <= 16 else 0
    features['is_power_hour'] = 1 if 15 <= hour_of_day <= 16 else 0
    features['is_morning'] = 1 if 9 <= hour_of_day <= 11 else 0
    features['is_monday'] = 1 if day_of_week == 0 else 0
    features['is_friday'] = 1 if day_of_week == 4 else 0
    features['fvg_trend_aligned'] = 1 if (
        (fvg_type == 'bullish' and market_structure == 'bullish') or
        (fvg_type == 'bearish' and market_structure == 'bearish')
    ) else 0

    # Gap category
    if gap_size_pct <= 0.15:
        features['gap_category'] = 0
    elif gap_size_pct <= 0.3:
        features['gap_category'] = 1
    elif gap_size_pct <= 0.5:
        features['gap_category'] = 2
    elif gap_size_pct <= 1.0:
        features['gap_category'] = 3
    else:
        features['gap_category'] = 4

    features['momentum_score'] = (
        (rsi_14 - 50) / 50 +
        (1 if macd > 0 else -1) * min(abs(macd_histogram), 1) +
        price_vs_sma20 / 10
    ) / 3
    features['atr_normalized'] = atr_14 / sma_20 * 100

    return features


def build_features(data, feature_cols):
    """Build feature vector from input data"""
    features = {}

    # Basic numeric features
    numeric_cols = [
        'gap_size_pct', 'validation_score', 'rsi_14', 'macd',
        'macd_signal', 'macd_histogram', 'atr_14', 'sma_20',
        'sma_50', 'ema_12', 'ema_26', 'bb_bandwidth', 'volume_ratio',
        'price_vs_sma20', 'price_vs_sma50', 'hour_of_day', 'day_of_week'
    ]

    for col in numeric_cols:
        features[col] = data.get(col, 0) or 0

    # Encode categorical features
    for cat_col, mapping in CATEGORICAL_MAPPINGS.items():
        value = str(data.get(cat_col, 'unknown')).lower()
        encoded = mapping.get(value, mapping.get('unknown', 0))
        features[f'{cat_col}_encoded'] = encoded

    # Add engineered features
    engineered = engineer_features(data)
    features.update(engineered)

    # Create DataFrame with correct column order
    df = pd.DataFrame([features])

    # Ensure columns match feature_cols
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


def predict_ensemble(model_data, df):
    """Make prediction using ensemble model"""
    model_type = model_data.get('model_type', 'xgboost')

    if model_type == 'ensemble':
        # Get component models
        xgb_model = model_data['model']['xgb']
        rf_model = model_data['model']['rf']
        lr_model = model_data['model']['lr']
        scaler = model_data['scaler']
        weights = model_data.get('weights', {'xgb': 0.4, 'rf': 0.35, 'lr': 0.25})

        # Get probabilities from each model
        xgb_prob = xgb_model.predict_proba(df)[0][1]
        rf_prob = rf_model.predict_proba(df)[0][1]

        # Scale for logistic regression
        df_scaled = scaler.transform(df)
        lr_prob = lr_model.predict_proba(df_scaled)[0][1]

        # Weighted ensemble
        probability = (
            xgb_prob * weights['xgb'] +
            rf_prob * weights['rf'] +
            lr_prob * weights['lr']
        )

        return float(probability)

    else:
        # Legacy single model
        model = model_data['model']
        probability = model.predict_proba(df)[0][1]
        return float(probability)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'version': 'improved_v2',
        'models_loaded': {
            'per_ticker': list(models.keys()),
            'combined': combined_model is not None
        },
        'total_models': len(models) + (1 if combined_model else 0),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict for a single FVG pattern"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()

        # Get appropriate model
        model_data, model_name = get_model_for_ticker(ticker)

        if model_data is None:
            return jsonify({'error': 'No models available'}), 500

        # Build features
        df = build_features(data, model_data['feature_cols'])

        # Make prediction
        probability = predict_ensemble(model_data, df)
        prediction = int(probability >= 0.5)

        # Confidence level
        confidence = probability if prediction == 1 else (1 - probability)

        # Confidence tier
        if probability >= 0.8:
            confidence_tier = 'very_high'
        elif probability >= 0.7:
            confidence_tier = 'high'
        elif probability >= 0.6:
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        # Get high confidence metrics
        hc_win_rate = model_data['metrics'].get('high_conf_win_rate', 0)

        return jsonify({
            'prediction': 'win' if prediction == 1 else 'loss',
            'win_probability': round(float(probability), 4),
            'confidence': round(float(confidence), 4),
            'confidence_tier': confidence_tier,
            'model_used': model_name,
            'model_version': model_data.get('version', 'unknown'),
            'model_accuracy': round(float(model_data['metrics']['accuracy']), 4),
            'high_conf_win_rate': round(float(hc_win_rate), 4),
            'ticker': ticker or 'N/A',
            'recommendation': 'TRADE' if probability >= 0.7 else ('CAUTIOUS' if probability >= 0.5 else 'SKIP'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict for multiple FVGs at once"""
    try:
        data = request.json
        fvgs = data.get('fvgs', [])
        default_ticker = data.get('ticker', '')

        results = []
        for fvg in fvgs:
            ticker = fvg.get('ticker', default_ticker).upper()
            model_data, model_name = get_model_for_ticker(ticker)

            if model_data is None:
                results.append({
                    'fvg_id': fvg.get('fvg_id'),
                    'error': 'No model available',
                })
                continue

            df = build_features(fvg, model_data['feature_cols'])
            probability = predict_ensemble(model_data, df)
            prediction = int(probability >= 0.5)
            confidence = probability if prediction == 1 else (1 - probability)

            results.append({
                'fvg_id': fvg.get('fvg_id'),
                'ticker': ticker,
                'prediction': 'win' if prediction == 1 else 'loss',
                'win_probability': round(float(probability), 4),
                'confidence': round(float(confidence), 4),
                'model_used': model_name,
                'recommendation': 'TRADE' if probability >= 0.7 else ('CAUTIOUS' if probability >= 0.5 else 'SKIP'),
            })

        return jsonify({
            'predictions': results,
            'models_available': list(models.keys()) + (['COMBINED'] if combined_model else []),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about all loaded models"""
    ticker = request.args.get('ticker', '').upper()

    if ticker and ticker in models:
        model_data = models[ticker]
        return jsonify({
            'ticker': ticker,
            'version': model_data.get('version', 'unknown'),
            'model_type': model_data.get('model_type', 'unknown'),
            'feature_count': len(model_data['feature_cols']),
            'confidence_threshold': model_data.get('confidence_threshold', 0.5),
            'metrics': model_data['metrics'],
            'trained_at': model_data.get('trained_at', 'unknown'),
        })

    # Return summary of all models
    info = {
        'models': {},
        'combined': None,
        'version': 'improved_v2'
    }

    for t, m in models.items():
        info['models'][t] = {
            'accuracy': round(float(m['metrics']['accuracy']), 4),
            'high_conf_win_rate': round(float(m['metrics'].get('high_conf_win_rate', 0)), 4),
            'high_conf_pf': round(float(m['metrics'].get('high_conf_pf', 0)), 4),
            'version': m.get('version', 'unknown'),
        }

    if combined_model:
        info['combined'] = {
            'accuracy': round(float(combined_model['metrics']['accuracy']), 4),
            'high_conf_win_rate': round(float(combined_model['metrics'].get('high_conf_win_rate', 0)), 4),
        }

    return jsonify(info)


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models and their accuracies"""
    models_list = []

    for ticker, m in models.items():
        models_list.append({
            'ticker': ticker,
            'type': 'ticker-specific',
            'version': m.get('version', 'unknown'),
            'accuracy': round(float(m['metrics']['accuracy']), 4),
            'high_conf_win_rate': round(float(m['metrics'].get('high_conf_win_rate', 0)), 4),
            'high_conf_pf': round(float(m['metrics'].get('high_conf_pf', 0)), 4),
        })

    if combined_model:
        models_list.append({
            'ticker': 'ALL',
            'type': 'combined',
            'version': combined_model.get('version', 'unknown'),
            'accuracy': round(float(combined_model['metrics']['accuracy']), 4),
            'high_conf_win_rate': round(float(combined_model['metrics'].get('high_conf_win_rate', 0)), 4),
        })

    return jsonify({
        'models': models_list,
        'supported_tickers': SUPPORTED_TICKERS,
        'recommended_threshold': 0.7,
    })


def calculate_daily_features(df):
    """Calculate features for daily prediction from OHLCV data"""
    # Price changes
    df['daily_return'] = df['Close'].pct_change() * 100
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)

    # Momentum
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)

    # Volatility
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)

    # Range
    df['daily_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)

    # Gap
    df['gap'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['prev_rsi'] = df['rsi_14'].shift(1)

    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()

    df['price_vs_sma5'] = ((df['Close'].shift(1) - df['sma_5'].shift(1)) / df['sma_5'].shift(1)) * 100
    df['price_vs_sma20'] = ((df['Close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['Close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['sma5_vs_sma20'] = ((df['sma_5'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['prev_macd_hist'] = df['macd_histogram'].shift(1)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['prev_atr_pct'] = (df['atr_14'].shift(1) / df['Close'].shift(1)) * 100

    # Volume
    df['prev_volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['prev_volume_ratio'] = df['prev_volume_ratio'].shift(1)

    # Day of week
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    return df


@app.route('/daily_prediction', methods=['GET'])
def daily_prediction():
    """Get daily direction prediction for a ticker"""
    ticker = request.args.get('ticker', 'SPY').upper()

    if ticker not in daily_models:
        return jsonify({'error': f'No daily model for {ticker}'}), 404

    if not HAS_YFINANCE:
        return jsonify({'error': 'yfinance not installed on server'}), 500

    try:
        # Fetch recent data
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")

        if len(df) < 50:
            return jsonify({'error': 'Insufficient historical data'}), 500

        # Calculate features
        df = calculate_daily_features(df)
        latest = df.iloc[-1]

        model_data = daily_models[ticker]
        feature_cols = model_data['feature_cols']

        # Build feature vector
        features = {col: latest[col] if col in latest else 0 for col in feature_cols}
        X = pd.DataFrame([features])[feature_cols]
        X_scaled = model_data['scaler'].transform(X)

        # Get predictions
        weights = model_data['weights']
        rf_prob = model_data['models']['rf'].predict_proba(X_scaled)[0][1]
        gb_prob = model_data['models']['gb'].predict_proba(X_scaled)[0][1]
        lr_prob = model_data['models']['lr'].predict_proba(X_scaled)[0][1]

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
        confidence = abs(bullish_prob - 0.5) * 2
        conf_tier = 'HIGH' if confidence >= 0.5 else ('MEDIUM' if confidence >= 0.25 else 'LOW')

        # FVG recommendation
        if bullish_prob >= 0.55:
            fvg_rec = 'BULLISH FVGs'
            fvg_avoid = 'bearish setups'
        elif bullish_prob <= 0.45:
            fvg_rec = 'BEARISH FVGs'
            fvg_avoid = 'bullish setups'
        else:
            fvg_rec = 'EITHER (low conviction)'
            fvg_avoid = 'aggressive positions'

        # Price info
        current_price = float(latest['Close'])
        prev_close = float(df.iloc[-2]['Close'])
        atr = float(latest['atr_14']) if not pd.isna(latest['atr_14']) else current_price * 0.01

        return jsonify({
            'ticker': ticker,
            'direction': direction,
            'direction_emoji': direction_emoji,
            'bullish_probability': round(float(bullish_prob), 3),
            'bearish_probability': round(float(1 - bullish_prob), 3),
            'confidence': round(float(confidence), 3),
            'confidence_tier': conf_tier,
            'fvg_recommendation': fvg_rec,
            'fvg_avoid': fvg_avoid,
            'current_price': round(current_price, 2),
            'predicted_range': {
                'low': round(current_price - atr, 2),
                'high': round(current_price + atr, 2),
            },
            'model_accuracy': round(float(model_data['metrics']['accuracy']), 3),
            'high_conf_accuracy': round(float(model_data['metrics'].get('high_conf_accuracy', 0)), 3),
            'model_version': model_data.get('version', 'daily_v1'),
            'generated_at': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/morning_briefing', methods=['GET'])
def morning_briefing():
    """Get morning briefing for all tickers"""
    if not HAS_YFINANCE:
        return jsonify({'error': 'yfinance not installed on server'}), 500

    briefing = {
        'generated_at': datetime.now().isoformat(),
        'market_day': datetime.now().strftime('%A, %B %d, %Y'),
        'tickers': {},
        'overall_bias': None,
        'overall_emoji': None,
        'best_opportunity': None,
    }

    bullish_count = 0
    best_conf = 0
    best_ticker = None

    for ticker in SUPPORTED_TICKERS:
        if ticker not in daily_models:
            briefing['tickers'][ticker] = {'error': 'No model available'}
            continue

        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")

            if len(df) < 50:
                briefing['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            df = calculate_daily_features(df)
            latest = df.iloc[-1]

            model_data = daily_models[ticker]
            feature_cols = model_data['feature_cols']

            features = {col: latest[col] if col in latest else 0 for col in feature_cols}
            X = pd.DataFrame([features])[feature_cols]
            X_scaled = model_data['scaler'].transform(X)

            weights = model_data['weights']
            rf_prob = model_data['models']['rf'].predict_proba(X_scaled)[0][1]
            gb_prob = model_data['models']['gb'].predict_proba(X_scaled)[0][1]
            lr_prob = model_data['models']['lr'].predict_proba(X_scaled)[0][1]

            bullish_prob = (
                rf_prob * weights['rf'] +
                gb_prob * weights['gb'] +
                lr_prob * weights['lr']
            )

            if bullish_prob >= 0.6:
                direction = 'BULLISH'
                emoji = '🟢'
                bullish_count += 1
            elif bullish_prob <= 0.4:
                direction = 'BEARISH'
                emoji = '🔴'
                bullish_count -= 1
            else:
                direction = 'NEUTRAL'
                emoji = '🟡'

            confidence = abs(bullish_prob - 0.5) * 2

            if bullish_prob >= 0.55:
                fvg_rec = 'BULLISH'
            elif bullish_prob <= 0.45:
                fvg_rec = 'BEARISH'
            else:
                fvg_rec = 'EITHER'

            current_price = float(latest['Close'])
            atr = float(latest['atr_14']) if not pd.isna(latest['atr_14']) else current_price * 0.01

            briefing['tickers'][ticker] = {
                'direction': direction,
                'emoji': emoji,
                'bullish_probability': round(float(bullish_prob), 3),
                'confidence': round(float(confidence), 3),
                'fvg_recommendation': fvg_rec,
                'current_price': round(current_price, 2),
                'predicted_range': {
                    'low': round(current_price - atr, 2),
                    'high': round(current_price + atr, 2),
                },
                'model_accuracy': round(float(model_data['metrics']['accuracy']), 3),
            }

            if confidence > best_conf:
                best_conf = confidence
                best_ticker = ticker

        except Exception as e:
            briefing['tickers'][ticker] = {'error': str(e)}

    # Overall bias
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
    if best_ticker and best_conf > 0.2:
        briefing['best_opportunity'] = {
            'ticker': best_ticker,
            'confidence': round(best_conf, 3),
            'direction': briefing['tickers'][best_ticker].get('direction', 'UNKNOWN'),
        }

    return jsonify(briefing)


# Load models on import (for gunicorn)
load_models()

if __name__ == '__main__':
    if len(models) > 0 or combined_model is not None:
        port = int(os.environ.get('PORT', 5001))
        print("\n" + "="*50)
        print("FVG Prediction Server (Improved v2)")
        print("="*50)
        print(f"Listening on http://0.0.0.0:{port}")
        print(f"Endpoints:")
        print(f"  POST /predict       - Single FVG prediction")
        print(f"  POST /batch_predict - Batch predictions")
        print(f"  GET  /health        - Health check")
        print(f"  GET  /models        - List available models")
        print(f"  GET  /model_info    - Model details")
        print("="*50)
        print("\nRECOMMENDATION: Only trade when win_probability >= 0.7")
        print("="*50 + "\n")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Error: No models could be loaded.")
        print("Run upgrade_production_models.py first to train models.")
