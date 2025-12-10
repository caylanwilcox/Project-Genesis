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
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model storage
models = {}  # ticker -> model_data
combined_model = None
daily_models = {}  # ticker -> daily model data
highlow_models = {}  # ticker -> high/low model data
shrinking_models = {}  # ticker -> shrinking range model data
regime_models = {}  # ticker -> volatility regime models
intraday_models = {}  # ticker -> intraday session update models
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Volatility regime thresholds
LOW_VOL_THRESHOLD = 0.30
HIGH_VOL_THRESHOLD = 0.70

# Supported tickers
SUPPORTED_TICKERS = ['SPY', 'QQQ', 'IWM']

# Polygon.io API for market data
import requests
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '').strip()
print(f"POLYGON_API_KEY loaded: {'Yes' if POLYGON_API_KEY else 'No'} (length: {len(POLYGON_API_KEY)})")

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

    # Load high/low prediction models
    print("\nLoading High/Low models...")
    for ticker in SUPPORTED_TICKERS:
        highlow_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
        if os.path.exists(highlow_path):
            try:
                with open(highlow_path, 'rb') as f:
                    highlow_models[ticker] = pickle.load(f)
                m = highlow_models[ticker]['metrics']
                print(f"  ✓ {ticker} high/low model loaded")
                print(f"      Capture Rate: {m.get('capture_rate', 0):.1f}%")
            except Exception as e:
                print(f"  ✗ {ticker} high/low model failed to load: {e}")

    # Load volatility regime models
    print("\nLoading Volatility Regime models...")
    for ticker in SUPPORTED_TICKERS:
        regime_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_regime_model.pkl')
        if os.path.exists(regime_path):
            try:
                with open(regime_path, 'rb') as f:
                    regime_models[ticker] = pickle.load(f)
                regimes = list(regime_models[ticker]['regime_models'].keys())
                print(f"  ✓ {ticker} regime models loaded ({', '.join(regimes)})")
            except Exception as e:
                print(f"  ✗ {ticker} regime model failed to load: {e}")

    # Load shrinking range models
    print("\nLoading Shrinking Range models...")
    for ticker in SUPPORTED_TICKERS:
        shrink_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_shrinking_model.pkl')
        if os.path.exists(shrink_path):
            try:
                with open(shrink_path, 'rb') as f:
                    shrinking_models[ticker] = pickle.load(f)
                m = shrinking_models[ticker]['metrics']
                print(f"  ✓ {ticker} shrinking model loaded")
                print(f"      Capture Rate: {m.get('capture_rate', 0):.1f}%")
            except Exception as e:
                print(f"  ✗ {ticker} shrinking model failed to load: {e}")

    # Load intraday session update models
    print("\nLoading Intraday Session models...")
    for ticker in SUPPORTED_TICKERS:
        intraday_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_model.pkl')
        if os.path.exists(intraday_path):
            try:
                with open(intraday_path, 'rb') as f:
                    intraday_models[ticker] = pickle.load(f)
                m = intraday_models[ticker]['metrics']
                print(f"  ✓ {ticker} intraday model loaded (accuracy: {m['accuracy']:.1%})")
            except Exception as e:
                print(f"  ✗ {ticker} intraday model failed to load: {e}")

    total_models = len(models) + (1 if combined_model else 0) + len(daily_models) + len(highlow_models) + len(shrinking_models) + len(intraday_models)
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


def fetch_polygon_data(ticker: str, days: int = 100) -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io"""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")

    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    print(f"Polygon API response for {ticker}: status={data.get('status')}, resultsCount={data.get('resultsCount', 0)}")

    if data.get('status') == 'ERROR':
        raise ValueError(f"Polygon API error for {ticker}: {data.get('error', 'Unknown error')}")

    if 'results' not in data or len(data['results']) == 0:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume'
    })
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df


def fetch_intraday_snapshot(ticker: str):
    """
    Fetch today's intraday snapshot: current price, today's high, today's low
    Uses Polygon's snapshot endpoint for real-time data
    """
    if not POLYGON_API_KEY:
        return None

    try:
        # Use snapshot endpoint for current price
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('ticker'):
            ticker_data = data['ticker']
            day_data = ticker_data.get('day', {})
            return {
                'current_price': ticker_data.get('lastTrade', {}).get('p') or day_data.get('c', 0),
                'today_open': day_data.get('o', 0),
                'today_high': day_data.get('h', 0),
                'today_low': day_data.get('l', 0),
                'today_volume': day_data.get('v', 0),
            }
    except Exception as e:
        print(f"Error fetching snapshot for {ticker}: {e}")

    # Fallback: use today's daily bar
    try:
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{today}/{today}"
        params = {'adjusted': 'true', 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('results') and len(data['results']) > 0:
            bar = data['results'][-1]
            return {
                'current_price': bar['c'],
                'today_open': bar['o'],
                'today_high': bar['h'],
                'today_low': bar['l'],
                'today_volume': bar['v'],
            }
    except Exception as e:
        print(f"Error fetching daily bar for {ticker}: {e}")

    return None


def get_session_progress():
    """
    Calculate how far through the trading session we are (0-1).
    0 = market open (9:30 ET)
    1 = market close (4:00 ET)
    """
    import pytz
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if now_et < market_open:
        return 0.0
    if now_et > market_close:
        return 1.0

    total_minutes = (market_close - market_open).total_seconds() / 60  # 390 minutes
    elapsed_minutes = (now_et - market_open).total_seconds() / 60

    return min(1.0, max(0.0, elapsed_minutes / total_minutes))


def predict_intraday(ticker: str, df: pd.DataFrame, snapshot: dict) -> dict:
    """
    Get intraday session-updated prediction.

    Uses current intraday data (price position, high/low so far) to update
    the daily prediction during market hours.

    Returns:
        dict with probability, confidence, time_pct, etc. or None if not available
    """
    if ticker not in intraday_models:
        return None

    if not snapshot:
        return None

    try:
        model_data = intraday_models[ticker]
        feature_cols = model_data['feature_cols']
        scaler = model_data['scaler']
        models = model_data['models']
        weights = model_data['weights']

        # Get session progress
        time_pct = get_session_progress()

        # Get snapshot data
        current_price = float(snapshot.get('current_price', 0))
        today_open = float(snapshot.get('today_open', 0))
        today_high = float(snapshot.get('today_high', 0))
        today_low = float(snapshot.get('today_low', 0))

        if today_open == 0:
            return None

        # Get previous day's data from df
        if len(df) < 2:
            return None

        prev_day = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        prev_close = float(prev_day['Close'])
        prev_high = float(prev_day['High'])
        prev_low = float(prev_day['Low'])

        # Calculate features matching train_intraday_model.py
        gap = (today_open - prev_close) / prev_close
        gap_direction = 1 if gap > 0 else (-1 if gap < 0 else 0)
        gap_size = abs(gap)
        prev_return = (prev_close - df.iloc[-3]['Close']) / df.iloc[-3]['Close'] if len(df) >= 3 else 0
        prev_range = (prev_high - prev_low) / prev_close

        # Current position features
        current_vs_open = (current_price - today_open) / today_open
        current_vs_open_direction = 1 if current_price > today_open else (-1 if current_price < today_open else 0)

        # Range so far
        range_so_far = today_high - today_low if today_high > today_low else 0.0001
        position_in_range = (current_price - today_low) / range_so_far if range_so_far > 0 else 0.5
        range_so_far_pct = range_so_far / today_open
        high_so_far_pct = (today_high - today_open) / today_open
        low_so_far_pct = (today_open - today_low) / today_open

        # Binary features
        above_open = 1 if current_price > today_open else 0
        near_high = 1 if (today_high - current_price) < (current_price - today_low) else 0

        # Gap fill status
        if gap > 0:  # Gap up
            gap_filled = 1 if today_low <= prev_close else 0
        else:  # Gap down
            gap_filled = 1 if today_high >= prev_close else 0

        # Build feature vector
        features = {
            'time_pct': time_pct,
            'time_remaining': 1 - time_pct,
            'gap': gap,
            'gap_direction': gap_direction,
            'gap_size': gap_size,
            'prev_return': prev_return,
            'prev_range': prev_range,
            'current_vs_open': current_vs_open,
            'current_vs_open_direction': current_vs_open_direction,
            'position_in_range': position_in_range,
            'range_so_far_pct': range_so_far_pct,
            'high_so_far_pct': high_so_far_pct,
            'low_so_far_pct': low_so_far_pct,
            'above_open': above_open,
            'near_high': near_high,
            'gap_filled': gap_filled,
        }

        # Create DataFrame with features in correct order
        X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])[feature_cols]
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X_scaled = scaler.transform(X)

        # Get ensemble prediction
        bullish_prob = 0.0
        for model_name, model in models.items():
            prob = model.predict_proba(X_scaled)[0][1]
            bullish_prob += prob * weights.get(model_name, 0.25)

        return {
            'probability': round(float(bullish_prob), 3),
            'confidence': round(abs(bullish_prob - 0.5) * 2, 3),
            'time_pct': round(time_pct, 2),
            'session_label': get_session_label(time_pct),
            'current_vs_open': round(current_vs_open * 100, 2),
            'position_in_range': round(position_in_range * 100, 1),
            'model_accuracy': round(float(model_data['metrics']['accuracy']), 3),
        }

    except Exception as e:
        print(f"Error in intraday prediction for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_session_label(time_pct: float) -> str:
    """Convert time percentage to human-readable session label"""
    if time_pct < 0.05:
        return "Pre-market / Open"
    elif time_pct < 0.15:
        return "First 30 min"
    elif time_pct < 0.30:
        return "Morning session"
    elif time_pct < 0.50:
        return "Mid-morning"
    elif time_pct < 0.70:
        return "Afternoon"
    elif time_pct < 0.90:
        return "Late session"
    else:
        return "Power hour"


def calculate_daily_features(df):
    """Calculate OPTIMIZED features for daily prediction (v3 - compatible with improved model)"""

    # ========== PRICE ACTION ==========
    df['daily_return'] = df['Close'].pct_change() * 100
    df['prev_return'] = df['daily_return'].shift(1)
    df['prev_2_return'] = df['daily_return'].shift(2)
    df['prev_3_return'] = df['daily_return'].shift(3)
    df['prev_5_return'] = df['daily_return'].shift(5)

    # New feature names for quant model (MUST be in decimal format, not percent!)
    ret_decimal = df['Close'].pct_change()  # Decimal format
    df['ret_lag_1'] = ret_decimal.shift(1)
    df['ret_lag_2'] = ret_decimal.shift(2)
    df['ret_lag_3'] = ret_decimal.shift(3)
    df['ret_lag_5'] = ret_decimal.shift(5)
    df['ret_lag_10'] = ret_decimal.shift(10)
    df['ret_lag_20'] = ret_decimal.shift(20)

    # Intraday return (previous day) - MUST be lagged to avoid lookahead
    df['intraday_ret'] = ((df['Close'] - df['Open']) / df['Open']).shift(1)

    # Overnight and intraday returns
    df['overnight_return'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100)
    df['gap_pct'] = df['overnight_return']  # Alias

    # ========== MOMENTUM ==========
    df['momentum_3d'] = df['daily_return'].rolling(3).sum().shift(1)
    df['momentum_5d'] = df['daily_return'].rolling(5).sum().shift(1)
    df['momentum_10d'] = df['daily_return'].rolling(10).sum().shift(1)
    df['momentum_20d'] = df['daily_return'].rolling(20).sum().shift(1)

    # Rate of Change (ROC) - Decimal format for quant model
    df['roc_5'] = (df['Close'] / df['Close'].shift(5) - 1).shift(1)
    df['roc_10'] = (df['Close'] / df['Close'].shift(10) - 1).shift(1)
    df['roc_20'] = (df['Close'] / df['Close'].shift(20) - 1).shift(1)

    # Momentum aliases for quant model (returns as decimals, not percent)
    df['mom_5'] = (df['Close'].pct_change().rolling(5).sum()).shift(1)
    df['mom_10'] = (df['Close'].pct_change().rolling(10).sum()).shift(1)
    df['mom_20'] = (df['Close'].pct_change().rolling(20).sum()).shift(1)
    df['mom_60'] = (df['Close'].pct_change().rolling(60).sum()).shift(1)

    # Momentum quality (consistency)
    df['mom_quality_10'] = df['Close'].pct_change().rolling(10).apply(lambda x: (x > 0).sum() / len(x)).shift(1)
    df['mom_quality_20'] = df['Close'].pct_change().rolling(20).apply(lambda x: (x > 0).sum() / len(x)).shift(1)

    # ========== VOLATILITY ==========
    df['volatility_5d'] = df['daily_return'].rolling(5).std().shift(1)
    df['volatility_10d'] = df['daily_return'].rolling(10).std().shift(1)
    df['volatility_20d'] = df['daily_return'].rolling(20).std().shift(1)
    df['volatility_60d'] = df['daily_return'].rolling(60).std().shift(1)
    df['vol_ratio_5_20'] = (df['volatility_5d'] / df['volatility_20d']).shift(1)
    df['vol_regime'] = df['vol_ratio_5_20']  # Alias for improved model

    # Quant model volatility aliases (annualized)
    ret = df['Close'].pct_change()
    df['vol_5'] = (ret.rolling(5).std() * np.sqrt(252)).shift(1)
    df['vol_10'] = (ret.rolling(10).std() * np.sqrt(252)).shift(1)
    df['vol_20'] = (ret.rolling(20).std() * np.sqrt(252)).shift(1)
    df['vol_60'] = (ret.rolling(60).std() * np.sqrt(252)).shift(1)
    df['vol_ratio'] = (df['vol_5'] / df['vol_20'])

    # Volatility percentile
    df['vol_percentile'] = df['vol_20'].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ).shift(1)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_5'] = tr.rolling(5).mean()
    df['prev_atr_pct'] = (df['atr_14'].shift(1) / df['Close'].shift(1)) * 100
    df['atr_pct'] = df['prev_atr_pct']  # Alias for improved model
    df['atr_ratio'] = (df['atr_5'] / df['atr_14']).shift(1)
    df['atr_regime'] = df['atr_ratio']  # Alias for improved model
    df['vol_surprise'] = (abs(df['daily_return'].shift(1)) / df['volatility_10d'] - 1)  # For improved model
    df['atr_percentile'] = df['atr_14'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).shift(1)

    # Quant model ATR alias (as pct of close)
    df['atr'] = (tr.rolling(14).mean() / df['Close']).shift(1)
    df['range_ratio'] = ((df['High'] - df['Low']) / df['Close'] / (df['atr'] + 0.0001)).shift(1)

    # ========== RANGE ANALYSIS ==========
    df['daily_range'] = ((df['High'] - df['Low']) / df['Close']) * 100
    df['daily_range_pct'] = df['daily_range'].shift(1)  # Alias for improved model
    df['prev_range'] = df['daily_range'].shift(1)
    df['avg_range_5d'] = df['daily_range'].rolling(5).mean().shift(1)
    df['avg_range_20d'] = df['daily_range'].rolling(20).mean().shift(1)
    df['range_expansion'] = (df['prev_range'] / df['avg_range_20d'])
    df['close_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)).shift(1)
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)  # Decimal format for quant model

    # Distance from 20d high/low (for improved model)
    df['dist_from_20d_high'] = ((df['Close'] - df['High'].rolling(20).max()) / df['Close'] * 100).shift(1)
    df['dist_from_20d_low'] = ((df['Close'] - df['Low'].rolling(20).min()) / df['Close'] * 100).shift(1)

    # Quant model aliases for distance
    df['dist_20d_high'] = (df['Close'] / df['High'].rolling(20).max() - 1).shift(1)
    df['dist_20d_low'] = (df['Close'] / df['Low'].rolling(20).min() - 1).shift(1)
    df['dist_52w_high'] = (df['Close'] / df['High'].rolling(252).max() - 1).shift(1)
    df['dist_52w_low'] = (df['Close'] / df['Low'].rolling(252).min() - 1).shift(1)

    # Z-score (distance from moving average in std units)
    for period in [5, 10, 20, 50]:
        ma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        df[f'zscore_{period}'] = ((df['Close'] - ma) / (std + 0.0001)).shift(1)

    # Upper/lower wick ratios (for improved model)
    df['upper_wick'] = ((df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'] + 0.001)).shift(1)
    df['lower_wick'] = ((np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'] + 0.001)).shift(1)

    # Body ratio (for quant model)
    body = abs(df['Close'] - df['Open'])
    total_range = df['High'] - df['Low']
    df['body_ratio'] = (body / (total_range + 0.0001)).shift(1)

    # ========== CONSECUTIVE DAYS (Mean Reversion) ==========
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['down_day'] = (df['Close'] < df['Open']).astype(int)

    def count_consecutive(series):
        result = []
        count = 0
        prev_val = None
        for val in series:
            if val == prev_val and val == 1:
                count += 1
            else:
                count = 1 if val == 1 else 0
            result.append(count)
            prev_val = val
        return result

    df['consec_up'] = count_consecutive(df['up_day'].values)
    df['consec_down'] = count_consecutive(df['down_day'].values)
    df['prev_consec_up'] = pd.Series(df['consec_up'].values, index=df.index).shift(1)
    df['prev_consec_down'] = pd.Series(df['consec_down'].values, index=df.index).shift(1)
    df['streak'] = df['prev_consec_up'] - df['prev_consec_down']
    df['streak_extreme'] = (abs(df['streak']) >= 3).astype(int)

    # ========== RSI ==========
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['prev_rsi'] = df['rsi_14'].shift(1)
    df['rsi'] = df['prev_rsi']  # Alias for quant model
    df['rsi_oversold'] = (df['prev_rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['prev_rsi'] > 70).astype(int)
    df['rsi_momentum'] = df['prev_rsi'] - df['rsi_14'].shift(2)
    df['rsi_change'] = df['rsi_14'] - df['rsi_14'].shift(1)  # For improved model

    # RSI 9 for improved model
    gain_9 = (delta.where(delta > 0, 0)).rolling(window=9).mean()
    loss_9 = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
    rs_9 = gain_9 / (loss_9 + 0.001)
    df['rsi_9'] = (100 - (100 / (1 + rs_9))).shift(1)

    # ========== STOCHASTIC ==========
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = ((df['Close'] - low_14) / (high_14 - low_14 + 0.001) * 100).shift(1)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_crossover'] = (df['stoch_k'] > df['stoch_d']).astype(int) - (df['stoch_k'] < df['stoch_d']).astype(int)

    # ========== WILLIAMS %R ==========
    df['williams_r'] = ((high_14 - df['Close']) / (high_14 - low_14 + 0.001) * -100).shift(1)

    # ========== MOVING AVERAGES ==========
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ema_9'] = df['Close'].ewm(span=9).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()

    df['price_vs_sma5'] = ((df['Close'].shift(1) - df['sma_5'].shift(1)) / df['sma_5'].shift(1)) * 100
    df['price_vs_sma10'] = ((df['Close'].shift(1) - df['sma_10'].shift(1)) / df['sma_10'].shift(1)) * 100
    df['price_vs_sma20'] = ((df['Close'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['price_vs_sma50'] = ((df['Close'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['price_vs_ema9'] = ((df['Close'].shift(1) - df['ema_9'].shift(1)) / df['ema_9'].shift(1)) * 100
    df['price_vs_ema21'] = ((df['Close'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100

    df['sma5_vs_sma20'] = ((df['sma_5'].shift(1) - df['sma_20'].shift(1)) / df['sma_20'].shift(1)) * 100
    df['sma10_vs_sma50'] = ((df['sma_10'].shift(1) - df['sma_50'].shift(1)) / df['sma_50'].shift(1)) * 100
    df['ema9_vs_ema21'] = ((df['ema_9'].shift(1) - df['ema_21'].shift(1)) / df['ema_21'].shift(1)) * 100
    df['ema_cross'] = df['ema9_vs_ema21']  # Alias for improved model

    df['trend_alignment'] = (
        (df['Close'].shift(1) > df['sma_5'].shift(1)).astype(int) +
        (df['Close'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['Close'].shift(1) > df['sma_50'].shift(1)).astype(int) +
        (df['sma_5'].shift(1) > df['sma_20'].shift(1)).astype(int) +
        (df['sma_20'].shift(1) > df['sma_50'].shift(1)).astype(int)
    ) / 5
    df['trend_strength'] = df['trend_alignment']  # Alias for improved model

    # ========== BOLLINGER BANDS ==========
    df['bb_middle'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100).shift(1)
    df['bb_position'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)).shift(1)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int).shift(1)

    # ========== MACD ==========
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['prev_macd_hist'] = df['macd_histogram'].shift(1)
    df['macd_hist'] = df['prev_macd_hist']  # Alias for improved model
    df['macd_crossover'] = (
        (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int) -
        (df['macd'].shift(2) > df['macd_signal'].shift(2)).astype(int)
    )
    # MACD cross for quant model (sign change in histogram)
    df['macd_cross'] = (np.sign(df['macd_histogram'].shift(1)) != np.sign(df['macd_histogram'].shift(2))).astype(int)
    df['macd_divergence'] = df['macd_histogram'].shift(1) - df['macd_histogram'].shift(2)
    df['macd_hist_change'] = df['macd_divergence']  # Alias for improved model

    # ========== ADX (Trend Strength) ==========
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean().shift(1)
    df['di_diff'] = (plus_di - minus_di).shift(1)

    # ========== VOLUME ==========
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['prev_volume_ratio'] = (df['Volume'] / df['volume_sma_20']).shift(1)
    df['volume_ratio'] = df['prev_volume_ratio']  # Alias for improved model
    df['volume_trend'] = (df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()).shift(1)
    df['volume_price_trend'] = (df['daily_return'] * df['prev_volume_ratio']).shift(1)
    df['vol_price_corr'] = df['daily_return'].rolling(10).corr(df['Volume'].pct_change()).shift(1)  # For improved model
    df['pv_corr'] = df['vol_price_corr']  # Alias for quant model

    # On-Balance Volume trend
    obv = (np.sign(df['Close'].pct_change()) * df['Volume']).cumsum()
    df['obv_trend'] = (obv / obv.rolling(20).mean() - 1).shift(1)

    # ========== GAP ANALYSIS ==========
    gap_raw = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['gap_size'] = abs(gap_raw)
    df['gap_direction'] = np.sign(gap_raw)
    df['large_gap'] = (df['gap_size'] > df['gap_size'].rolling(50).quantile(0.9)).astype(int)

    # ========== CALENDAR ==========
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['day_of_month'] = df.index.day
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 27).astype(int)

    return df


def calculate_highlow_features(df):
    """Calculate features for high/low prediction model"""
    # Previous day metrics
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['prev_open'] = df['Open'].shift(1)

    # Gap from previous close
    df['gap_pct'] = ((df['Open'] - df['prev_close']) / df['prev_close']) * 100

    # Previous day's range
    df['prev_range_pct'] = ((df['prev_high'] - df['prev_low']) / df['prev_close']) * 100
    df['prev_high_pct'] = ((df['prev_high'] - df['prev_open']) / df['prev_open']) * 100
    df['prev_low_pct'] = ((df['prev_open'] - df['prev_low']) / df['prev_open']) * 100
    df['prev_close_pct'] = ((df['prev_close'] - df['prev_open']) / df['prev_open']) * 100

    # Actual high/low from open
    df['actual_high_pct'] = ((df['High'] - df['Open']) / df['Open']) * 100
    df['actual_low_pct'] = ((df['Open'] - df['Low']) / df['Open']) * 100

    # Returns
    df['prev_return'] = df['Close'].pct_change().shift(1) * 100
    df['prev_2_return'] = df['Close'].pct_change().shift(2) * 100
    df['prev_3_return'] = df['Close'].pct_change().shift(3) * 100

    # Momentum
    df['momentum_3d'] = df['prev_return'].rolling(3).sum()
    df['momentum_5d'] = df['prev_return'].rolling(5).sum()

    # Volatility
    df['volatility_5d'] = df['prev_return'].rolling(5).std()
    df['volatility_10d'] = df['prev_return'].rolling(10).std()
    df['volatility_20d'] = df['prev_return'].rolling(20).std()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = (tr.rolling(5).mean().shift(1) / df['prev_close']) * 100
    df['atr_10'] = (tr.rolling(10).mean().shift(1) / df['prev_close']) * 100
    df['atr_14'] = (tr.rolling(14).mean().shift(1) / df['prev_close']) * 100

    # Historical patterns
    df['avg_high_5d'] = df['actual_high_pct'].rolling(5).mean().shift(1)
    df['avg_low_5d'] = df['actual_low_pct'].rolling(5).mean().shift(1)
    df['avg_high_10d'] = df['actual_high_pct'].rolling(10).mean().shift(1)
    df['avg_low_10d'] = df['actual_low_pct'].rolling(10).mean().shift(1)
    df['max_high_5d'] = df['actual_high_pct'].rolling(5).max().shift(1)
    df['max_low_5d'] = df['actual_low_pct'].rolling(5).max().shift(1)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = (ema_12 - ema_26).shift(1)
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Price vs SMAs
    sma_20 = df['Close'].rolling(20).mean()
    sma_50 = df['Close'].rolling(50).mean()
    df['price_vs_sma20'] = ((df['prev_close'] - sma_20.shift(1)) / sma_20.shift(1)) * 100
    df['price_vs_sma50'] = ((df['prev_close'] - sma_50.shift(1)) / sma_50.shift(1)) * 100

    # Day of week
    df['day_of_week'] = df.index.dayofweek

    # Volume
    df['volume_ratio'] = df['Volume'].shift(1) / df['Volume'].rolling(20).mean().shift(1)

    # Consecutive
    df['up_day'] = (df['Close'] > df['Open']).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum().shift(1)

    return df


def predict_highlow(ticker: str, df: pd.DataFrame):
    """Predict high/low range using ML ensemble model"""
    if ticker not in highlow_models:
        return None

    model_data = highlow_models[ticker]
    feature_cols = model_data['feature_cols']
    buffer = model_data.get('buffer', 0)
    weights = model_data.get('weights', {'xgb': 0.4, 'gb': 0.3, 'rf': 0.3})

    # Calculate features
    df = calculate_highlow_features(df)
    latest = df.iloc[-1]

    # Build feature vector
    features = {col: latest[col] if col in latest and not pd.isna(latest[col]) else 0 for col in feature_cols}
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = model_data['scaler'].transform(X)

    # Ensemble prediction for high
    high_models = model_data['high_models']
    pred_high_pct = (
        high_models['xgb'].predict(X_scaled)[0] * weights['xgb'] +
        high_models['gb'].predict(X_scaled)[0] * weights['gb'] +
        high_models['rf'].predict(X_scaled)[0] * weights['rf']
    ) + buffer

    # Ensemble prediction for low
    low_models = model_data['low_models']
    pred_low_pct = (
        low_models['xgb'].predict(X_scaled)[0] * weights['xgb'] +
        low_models['gb'].predict(X_scaled)[0] * weights['gb'] +
        low_models['rf'].predict(X_scaled)[0] * weights['rf']
    ) + buffer

    # Convert to prices (from today's open)
    open_price = float(latest['Open'])
    pred_high = open_price * (1 + pred_high_pct / 100)
    pred_low = open_price * (1 - pred_low_pct / 100)

    return {
        'wide_range': {
            'high': round(pred_high, 2),
            'low': round(pred_low, 2),
        },
        'high_pct': round(pred_high_pct, 3),
        'low_pct': round(pred_low_pct, 3),
        'open_price': round(open_price, 2),
        'capture_rate': model_data['metrics'].get('capture_rate', 0),
        'model_version': model_data.get('version', 'highlow_v1'),
    }


def predict_shrinking_range(ticker: str, current_price: float, today_open: float,
                             today_high: float, today_low: float, prev_range_pct: float,
                             prev_return: float, gap_pct: float, volatility_5d: float):
    """
    Predict shrinking range using trained ML model.

    This model predicts:
    - Remaining upside: how much higher can price go?
    - Remaining downside: how much lower can price go?

    Trained on ~400 days of simulated time-slice data.
    Accuracy: 91-93% capture rate for EOD close.
    """
    import pytz

    if ticker not in shrinking_models:
        return None

    model_data = shrinking_models[ticker]
    feature_cols = model_data['feature_cols']
    buffer = model_data.get('buffer', 0)

    # Calculate time remaining
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if now_et < market_open:
        hours_elapsed = 0
        time_remaining = 1.0
    elif now_et > market_close:
        hours_elapsed = 6.5
        time_remaining = 0.05
    else:
        hours_elapsed = (now_et - market_open).total_seconds() / 3600
        time_remaining = max(0.05, 1 - (hours_elapsed / 6.5))

    # Calculate current state features
    current_vs_open_pct = ((current_price - today_open) / today_open) * 100
    high_so_far_pct = ((today_high - today_open) / today_open) * 100
    low_so_far_pct = ((today_open - today_low) / today_open) * 100
    range_so_far_pct = high_so_far_pct + low_so_far_pct

    # Build feature vector
    features = {
        'time_remaining': time_remaining,
        'hours_elapsed': hours_elapsed,
        'current_vs_open_pct': current_vs_open_pct,
        'high_so_far_pct': high_so_far_pct,
        'low_so_far_pct': low_so_far_pct,
        'range_so_far_pct': range_so_far_pct,
        'gap_pct': gap_pct,
        'prev_range_pct': prev_range_pct,
        'prev_return': prev_return,
        'volatility_5d': volatility_5d,
    }

    X = pd.DataFrame([features])[feature_cols]
    X_scaled = model_data['scaler'].transform(X)

    # Predict remaining upside/downside
    remaining_up = model_data['up_model'].predict(X_scaled)[0] + buffer
    remaining_down = model_data['down_model'].predict(X_scaled)[0] + buffer

    # Convert to prices
    shrink_high = current_price * (1 + remaining_up / 100)
    shrink_low = current_price * (1 - remaining_down / 100)

    return {
        'high': round(shrink_high, 2),
        'low': round(shrink_low, 2),
        'time_remaining_pct': round(time_remaining * 100, 1),
        'capture_rate': model_data['metrics'].get('capture_rate', 0),
    }


@app.route('/daily_prediction', methods=['GET'])
def daily_prediction():
    """Get daily direction prediction for a ticker"""
    ticker = request.args.get('ticker', 'SPY').upper()

    if ticker not in daily_models:
        return jsonify({'error': f'No daily model for {ticker}'}), 404

    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    try:
        # Fetch recent data from Polygon
        df = fetch_polygon_data(ticker, days=100)

        if len(df) < 50:
            return jsonify({'error': 'Insufficient historical data'}), 500

        # Calculate features
        df = calculate_daily_features(df)
        latest = df.iloc[-1]

        model_data = daily_models[ticker]
        feature_cols = model_data['feature_cols']

        # New improved models have 'all_feature_cols' for scaling
        all_feature_cols = model_data.get('all_feature_cols', feature_cols)

        # Build feature vector using all features for scaling
        features = {col: latest[col] if col in latest and not pd.isna(latest[col]) else 0 for col in all_feature_cols}
        X_all = pd.DataFrame([features])[all_feature_cols]
        X_scaled_all = model_data['scaler'].transform(X_all)

        # Extract selected features after scaling
        if len(feature_cols) != len(all_feature_cols):
            selected_idx = [all_feature_cols.index(f) for f in feature_cols]
            X_scaled = X_scaled_all[:, selected_idx]
        else:
            X_scaled = X_scaled_all

        # Get predictions - handle both old and new model structures
        weights = model_data['weights']
        models = model_data['models']

        bullish_prob = 0.0
        for model_name, weight in weights.items():
            if model_name in models:
                prob = models[model_name].predict_proba(X_scaled)[0][1]
                bullish_prob += prob * weight

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


def is_market_hours():
    """Check if market is currently open (9:30 AM - 4:00 PM ET)"""
    import pytz
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    # Check if weekend
    if now_et.weekday() >= 5:
        return False

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close


def get_next_trading_day():
    """Get the next trading day (skips weekends)"""
    import pytz
    from datetime import timedelta

    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    # If before 4pm on weekday, today is the trading day
    if now_et.weekday() < 5:
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if now_et < market_close:
            return now_et.strftime('%A, %b %d')

    # Find next weekday
    next_day = now_et + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day.strftime('%A, %b %d')


@app.route('/daily_signals', methods=['GET'])
def daily_signals():
    """
    Get clear BUY/SELL/HOLD signals for each ticker.

    This is a simplified endpoint that provides actionable trading signals:
    - STRONG BUY: High confidence bullish (>=70% probability)
    - BUY: Moderate bullish (60-70% probability)
    - HOLD: Neutral/uncertain (40-60% probability)
    - SELL: Moderate bearish (30-40% probability)
    - STRONG SELL: High confidence bearish (<=30% probability)

    Each signal includes:
    - Clear action (BUY/SELL/HOLD)
    - Strength (STRONG/MODERATE/WEAK)
    - Target prices (high/low predictions)
    - Stop loss suggestion
    - High/Low model predictions (for after-hours analysis)
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    # Determine if after hours
    after_hours = not is_market_hours()
    next_trading_day = get_next_trading_day()

    signals = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat(),
        'is_after_hours': after_hours,
        'next_trading_day': next_trading_day,
        'tickers': {},
        'summary': {
            'buys': [],
            'sells': [],
            'holds': [],
        }
    }

    for ticker in SUPPORTED_TICKERS:
        if ticker not in daily_models:
            signals['tickers'][ticker] = {'error': 'No model available'}
            continue

        try:
            # Fetch data
            df = fetch_polygon_data(ticker, days=100)
            if len(df) < 50:
                signals['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            df = calculate_daily_features(df)
            latest = df.iloc[-1]

            model_data = daily_models[ticker]
            feature_cols = model_data['feature_cols']  # Selected features for prediction

            # New improved models have 'all_feature_cols' for scaling
            all_feature_cols = model_data.get('all_feature_cols', feature_cols)

            # Build features for scaling (use all_feature_cols if available)
            scale_features = {col: latest[col] if col in latest and not pd.isna(latest[col]) else 0 for col in all_feature_cols}
            X_all = pd.DataFrame([scale_features])[all_feature_cols]
            X_scaled_all = model_data['scaler'].transform(X_all)

            # If we have selected features, extract only those columns after scaling
            if len(feature_cols) != len(all_feature_cols):
                selected_idx = [all_feature_cols.index(f) for f in feature_cols]
                X_scaled = X_scaled_all[:, selected_idx]
            else:
                X_scaled = X_scaled_all

            # Get ensemble probability - handle both old (rf,gb,lr) and new (lgbm,xgb,gb,rf) models
            weights = model_data['weights']
            models = model_data['models']

            bullish_prob = 0.0
            for model_name, weight in weights.items():
                if model_name in models:
                    prob = models[model_name].predict_proba(X_scaled)[0][1]
                    bullish_prob += prob * weight

            # Convert probability to clear signal
            if bullish_prob >= 0.70:
                signal = 'BUY'
                strength = 'STRONG'
                action_emoji = '🟢🟢'
            elif bullish_prob >= 0.60:
                signal = 'BUY'
                strength = 'MODERATE'
                action_emoji = '🟢'
            elif bullish_prob >= 0.55:
                signal = 'BUY'
                strength = 'WEAK'
                action_emoji = '🟢'
            elif bullish_prob <= 0.30:
                signal = 'SELL'
                strength = 'STRONG'
                action_emoji = '🔴🔴'
            elif bullish_prob <= 0.40:
                signal = 'SELL'
                strength = 'MODERATE'
                action_emoji = '🔴'
            elif bullish_prob <= 0.45:
                signal = 'SELL'
                strength = 'WEAK'
                action_emoji = '🔴'
            else:
                signal = 'HOLD'
                strength = 'NEUTRAL'
                action_emoji = '🟡'

            # Get price data
            snapshot = fetch_intraday_snapshot(ticker)
            if snapshot:
                current_price = float(snapshot['current_price'])
                today_open = float(snapshot['today_open'])
            else:
                current_price = float(latest['Close'])
                today_open = float(latest['Open'])

            # Get high/low predictions for targets
            highlow_pred = predict_highlow(ticker, df.copy())

            if highlow_pred:
                target_high = highlow_pred['wide_range']['high']
                target_low = highlow_pred['wide_range']['low']
            else:
                # Fallback to ATR-based
                atr = float(latest['atr_14']) if not pd.isna(latest.get('atr_14', None)) else current_price * 0.01
                target_high = round(current_price + atr, 2)
                target_low = round(current_price - atr, 2)

            # Calculate risk/reward
            if signal == 'BUY':
                entry = current_price
                target = target_high
                stop_loss = target_low
                risk = entry - stop_loss
                reward = target - entry
            elif signal == 'SELL':
                entry = current_price
                target = target_low
                stop_loss = target_high
                risk = stop_loss - entry
                reward = entry - target
            else:
                entry = current_price
                target = current_price
                stop_loss = current_price
                risk = 0
                reward = 0

            risk_reward_ratio = round(reward / risk, 2) if risk > 0 else 0

            # Build highlow_model data for after-hours analysis
            highlow_model_data = None
            if highlow_pred:
                highlow_model_data = {
                    'predicted_high': highlow_pred['wide_range']['high'],
                    'predicted_low': highlow_pred['wide_range']['low'],
                    'high_pct': highlow_pred['high_pct'],
                    'low_pct': highlow_pred['low_pct'],
                    'capture_rate': highlow_pred['capture_rate'],
                }

            # Get intraday session-updated prediction (during market hours)
            intraday_pred = None
            intraday_model_data = None
            display_probability = bullish_prob  # Default to daily model
            display_accuracy = float(model_data['metrics']['accuracy'])

            if not after_hours and snapshot:
                intraday_pred = predict_intraday(ticker, df, snapshot)
                if intraday_pred and intraday_pred['time_pct'] > 0.05:
                    # Use intraday prediction if we're past the first few minutes
                    display_probability = intraday_pred['probability']
                    display_accuracy = intraday_pred['model_accuracy']

                    # Update signal based on intraday prediction
                    if display_probability >= 0.70:
                        signal = 'BUY'
                        strength = 'STRONG'
                        action_emoji = '🟢🟢'
                    elif display_probability >= 0.60:
                        signal = 'BUY'
                        strength = 'MODERATE'
                        action_emoji = '🟢'
                    elif display_probability >= 0.55:
                        signal = 'BUY'
                        strength = 'WEAK'
                        action_emoji = '🟢'
                    elif display_probability <= 0.30:
                        signal = 'SELL'
                        strength = 'STRONG'
                        action_emoji = '🔴🔴'
                    elif display_probability <= 0.40:
                        signal = 'SELL'
                        strength = 'MODERATE'
                        action_emoji = '🔴'
                    elif display_probability <= 0.45:
                        signal = 'SELL'
                        strength = 'WEAK'
                        action_emoji = '🔴'
                    else:
                        signal = 'HOLD'
                        strength = 'NEUTRAL'
                        action_emoji = '🟡'

                    # Recalculate targets based on updated signal
                    if signal == 'BUY':
                        target = target_high
                        stop_loss = target_low
                    elif signal == 'SELL':
                        target = target_low
                        stop_loss = target_high
                    else:
                        target = current_price
                        stop_loss = current_price

                    intraday_model_data = {
                        'probability': intraday_pred['probability'],
                        'confidence': intraday_pred['confidence'],
                        'time_pct': intraday_pred['time_pct'],
                        'session_label': intraday_pred['session_label'],
                        'current_vs_open': intraday_pred['current_vs_open'],
                        'position_in_range': intraday_pred['position_in_range'],
                        'model_accuracy': intraday_pred['model_accuracy'],
                    }

            ticker_signal = {
                'signal': signal,
                'strength': strength,
                'action': f'{strength} {signal}',
                'emoji': action_emoji,
                'probability': round(float(display_probability), 3),
                'confidence': round(abs(display_probability - 0.5) * 2, 3),
                'current_price': round(current_price, 2),
                'entry_price': round(entry, 2),
                'target_price': round(target, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward': risk_reward_ratio,
                'predicted_range': {
                    'high': target_high,
                    'low': target_low,
                },
                'highlow_model': highlow_model_data,
                'intraday_model': intraday_model_data,
                'daily_probability': round(float(bullish_prob), 3),  # Original daily prediction
                'model_accuracy': round(float(display_accuracy), 3),
                'prediction_source': 'intraday' if intraday_model_data else 'daily',
            }

            signals['tickers'][ticker] = ticker_signal

            # Add to summary
            if signal == 'BUY':
                signals['summary']['buys'].append({
                    'ticker': ticker,
                    'strength': strength,
                    'probability': round(float(bullish_prob), 3),
                })
            elif signal == 'SELL':
                signals['summary']['sells'].append({
                    'ticker': ticker,
                    'strength': strength,
                    'probability': round(float(1 - bullish_prob), 3),
                })
            else:
                signals['summary']['holds'].append({
                    'ticker': ticker,
                    'probability': round(float(bullish_prob), 3),
                })

        except Exception as e:
            signals['tickers'][ticker] = {'error': str(e)}

    # Determine overall market signal
    buy_count = len(signals['summary']['buys'])
    sell_count = len(signals['summary']['sells'])

    if buy_count > sell_count and buy_count >= 2:
        signals['market_signal'] = 'BULLISH'
        signals['market_emoji'] = '🟢'
        signals['market_action'] = 'FAVOR LONGS'
    elif sell_count > buy_count and sell_count >= 2:
        signals['market_signal'] = 'BEARISH'
        signals['market_emoji'] = '🔴'
        signals['market_action'] = 'FAVOR SHORTS'
    else:
        signals['market_signal'] = 'MIXED'
        signals['market_emoji'] = '🟡'
        signals['market_action'] = 'BE SELECTIVE'

    # Find best opportunity
    all_signals = []
    for ticker, data in signals['tickers'].items():
        if 'error' not in data:
            all_signals.append({
                'ticker': ticker,
                'signal': data['signal'],
                'strength': data['strength'],
                'confidence': data['confidence'],
                'risk_reward': data['risk_reward'],
            })

    # Sort by confidence then risk/reward
    all_signals.sort(key=lambda x: (x['confidence'], x['risk_reward']), reverse=True)

    if all_signals and all_signals[0]['signal'] != 'HOLD':
        signals['best_trade'] = all_signals[0]
    else:
        signals['best_trade'] = None

    return jsonify(signals)


@app.route('/signal_breakdown', methods=['GET'])
def signal_breakdown():
    """
    Get detailed technical indicator breakdown for each ticker.
    Shows exactly which indicators are bullish, bearish, or neutral.
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat(),
        'tickers': {}
    }

    for ticker in SUPPORTED_TICKERS:
        try:
            # Fetch data
            df = fetch_polygon_data(ticker, days=100)
            if len(df) < 50:
                result['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            df = calculate_daily_features(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            current_price = float(latest['Close'])

            # Build technical breakdown
            indicators = []

            # === RSI ===
            rsi = float(latest['rsi_14']) if not pd.isna(latest.get('rsi_14')) else 50
            if rsi > 70:
                indicators.append({'name': 'RSI (14)', 'value': f'{rsi:.1f}', 'signal': 'BEARISH', 'reason': 'Overbought (>70)'})
            elif rsi > 60:
                indicators.append({'name': 'RSI (14)', 'value': f'{rsi:.1f}', 'signal': 'BEARISH', 'reason': 'Elevated (>60)'})
            elif rsi < 30:
                indicators.append({'name': 'RSI (14)', 'value': f'{rsi:.1f}', 'signal': 'BULLISH', 'reason': 'Oversold (<30)'})
            elif rsi < 40:
                indicators.append({'name': 'RSI (14)', 'value': f'{rsi:.1f}', 'signal': 'BULLISH', 'reason': 'Low (<40)'})
            else:
                indicators.append({'name': 'RSI (14)', 'value': f'{rsi:.1f}', 'signal': 'NEUTRAL', 'reason': 'Neutral zone'})

            # === MACD ===
            macd_hist = float(latest['macd_histogram']) if not pd.isna(latest.get('macd_histogram')) else 0
            prev_macd_hist = float(latest['prev_macd_hist']) if not pd.isna(latest.get('prev_macd_hist')) else 0
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                indicators.append({'name': 'MACD Histogram', 'value': f'{macd_hist:.3f}', 'signal': 'BULLISH', 'reason': 'Positive & rising'})
            elif macd_hist > 0 and macd_hist < prev_macd_hist:
                indicators.append({'name': 'MACD Histogram', 'value': f'{macd_hist:.3f}', 'signal': 'BEARISH', 'reason': 'Positive but falling'})
            elif macd_hist < 0 and macd_hist > prev_macd_hist:
                indicators.append({'name': 'MACD Histogram', 'value': f'{macd_hist:.3f}', 'signal': 'BULLISH', 'reason': 'Negative but rising'})
            elif macd_hist < 0:
                indicators.append({'name': 'MACD Histogram', 'value': f'{macd_hist:.3f}', 'signal': 'BEARISH', 'reason': 'Negative & falling'})
            else:
                indicators.append({'name': 'MACD Histogram', 'value': f'{macd_hist:.3f}', 'signal': 'NEUTRAL', 'reason': 'Flat'})

            # === Price vs SMA 20 ===
            price_vs_sma20 = float(latest['price_vs_sma20']) if not pd.isna(latest.get('price_vs_sma20')) else 0
            if price_vs_sma20 > 2:
                indicators.append({'name': 'Price vs SMA20', 'value': f'{price_vs_sma20:+.2f}%', 'signal': 'BEARISH', 'reason': 'Extended above (>2%)'})
            elif price_vs_sma20 > 0.5:
                indicators.append({'name': 'Price vs SMA20', 'value': f'{price_vs_sma20:+.2f}%', 'signal': 'BULLISH', 'reason': 'Above SMA20'})
            elif price_vs_sma20 < -2:
                indicators.append({'name': 'Price vs SMA20', 'value': f'{price_vs_sma20:+.2f}%', 'signal': 'BULLISH', 'reason': 'Extended below (<-2%)'})
            elif price_vs_sma20 < -0.5:
                indicators.append({'name': 'Price vs SMA20', 'value': f'{price_vs_sma20:+.2f}%', 'signal': 'BEARISH', 'reason': 'Below SMA20'})
            else:
                indicators.append({'name': 'Price vs SMA20', 'value': f'{price_vs_sma20:+.2f}%', 'signal': 'NEUTRAL', 'reason': 'Near SMA20'})

            # === Price vs SMA 50 ===
            price_vs_sma50 = float(latest['price_vs_sma50']) if not pd.isna(latest.get('price_vs_sma50')) else 0
            if price_vs_sma50 > 4:
                indicators.append({'name': 'Price vs SMA50', 'value': f'{price_vs_sma50:+.2f}%', 'signal': 'BEARISH', 'reason': 'Extended above (>4%)'})
            elif price_vs_sma50 > 0:
                indicators.append({'name': 'Price vs SMA50', 'value': f'{price_vs_sma50:+.2f}%', 'signal': 'BULLISH', 'reason': 'Above SMA50'})
            elif price_vs_sma50 < -4:
                indicators.append({'name': 'Price vs SMA50', 'value': f'{price_vs_sma50:+.2f}%', 'signal': 'BULLISH', 'reason': 'Extended below (<-4%)'})
            else:
                indicators.append({'name': 'Price vs SMA50', 'value': f'{price_vs_sma50:+.2f}%', 'signal': 'BEARISH', 'reason': 'Below SMA50'})

            # === Momentum 5D ===
            momentum_5d = float(latest['momentum_5d']) if not pd.isna(latest.get('momentum_5d')) else 0
            if momentum_5d > 3:
                indicators.append({'name': '5-Day Momentum', 'value': f'{momentum_5d:+.2f}%', 'signal': 'BEARISH', 'reason': 'Overextended (>3%)'})
            elif momentum_5d > 1:
                indicators.append({'name': '5-Day Momentum', 'value': f'{momentum_5d:+.2f}%', 'signal': 'BULLISH', 'reason': 'Positive momentum'})
            elif momentum_5d < -3:
                indicators.append({'name': '5-Day Momentum', 'value': f'{momentum_5d:+.2f}%', 'signal': 'BULLISH', 'reason': 'Oversold bounce likely'})
            elif momentum_5d < -1:
                indicators.append({'name': '5-Day Momentum', 'value': f'{momentum_5d:+.2f}%', 'signal': 'BEARISH', 'reason': 'Negative momentum'})
            else:
                indicators.append({'name': '5-Day Momentum', 'value': f'{momentum_5d:+.2f}%', 'signal': 'NEUTRAL', 'reason': 'Flat'})

            # === Consecutive Days ===
            consec_up = int(latest['prev_consec_up']) if not pd.isna(latest.get('prev_consec_up')) else 0
            consec_down = int(latest['prev_consec_down']) if not pd.isna(latest.get('prev_consec_down')) else 0
            if consec_up >= 4:
                indicators.append({'name': 'Consecutive Days', 'value': f'{consec_up} up', 'signal': 'BEARISH', 'reason': f'{consec_up} up days - reversal likely'})
            elif consec_up >= 2:
                indicators.append({'name': 'Consecutive Days', 'value': f'{consec_up} up', 'signal': 'BEARISH', 'reason': f'{consec_up} up days - pullback possible'})
            elif consec_down >= 4:
                indicators.append({'name': 'Consecutive Days', 'value': f'{consec_down} down', 'signal': 'BULLISH', 'reason': f'{consec_down} down days - bounce likely'})
            elif consec_down >= 2:
                indicators.append({'name': 'Consecutive Days', 'value': f'{consec_down} down', 'signal': 'BULLISH', 'reason': f'{consec_down} down days - bounce possible'})
            else:
                indicators.append({'name': 'Consecutive Days', 'value': 'Mixed', 'signal': 'NEUTRAL', 'reason': 'No streak'})

            # === Stochastic ===
            stoch_k = float(latest['stoch_k']) if not pd.isna(latest.get('stoch_k')) else 50
            if stoch_k > 80:
                indicators.append({'name': 'Stochastic %K', 'value': f'{stoch_k:.1f}', 'signal': 'BEARISH', 'reason': 'Overbought (>80)'})
            elif stoch_k < 20:
                indicators.append({'name': 'Stochastic %K', 'value': f'{stoch_k:.1f}', 'signal': 'BULLISH', 'reason': 'Oversold (<20)'})
            else:
                indicators.append({'name': 'Stochastic %K', 'value': f'{stoch_k:.1f}', 'signal': 'NEUTRAL', 'reason': 'Middle range'})

            # === Williams %R ===
            williams_r = float(latest['williams_r']) if not pd.isna(latest.get('williams_r')) else -50
            if williams_r > -20:
                indicators.append({'name': 'Williams %R', 'value': f'{williams_r:.1f}', 'signal': 'BEARISH', 'reason': 'Overbought (>-20)'})
            elif williams_r < -80:
                indicators.append({'name': 'Williams %R', 'value': f'{williams_r:.1f}', 'signal': 'BULLISH', 'reason': 'Oversold (<-80)'})
            else:
                indicators.append({'name': 'Williams %R', 'value': f'{williams_r:.1f}', 'signal': 'NEUTRAL', 'reason': 'Middle range'})

            # === Bollinger Band Position ===
            bb_position = float(latest['bb_position']) if not pd.isna(latest.get('bb_position')) else 0.5
            if bb_position > 0.9:
                indicators.append({'name': 'BB Position', 'value': f'{bb_position*100:.0f}%', 'signal': 'BEARISH', 'reason': 'Near upper band'})
            elif bb_position < 0.1:
                indicators.append({'name': 'BB Position', 'value': f'{bb_position*100:.0f}%', 'signal': 'BULLISH', 'reason': 'Near lower band'})
            elif bb_position > 0.7:
                indicators.append({'name': 'BB Position', 'value': f'{bb_position*100:.0f}%', 'signal': 'BEARISH', 'reason': 'Upper half of bands'})
            elif bb_position < 0.3:
                indicators.append({'name': 'BB Position', 'value': f'{bb_position*100:.0f}%', 'signal': 'BULLISH', 'reason': 'Lower half of bands'})
            else:
                indicators.append({'name': 'BB Position', 'value': f'{bb_position*100:.0f}%', 'signal': 'NEUTRAL', 'reason': 'Middle of bands'})

            # === ADX (Trend Strength) ===
            adx = float(latest['adx']) if not pd.isna(latest.get('adx')) else 20
            di_diff = float(latest['di_diff']) if not pd.isna(latest.get('di_diff')) else 0
            if adx > 25:
                if di_diff > 0:
                    indicators.append({'name': 'ADX Trend', 'value': f'{adx:.1f}', 'signal': 'BULLISH', 'reason': f'Strong uptrend (ADX>{25})'})
                else:
                    indicators.append({'name': 'ADX Trend', 'value': f'{adx:.1f}', 'signal': 'BEARISH', 'reason': f'Strong downtrend (ADX>{25})'})
            else:
                indicators.append({'name': 'ADX Trend', 'value': f'{adx:.1f}', 'signal': 'NEUTRAL', 'reason': 'Weak trend'})

            # === Volume ===
            volume_ratio = float(latest['prev_volume_ratio']) if not pd.isna(latest.get('prev_volume_ratio')) else 1
            prev_return = float(latest['prev_return']) if not pd.isna(latest.get('prev_return')) else 0
            if volume_ratio > 1.5 and prev_return > 0:
                indicators.append({'name': 'Volume', 'value': f'{volume_ratio:.2f}x avg', 'signal': 'BULLISH', 'reason': 'High volume on up day'})
            elif volume_ratio > 1.5 and prev_return < 0:
                indicators.append({'name': 'Volume', 'value': f'{volume_ratio:.2f}x avg', 'signal': 'BEARISH', 'reason': 'High volume on down day'})
            elif volume_ratio < 0.7:
                indicators.append({'name': 'Volume', 'value': f'{volume_ratio:.2f}x avg', 'signal': 'NEUTRAL', 'reason': 'Low volume - weak conviction'})
            else:
                indicators.append({'name': 'Volume', 'value': f'{volume_ratio:.2f}x avg', 'signal': 'NEUTRAL', 'reason': 'Average volume'})

            # === Previous Day Return ===
            if prev_return > 1.5:
                indicators.append({'name': 'Prev Day Return', 'value': f'{prev_return:+.2f}%', 'signal': 'BEARISH', 'reason': 'Large up day - pullback likely'})
            elif prev_return > 0.5:
                indicators.append({'name': 'Prev Day Return', 'value': f'{prev_return:+.2f}%', 'signal': 'BULLISH', 'reason': 'Positive momentum'})
            elif prev_return < -1.5:
                indicators.append({'name': 'Prev Day Return', 'value': f'{prev_return:+.2f}%', 'signal': 'BULLISH', 'reason': 'Large down day - bounce likely'})
            elif prev_return < -0.5:
                indicators.append({'name': 'Prev Day Return', 'value': f'{prev_return:+.2f}%', 'signal': 'BEARISH', 'reason': 'Negative momentum'})
            else:
                indicators.append({'name': 'Prev Day Return', 'value': f'{prev_return:+.2f}%', 'signal': 'NEUTRAL', 'reason': 'Small move'})

            # Count signals
            bullish_count = sum(1 for i in indicators if i['signal'] == 'BULLISH')
            bearish_count = sum(1 for i in indicators if i['signal'] == 'BEARISH')
            neutral_count = sum(1 for i in indicators if i['signal'] == 'NEUTRAL')

            result['tickers'][ticker] = {
                'current_price': round(current_price, 2),
                'indicators': indicators,
                'summary': {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count,
                    'total': len(indicators)
                }
            }

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    return jsonify(result)


@app.route('/morning_briefing', methods=['GET'])
def morning_briefing():
    """Get morning briefing for all tickers"""
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

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
            # Fetch data from Polygon
            df = fetch_polygon_data(ticker, days=100)

            if len(df) < 50:
                briefing['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            df = calculate_daily_features(df)
            latest = df.iloc[-1]

            model_data = daily_models[ticker]
            feature_cols = model_data['feature_cols']

            # New improved models have 'all_feature_cols' for scaling
            all_feature_cols = model_data.get('all_feature_cols', feature_cols)

            # Build features for scaling
            features = {col: latest[col] if col in latest and not pd.isna(latest[col]) else 0 for col in all_feature_cols}
            X_all = pd.DataFrame([features])[all_feature_cols]
            X_scaled_all = model_data['scaler'].transform(X_all)

            # Extract selected features after scaling
            if len(feature_cols) != len(all_feature_cols):
                selected_idx = [all_feature_cols.index(f) for f in feature_cols]
                X_scaled = X_scaled_all[:, selected_idx]
            else:
                X_scaled = X_scaled_all

            weights = model_data['weights']
            models = model_data['models']

            bullish_prob = 0.0
            for model_name, weight in weights.items():
                if model_name in models:
                    prob = models[model_name].predict_proba(X_scaled)[0][1]
                    bullish_prob += prob * weight

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

            # Get real-time snapshot for current price and today's high/low
            snapshot = fetch_intraday_snapshot(ticker)

            if snapshot:
                current_price = float(snapshot['current_price'])
                today_high = float(snapshot['today_high'])
                today_low = float(snapshot['today_low'])
                today_open = float(snapshot['today_open'])
            else:
                current_price = float(latest['Close'])
                today_high = float(latest['High'])
                today_low = float(latest['Low'])
                today_open = float(latest['Open'])

            # Calculate context features for shrinking model
            prev_range_pct = ((float(df.iloc[-2]['High']) - float(df.iloc[-2]['Low'])) / float(df.iloc[-2]['Close'])) * 100
            prev_return = ((float(df.iloc[-1]['Close']) - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close'])) * 100
            gap_pct = ((today_open - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close'])) * 100
            volatility_5d = df['Close'].pct_change().tail(6).std() * 100

            # Get ML high/low prediction (wide range)
            highlow_pred = predict_highlow(ticker, df.copy())

            # Get ML shrinking range prediction
            shrinking_pred = predict_shrinking_range(
                ticker=ticker,
                current_price=current_price,
                today_open=today_open,
                today_high=today_high,
                today_low=today_low,
                prev_range_pct=prev_range_pct,
                prev_return=prev_return,
                gap_pct=gap_pct,
                volatility_5d=volatility_5d
            )

            if highlow_pred:
                wide_range = highlow_pred['wide_range']
                wide_capture_rate = highlow_pred['capture_rate']

                if shrinking_pred:
                    predicted_range = {
                        'wide': wide_range,
                        'wide_capture_rate': wide_capture_rate,
                        'shrinking': {
                            'high': shrinking_pred['high'],
                            'low': shrinking_pred['low'],
                        },
                        'shrinking_capture_rate': shrinking_pred['capture_rate'],
                        'time_remaining_pct': shrinking_pred['time_remaining_pct'],
                        'ml_predicted': True,
                    }
                else:
                    # Shrinking model not available, use wide range
                    predicted_range = {
                        'wide': wide_range,
                        'wide_capture_rate': wide_capture_rate,
                        'shrinking': wide_range,  # Same as wide
                        'shrinking_capture_rate': wide_capture_rate,
                        'time_remaining_pct': 100,
                        'ml_predicted': True,
                    }
            else:
                # Fallback to ATR-based
                atr = float(latest['atr_14']) if not pd.isna(latest['atr_14']) else current_price * 0.01
                predicted_range = {
                    'wide': {
                        'low': round(current_price - atr, 2),
                        'high': round(current_price + atr, 2),
                    },
                    'wide_capture_rate': 0,
                    'shrinking': {
                        'low': round(current_price - atr * 0.5, 2),
                        'high': round(current_price + atr * 0.5, 2),
                    },
                    'shrinking_capture_rate': 0,
                    'time_remaining_pct': 100,
                    'ml_predicted': False,
                }

            briefing['tickers'][ticker] = {
                'direction': direction,
                'emoji': emoji,
                'bullish_probability': round(float(bullish_prob), 3),
                'confidence': round(float(confidence), 3),
                'fvg_recommendation': fvg_rec,
                'current_price': round(current_price, 2),
                'today_high': round(today_high, 2),
                'today_low': round(today_low, 2),
                'today_open': round(today_open, 2),
                'predicted_range': predicted_range,
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


@app.route('/volatility_meter', methods=['GET'])
def volatility_meter():
    """
    Get current volatility regime for each ticker and regime-specific predictions.

    Returns:
    - Current volatility regime (LOW/NORMAL/HIGH)
    - Volatility score (0-1, where 0=very calm, 1=very volatile)
    - Regime-specific model predictions
    - Recommended trading approach based on regime
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat(),
        'tickers': {},
        'market_volatility': 'NORMAL',
        'trading_guidance': ''
    }

    vol_scores = []

    for ticker in SUPPORTED_TICKERS:
        try:
            # Fetch data
            df = fetch_polygon_data(ticker, days=300)
            if len(df) < 100:
                result['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            # Calculate volatility metrics
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean()

            # ATR percentile - use available data or fallback to 0.5 (normal)
            if len(atr_14.dropna()) >= 100:
                atr_values = atr_14.dropna()
                current_atr = atr_values.iloc[-1]
                atr_pct_val = (atr_values < current_atr).sum() / len(atr_values)
            else:
                atr_pct_val = 0.5

            # Return volatility
            daily_return = df['Close'].pct_change() * 100
            vol_20d = daily_return.rolling(20).std()

            if len(vol_20d.dropna()) >= 100:
                vol_values = vol_20d.dropna()
                current_vol = vol_values.iloc[-1]
                vol_pct_val = (vol_values < current_vol).sum() / len(vol_values)
            else:
                vol_pct_val = 0.5

            # Combined score
            vol_score = (atr_pct_val + vol_pct_val) / 2
            vol_scores.append(vol_score)

            # Determine regime
            if vol_score < LOW_VOL_THRESHOLD:
                regime = 'LOW'
                regime_label = 'Low Volatility'
                regime_color = 'green'
            elif vol_score > HIGH_VOL_THRESHOLD:
                regime = 'HIGH'
                regime_label = 'High Volatility'
                regime_color = 'red'
            else:
                regime = 'NORMAL'
                regime_label = 'Normal Volatility'
                regime_color = 'yellow'

            # Get regime-specific prediction if available
            regime_prediction = None
            if ticker in regime_models:
                model_data = regime_models[ticker]
                if regime in model_data['regime_models']:
                    regime_model = model_data['regime_models'][regime]
                    regime_prediction = {
                        'direction_accuracy': round(regime_model['direction']['accuracy'] * 100, 1),
                        'high_conf_accuracy': round(regime_model['direction']['high_conf_accuracy'] * 100, 1),
                        'high_mae': round(regime_model['highlow']['high_mae'], 3),
                        'low_mae': round(regime_model['highlow']['low_mae'], 3),
                    }

            # Current metrics
            current_atr = (atr_14.iloc[-1] / df['Close'].iloc[-1]) * 100
            current_vol = vol_20d.iloc[-1]

            result['tickers'][ticker] = {
                'regime': regime,
                'regime_label': regime_label,
                'regime_color': regime_color,
                'volatility_score': round(vol_score, 3),
                'volatility_percentile': round(vol_score * 100, 1),
                'current_atr_pct': round(current_atr, 3),
                'current_daily_vol': round(current_vol, 3),
                'regime_model_stats': regime_prediction,
                'expected_range': {
                    'description': 'Expected daily range based on regime',
                    'low_vol': '0.2-0.5%',
                    'normal_vol': '0.5-1.0%',
                    'high_vol': '1.0-2.0%+'
                }[regime.lower() + '_vol']
            }

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    # Overall market volatility
    if vol_scores:
        avg_vol = sum(vol_scores) / len(vol_scores)
        if avg_vol < LOW_VOL_THRESHOLD:
            result['market_volatility'] = 'LOW'
            result['trading_guidance'] = 'Low volatility environment. Tighter ranges expected. Good for mean-reversion strategies. High/Low predictions are most accurate.'
        elif avg_vol > HIGH_VOL_THRESHOLD:
            result['market_volatility'] = 'HIGH'
            result['trading_guidance'] = 'High volatility environment. Wider ranges expected. Good for momentum/breakout strategies. Direction predictions are most accurate.'
        else:
            result['market_volatility'] = 'NORMAL'
            result['trading_guidance'] = 'Normal volatility environment. Standard risk management applies. Both direction and range predictions are reliable.'

        result['market_volatility_score'] = round(avg_vol, 3)

    return jsonify(result)


@app.route('/regime_prediction', methods=['GET'])
def regime_prediction():
    """
    Get predictions using volatility regime-specific models.

    Automatically detects current volatility regime and uses
    the appropriate model for more accurate predictions.
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'generated_at': datetime.now().isoformat(),
        'tickers': {}
    }

    for ticker in SUPPORTED_TICKERS:
        if ticker not in regime_models:
            result['tickers'][ticker] = {'error': 'No regime model available'}
            continue

        try:
            # Fetch data
            df = fetch_polygon_data(ticker, days=300)
            if len(df) < 100:
                result['tickers'][ticker] = {'error': 'Insufficient data'}
                continue

            # Calculate volatility for regime detection
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean()

            # ATR percentile - use available data or fallback to 0.5 (normal)
            if len(atr_14.dropna()) >= 100:
                atr_values = atr_14.dropna()
                current_atr = atr_values.iloc[-1]
                atr_pct_val = (atr_values < current_atr).sum() / len(atr_values)
            else:
                atr_pct_val = 0.5

            daily_return = df['Close'].pct_change() * 100
            vol_20d = daily_return.rolling(20).std()

            if len(vol_20d.dropna()) >= 100:
                vol_values = vol_20d.dropna()
                current_vol = vol_values.iloc[-1]
                vol_pct_val = (vol_values < current_vol).sum() / len(vol_values)
            else:
                vol_pct_val = 0.5

            vol_score = (atr_pct_val + vol_pct_val) / 2

            # Determine regime
            if vol_score < LOW_VOL_THRESHOLD:
                regime = 'LOW'
            elif vol_score > HIGH_VOL_THRESHOLD:
                regime = 'HIGH'
            else:
                regime = 'NORMAL'

            # Get model
            model_data = regime_models[ticker]
            if regime not in model_data['regime_models']:
                regime = 'ALL'  # Fallback

            regime_model = model_data['regime_models'][regime]
            feature_cols = model_data['feature_cols']

            # Calculate features for prediction
            df = calculate_daily_features(df)
            latest = df.iloc[-1]

            # Build feature vector
            features = {}
            for col in feature_cols:
                if col in latest:
                    features[col] = latest[col]
                elif col == 'vol_score':
                    features[col] = vol_score
                else:
                    features[col] = 0

            X = pd.DataFrame([features])[feature_cols]

            # Fill any NaN values with 0 (neutral)
            X = X.fillna(0)

            # Direction prediction
            dir_model = regime_model['direction']
            X_scaled = dir_model['scaler'].transform(X)

            weights = dir_model['weights']
            probs = {}
            for name, model in dir_model['models'].items():
                probs[name] = model.predict_proba(X_scaled)[0][1]

            bullish_prob = sum(probs[k] * weights[k] for k in probs.keys())

            # High/Low prediction
            hl_model = regime_model['highlow']
            X_hl_scaled = hl_model['scaler'].transform(X)

            high_pred = (
                hl_model['high_models']['xgb'].predict(X_hl_scaled)[0] * 0.4 +
                hl_model['high_models']['gb'].predict(X_hl_scaled)[0] * 0.3 +
                hl_model['high_models']['rf'].predict(X_hl_scaled)[0] * 0.3
            )

            low_pred = (
                hl_model['low_models']['xgb'].predict(X_hl_scaled)[0] * 0.4 +
                hl_model['low_models']['gb'].predict(X_hl_scaled)[0] * 0.3 +
                hl_model['low_models']['rf'].predict(X_hl_scaled)[0] * 0.3
            )

            # Safety clip predictions to reasonable bounds (0-5% for intraday)
            high_pred = max(0.1, min(5.0, high_pred))
            low_pred = max(0.1, min(5.0, low_pred))

            # Current price
            current_price = df['Close'].iloc[-1]

            # Signal
            if bullish_prob >= 0.65:
                signal = 'BUY'
                strength = 'STRONG' if bullish_prob >= 0.70 else 'MODERATE'
            elif bullish_prob <= 0.35:
                signal = 'SELL'
                strength = 'STRONG' if bullish_prob <= 0.30 else 'MODERATE'
            else:
                signal = 'HOLD'
                strength = 'NEUTRAL'

            result['tickers'][ticker] = {
                'regime': regime,
                'volatility_score': round(vol_score, 3),
                'signal': signal,
                'strength': strength,
                'bullish_probability': round(bullish_prob, 3),
                'predicted_high_pct': round(high_pred, 3),
                'predicted_low_pct': round(low_pred, 3),
                'predicted_high': round(current_price * (1 + high_pred/100), 2),
                'predicted_low': round(current_price * (1 - low_pred/100), 2),
                'current_price': round(current_price, 2),
                'model_accuracy': {
                    'direction': round(regime_model['direction']['accuracy'] * 100, 1),
                    'high_conf': round(regime_model['direction']['high_conf_accuracy'] * 100, 1),
                    'high_mae': round(regime_model['highlow']['high_mae'], 3),
                    'low_mae': round(regime_model['highlow']['low_mae'], 3),
                }
            }

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    return jsonify(result)


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
