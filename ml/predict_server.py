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
from datetime import datetime, timedelta

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
intraday_v6_models = {}  # ticker -> V6 time-split intraday models
target_models = {}  # ticker -> multi-timeframe target refinement models
enhanced_v3_models = {}  # ticker -> enhanced v3 model with 80 new features
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'v6_models')

# Model version selection (can be set via environment variable)
# Options: 'standard', 'enhanced_v3', 'compare' (returns both)
ACTIVE_MODEL_VERSION = os.environ.get('MODEL_VERSION', 'standard')

# Volatility regime thresholds
LOW_VOL_THRESHOLD = 0.30
HIGH_VOL_THRESHOLD = 0.70

# Supported tickers
SUPPORTED_TICKERS = ['SPY', 'QQQ', 'IWM']

# Signal cache to prevent flip-flopping (locks signals for 1 hour)
# Key: "{ticker}_{date}_{hour}" -> {"action": "BUY_CALL", "locked_at": datetime, "data": {...}}
signal_cache = {}
SIGNAL_LOCK_MINUTES = 60  # Lock signals for 1 hour

def get_cached_signal(ticker, hour):
    """Get cached signal if it exists and is still valid"""
    import pytz
    et_tz = pytz.timezone('America/New_York')
    now = datetime.now(et_tz)
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"

    if cache_key in signal_cache:
        cached = signal_cache[cache_key]
        locked_at = cached.get('locked_at')
        if locked_at and (now - locked_at).total_seconds() < SIGNAL_LOCK_MINUTES * 60:
            return cached.get('data')
    return None

def cache_signal(ticker, hour, data):
    """Cache a signal for the given ticker and hour"""
    import pytz
    et_tz = pytz.timezone('America/New_York')
    now = datetime.now(et_tz)
    cache_key = f"{ticker}_{now.strftime('%Y-%m-%d')}_{hour}"
    signal_cache[cache_key] = {
        'locked_at': now,
        'data': data
    }

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

    # Load V6 time-split intraday models (high accuracy)
    print("\nLoading V6 Time-Split Intraday models...")
    for ticker in SUPPORTED_TICKERS:
        v6_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
        if os.path.exists(v6_path):
            try:
                with open(v6_path, 'rb') as f:
                    intraday_v6_models[ticker] = pickle.load(f)
                acc_early = intraday_v6_models[ticker].get('acc_early', 0)
                acc_late_a = intraday_v6_models[ticker].get('acc_late_a', 0)
                acc_late_b = intraday_v6_models[ticker].get('acc_late_b', 0)
                print(f"  ✓ {ticker} V6 model loaded")
                print(f"      Early: {acc_early:.1%}, Late A: {acc_late_a:.1%}, Late B: {acc_late_b:.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} V6 model failed to load: {e}")
        else:
            print(f"  - {ticker} V6 model not found (run train_time_split.py)")

    # Load target refinement models (multi-timeframe)
    print("\nLoading Target Refinement models...")
    for ticker in SUPPORTED_TICKERS:
        target_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_target_model.pkl')
        if os.path.exists(target_path):
            try:
                with open(target_path, 'rb') as f:
                    target_models[ticker] = pickle.load(f)
                m = target_models[ticker]['metrics']
                print(f"  ✓ {ticker} target model loaded")
                print(f"      Both Capture Rate: {m.get('both_capture_rate', 0):.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} target model failed to load: {e}")

    # Load Enhanced v3 models (with 80 new features)
    print("\nLoading Enhanced v3 models...")
    for ticker in SUPPORTED_TICKERS:
        enhanced_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_enhanced_v3_model.pkl')
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, 'rb') as f:
                    enhanced_v3_models[ticker] = pickle.load(f)
                m = enhanced_v3_models[ticker]['metrics']
                print(f"  ✓ {ticker} enhanced v3 model loaded")
                print(f"      Accuracy: {m.get('ensemble_accuracy', 0):.1%}")
                print(f"      High Conf: {m.get('high_conf_accuracy', 0):.1%}")
                print(f"      New Features: {len(enhanced_v3_models[ticker].get('new_features_used', []))}")
            except Exception as e:
                print(f"  ✗ {ticker} enhanced v3 model failed to load: {e}")
        else:
            print(f"  - {ticker} enhanced v3 model not found (run train_enhanced_v3_model.py)")

    print(f"\nActive Model Version: {ACTIVE_MODEL_VERSION}")

    total_models = len(models) + (1 if combined_model else 0) + len(daily_models) + len(highlow_models) + len(shrinking_models) + len(intraday_models) + len(target_models) + len(enhanced_v3_models)
    print(f"Total models loaded: {total_models}")
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

            # Determine confidence tier
            if confidence >= 0.85:
                confidence_tier = 'very_high'
            elif confidence >= 0.7:
                confidence_tier = 'high'
            elif confidence >= 0.55:
                confidence_tier = 'medium'
            else:
                confidence_tier = 'low'

            results.append({
                'fvg_id': fvg.get('fvg_id'),
                'ticker': ticker,
                'prediction': 'win' if prediction == 1 else 'loss',
                'win_probability': round(float(probability), 4),
                'confidence': round(float(confidence), 4),
                'confidence_tier': confidence_tier,
                'model_used': model_name,
                'model_accuracy': round(float(model_data['metrics']['accuracy']), 4),
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


@app.route('/model_compare', methods=['GET'])
def model_compare():
    """Compare standard vs enhanced_v3 models"""
    comparison = {
        'active_version': ACTIVE_MODEL_VERSION,
        'tickers': {},
        'summary': {
            'enhanced_v3_available': len(enhanced_v3_models) > 0,
            'new_features_count': 80
        }
    }

    for ticker in SUPPORTED_TICKERS:
        ticker_data = {
            'standard': None,
            'enhanced_v3': None,
            'improvement': None
        }

        # Standard model (daily_models or models)
        if ticker in daily_models:
            m = daily_models[ticker]['metrics']
            ticker_data['standard'] = {
                'version': daily_models[ticker].get('version', 'standard'),
                'accuracy': round(float(m.get('accuracy', 0)), 4),
                'features': len(daily_models[ticker].get('features', []))
            }
        elif ticker in models:
            m = models[ticker]['metrics']
            ticker_data['standard'] = {
                'version': models[ticker].get('version', 'unknown'),
                'accuracy': round(float(m.get('accuracy', 0)), 4),
                'features': len(models[ticker].get('feature_cols', []))
            }

        # Enhanced v3 model
        if ticker in enhanced_v3_models:
            m = enhanced_v3_models[ticker]['metrics']
            ticker_data['enhanced_v3'] = {
                'version': 'enhanced_v3',
                'accuracy': round(float(m.get('ensemble_accuracy', 0)), 4),
                'high_conf_accuracy': round(float(m.get('high_conf_accuracy', 0)), 4),
                'high_conf_pct': round(float(m.get('high_conf_pct', 0)), 1),
                'features': len(enhanced_v3_models[ticker].get('features', [])),
                'new_features': len(enhanced_v3_models[ticker].get('new_features_used', [])),
                'trained_at': enhanced_v3_models[ticker].get('trained_at', 'unknown')
            }

            # Calculate improvement if both exist
            if ticker_data['standard']:
                std_acc = ticker_data['standard']['accuracy']
                enh_acc = ticker_data['enhanced_v3']['accuracy']
                ticker_data['improvement'] = {
                    'accuracy_delta': round(enh_acc - std_acc, 4),
                    'accuracy_pct_change': round((enh_acc - std_acc) / std_acc * 100, 1) if std_acc > 0 else 0
                }

        comparison['tickers'][ticker] = ticker_data

    return jsonify(comparison)


@app.route('/set_model_version', methods=['POST'])
def set_model_version():
    """Set the active model version"""
    global ACTIVE_MODEL_VERSION

    data = request.get_json() or {}
    version = data.get('version', 'standard')

    if version not in ['standard', 'enhanced_v3', 'compare']:
        return jsonify({
            'error': f'Invalid version: {version}',
            'valid_versions': ['standard', 'enhanced_v3', 'compare']
        }), 400

    ACTIVE_MODEL_VERSION = version
    return jsonify({
        'status': 'success',
        'active_version': ACTIVE_MODEL_VERSION,
        'message': f'Model version set to: {version}'
    })


def fetch_polygon_data(ticker: str, days: int = 100, ticker_type: str = 'stock') -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io

    Args:
        ticker: Stock symbol or index name
        days: Number of days of history
        ticker_type: 'stock' for regular stocks, 'index' for indices like VIX
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")

    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # For indices like VIX, use I: prefix
    api_ticker = f"I:{ticker}" if ticker_type == 'index' else ticker

    url = f"https://api.polygon.io/v2/aggs/ticker/{api_ticker}/range/1/day/{start_date}/{end_date}"
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
        # Target A: close > open
        bullish_prob_open = 0.0
        for model_name, model in models.items():
            prob = model.predict_proba(X_scaled)[0][1]
            bullish_prob_open += prob * weights.get(model_name, 0.25)

        # Target B (optional): close > CURRENT price
        bullish_prob_current = None
        models_current = model_data.get('models_current')
        weights_current = model_data.get('weights_current')
        if models_current and weights_current:
            p = 0.0
            for model_name, model in models_current.items():
                prob = model.predict_proba(X_scaled)[0][1]
                p += prob * weights_current.get(model_name, 0.25)
            bullish_prob_current = p

        # Default to "from now" probability if available
        bullish_prob = bullish_prob_current if bullish_prob_current is not None else bullish_prob_open

        return {
            'probability': round(float(bullish_prob), 3),
            'probability_close_above_open': round(float(bullish_prob_open), 3),
            'probability_close_above_current': round(float(bullish_prob_current), 3) if bullish_prob_current is not None else None,
            'confidence': round(abs(bullish_prob - 0.5) * 2, 3),
            'time_pct': round(time_pct, 2),
            'session_label': get_session_label(time_pct),
            'current_vs_open': round(current_vs_open * 100, 2),
            'position_in_range': round(position_in_range * 100, 1),
            'model_accuracy': round(float(model_data.get('metrics_current', {}).get('accuracy', model_data['metrics']['accuracy'])), 3),
            'prediction_target': 'close_above_current' if bullish_prob_current is not None else 'close_above_open',
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

    # ========== TTM SQUEEZE ==========
    # TTM Squeeze detects volatility compression (BB inside KC) and momentum direction
    # Bollinger Bands (already calculated above): bb_upper, bb_lower
    # Keltner Channels: 20 EMA +/- 1.5 * ATR
    kc_middle = df['Close'].ewm(span=20).mean()
    kc_atr = tr.rolling(20).mean()  # Using True Range calculated earlier
    kc_upper = kc_middle + 1.5 * kc_atr
    kc_lower = kc_middle - 1.5 * kc_atr

    # Squeeze is ON when BB is inside KC (low volatility compression)
    df['ttm_squeeze_on'] = ((df['bb_lower'] > kc_lower) & (df['bb_upper'] < kc_upper)).astype(int).shift(1)
    df['ttm_squeeze_off'] = (1 - df['ttm_squeeze_on']).shift(1)  # Volatility expanding

    # Squeeze "fires" when transitioning from squeeze ON to OFF
    squeeze_state = ((df['bb_lower'] > kc_lower) & (df['bb_upper'] < kc_upper)).astype(int)
    df['ttm_squeeze_fired'] = ((squeeze_state.shift(1) == 1) & (squeeze_state == 0)).astype(int).shift(1)

    # Momentum oscillator (linear regression of price - midline over 20 bars)
    midline = (df['bb_upper'] + df['bb_lower']) / 2
    momentum_val = df['Close'] - midline
    df['ttm_squeeze_momentum'] = momentum_val.shift(1)
    df['ttm_squeeze_momentum_rising'] = (momentum_val > momentum_val.shift(1)).astype(int).shift(1)

    # Count consecutive squeeze bars
    squeeze_groups = (squeeze_state != squeeze_state.shift(1)).cumsum()
    df['ttm_squeeze_bars'] = squeeze_state.groupby(squeeze_groups).cumsum().shift(1)

    # ========== KDJ (9,3,3) INDICATOR ==========
    # KDJ is an enhanced Stochastic with a J line for extreme readings
    kdj_n = 9  # Lookback period
    kdj_m1 = 3  # K smoothing
    kdj_m2 = 3  # D smoothing

    lowest_low_kdj = df['Low'].rolling(kdj_n).min()
    highest_high_kdj = df['High'].rolling(kdj_n).max()
    rsv = (df['Close'] - lowest_low_kdj) / (highest_high_kdj - lowest_low_kdj + 0.001) * 100

    # K = SMA of RSV, D = SMA of K, J = 3*K - 2*D
    df['kdj_k'] = rsv.ewm(span=kdj_m1, adjust=False).mean().shift(1)
    df['kdj_d'] = df['kdj_k'].ewm(span=kdj_m2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']  # J line for extreme readings

    # Golden Cross: K crosses above D (bullish)
    df['kdj_golden_cross'] = ((df['kdj_k'] > df['kdj_d']) & (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))).astype(int)
    # Death Cross: K crosses below D (bearish)
    df['kdj_death_cross'] = ((df['kdj_k'] < df['kdj_d']) & (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))).astype(int)

    # J line extremes (J > 100 = overbought, J < 0 = oversold)
    df['kdj_j_overbought'] = (df['kdj_j'] > 100).astype(int)
    df['kdj_j_oversold'] = (df['kdj_j'] < 0).astype(int)
    df['kdj_zone'] = np.where(df['kdj_j'] > 80, 1, np.where(df['kdj_j'] < 20, -1, 0))

    # ========== VOLATILITY REGIME ==========
    # Historical Volatility (annualized standard deviation of returns)
    # Note: daily_return is already in % (multiplied by 100), so we annualize with sqrt(252)
    df['hv_10'] = df['daily_return'].rolling(10).std() * np.sqrt(252)  # 10-day HV (already %)
    df['hv_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)  # 20-day HV (already %)
    df['hv_ratio'] = (df['hv_10'] / (df['hv_20'] + 0.001)).shift(1)  # Short/Long HV ratio

    # Volatility regime classification (use 50-day instead of 252 for faster response)
    hv_20_percentile = df['hv_20'].rolling(50, min_periods=20).rank(pct=True)  # 50-day percentile
    df['vol_regime'] = np.where(hv_20_percentile > 0.8, 2,  # High vol regime
                        np.where(hv_20_percentile < 0.2, 0,  # Low vol regime
                                 1)).astype(int)  # Normal regime
    df['vol_regime'] = df['vol_regime'].shift(1)

    df['vol_percentile'] = hv_20_percentile.shift(1) * 100  # 0-100 scale

    # Volatility expansion/contraction
    df['vol_expanding'] = (df['hv_10'] > df['hv_20']).astype(int).shift(1)
    df['vol_contracting'] = (df['hv_10'] < df['hv_20'] * 0.8).astype(int).shift(1)

    # ATR-based volatility normalized
    # Note: df['atr'] is already normalized by Close, so just multiply by 100 for %
    df['atr_pct'] = (df['atr'] * 100).shift(1)  # ATR as % of price
    df['atr_regime'] = np.where(df['atr_pct'] > df['atr_pct'].rolling(50).quantile(0.8), 2,
                        np.where(df['atr_pct'] < df['atr_pct'].rolling(50).quantile(0.2), 0,
                                 1)).astype(int)

    # ========== RSI DIVERGENCE ==========
    # Detect when price makes new extremes but RSI doesn't confirm
    div_lookback = 20

    # Price extremes
    price_new_high = (df['Close'] == df['Close'].rolling(div_lookback).max())
    price_new_low = (df['Close'] == df['Close'].rolling(div_lookback).min())

    # RSI not confirming the move
    rsi_not_high = df['rsi_14'] < df['rsi_14'].rolling(div_lookback).max()
    rsi_not_low = df['rsi_14'] > df['rsi_14'].rolling(div_lookback).min()

    df['rsi_bearish_div'] = (price_new_high & rsi_not_high).astype(int).shift(1)
    df['rsi_bullish_div'] = (price_new_low & rsi_not_low).astype(int).shift(1)

    # Divergence strength (how far RSI is from confirming)
    rsi_high_gap = df['rsi_14'].rolling(div_lookback).max() - df['rsi_14']
    rsi_low_gap = df['rsi_14'] - df['rsi_14'].rolling(div_lookback).min()
    df['rsi_div_strength'] = np.where(price_new_high, rsi_high_gap,
                              np.where(price_new_low, rsi_low_gap, 0)).astype(float)
    df['rsi_div_strength'] = df['rsi_div_strength'].shift(1)

    # ========== MACD DIVERGENCE ==========
    macd_not_high = df['macd_histogram'] < df['macd_histogram'].rolling(div_lookback).max()
    macd_not_low = df['macd_histogram'] > df['macd_histogram'].rolling(div_lookback).min()

    df['macd_bearish_div'] = (price_new_high & macd_not_high).astype(int).shift(1)
    df['macd_bullish_div'] = (price_new_low & macd_not_low).astype(int).shift(1)

    # ========== OBV DIVERGENCE ==========
    # OBV already calculated above, create running version for divergence check
    obv_running = (np.sign(df['Close'].pct_change()) * df['Volume']).cumsum()
    obv_not_high = obv_running < obv_running.rolling(div_lookback).max()
    obv_not_low = obv_running > obv_running.rolling(div_lookback).min()

    df['obv_bearish_div'] = (price_new_high & obv_not_high).astype(int).shift(1)
    df['obv_bullish_div'] = (price_new_low & obv_not_low).astype(int).shift(1)

    # Combined divergence scores
    df['bullish_div_count'] = (df['rsi_bullish_div'] + df['macd_bullish_div'] + df['obv_bullish_div'])
    df['bearish_div_count'] = (df['rsi_bearish_div'] + df['macd_bearish_div'] + df['obv_bearish_div'])
    df['div_signal'] = df['bullish_div_count'] - df['bearish_div_count']  # Positive = bullish, Negative = bearish

    # ========== BAR PATTERNS ==========
    daily_range = df['High'] - df['Low']

    # Inside bar: Range contained within previous bar
    df['inside_bar'] = ((df['High'] < df['High'].shift(1)) &
                        (df['Low'] > df['Low'].shift(1))).astype(int).shift(1)

    # Outside bar (engulfing): Range exceeds previous bar
    df['outside_bar'] = ((df['High'] > df['High'].shift(1)) &
                         (df['Low'] < df['Low'].shift(1))).astype(int).shift(1)

    # Narrow range bars (NR4, NR7) - volatility contraction
    df['narrow_range_4'] = (daily_range == daily_range.rolling(4).min()).astype(int).shift(1)
    df['narrow_range_7'] = (daily_range == daily_range.rolling(7).min()).astype(int).shift(1)

    # Wide range bar - expansion/breakout
    avg_range = daily_range.rolling(20).mean()
    df['wide_range_bar'] = (daily_range > avg_range * 2).astype(int).shift(1)

    # ========== TREND STRUCTURE ==========
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int).shift(1)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int).shift(1)
    df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int).shift(1)
    df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int).shift(1)

    # Trend structure score over 3 bars
    bullish_structure = (df['higher_high'].rolling(3, min_periods=1).sum() +
                         df['higher_low'].rolling(3, min_periods=1).sum())
    bearish_structure = (df['lower_high'].rolling(3, min_periods=1).sum() +
                         df['lower_low'].rolling(3, min_periods=1).sum())
    df['trend_structure_3'] = (bullish_structure - bearish_structure).shift(1)

    # ========== PIVOT POINTS ==========
    # Classic floor trader pivots using previous day's data
    df['pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['pivot_r1'] = 2 * df['pivot'] - df['Low'].shift(1)
    df['pivot_s1'] = 2 * df['pivot'] - df['High'].shift(1)
    df['pivot_r2'] = df['pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['pivot_s2'] = df['pivot'] - (df['High'].shift(1) - df['Low'].shift(1))

    # Distance to pivot levels (as % of price)
    df['dist_to_pivot'] = ((df['Close'].shift(1) - df['pivot']) / df['pivot'] * 100)
    df['dist_to_r1'] = ((df['pivot_r1'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100)
    df['dist_to_s1'] = ((df['Close'].shift(1) - df['pivot_s1']) / df['Close'].shift(1) * 100)

    # Position relative to pivot (above = 1, below = -1)
    df['above_pivot'] = (df['Close'].shift(1) > df['pivot']).astype(int)

    # ========== RANGE PATTERN FEATURES ==========
    range_pct = (df['High'] - df['Low']) / df['Close'] * 100
    df['avg_range_10'] = range_pct.rolling(10).mean().shift(1)
    df['avg_range_20'] = range_pct.rolling(20).mean().shift(1)
    df['range_vs_avg'] = (range_pct / (df['avg_range_20'] + 0.001)).shift(1)
    df['range_expansion'] = (df['range_vs_avg'] > 1.5).astype(int)
    df['range_contraction'] = (df['range_vs_avg'] < 0.5).astype(int)

    # Range percentile rank
    df['range_rank_20'] = range_pct.rolling(20).rank(pct=True).shift(1)

    # Consecutive narrow range bars
    narrow = (range_pct < range_pct.rolling(20).mean() * 0.7).astype(int)
    narrow_groups = (narrow != narrow.shift()).cumsum()
    df['consec_narrow'] = narrow.groupby(narrow_groups).cumsum().shift(1)

    # Breakout potential: narrow range + above-average volume
    df['breakout_potential'] = (
        (df['consec_narrow'] >= 2) &
        (df['prev_volume_ratio'] > 1.2)
    ).astype(int)

    # ========== FIBONACCI RETRACEMENTS ==========
    # Based on 20-day swing high/low
    swing_high_20 = df['High'].rolling(20).max()
    swing_low_20 = df['Low'].rolling(20).min()
    swing_range = swing_high_20 - swing_low_20

    df['fib_236'] = (swing_high_20 - 0.236 * swing_range).shift(1)
    df['fib_382'] = (swing_high_20 - 0.382 * swing_range).shift(1)
    df['fib_500'] = (swing_high_20 - 0.500 * swing_range).shift(1)
    df['fib_618'] = (swing_high_20 - 0.618 * swing_range).shift(1)

    # Distance to key fib levels (as % of price)
    df['dist_to_fib_382'] = (abs(df['Close'].shift(1) - df['fib_382']) / df['Close'].shift(1) * 100)
    df['dist_to_fib_618'] = (abs(df['Close'].shift(1) - df['fib_618']) / df['Close'].shift(1) * 100)

    # Near any key fib level (within 0.5%)
    df['near_fib_level'] = (
        (df['dist_to_fib_382'] < 0.5) |
        (df['dist_to_fib_618'] < 0.5)
    ).astype(int)

    # ========== SWING HIGH/LOW SUPPORT/RESISTANCE ==========
    df['swing_high_20'] = swing_high_20.shift(1)
    df['swing_low_20'] = swing_low_20.shift(1)

    # Distance to swing levels
    df['dist_to_resistance'] = ((df['swing_high_20'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100)
    df['dist_to_support'] = ((df['Close'].shift(1) - df['swing_low_20']) / df['Close'].shift(1) * 100)

    # At key levels (within 0.5%)
    df['at_resistance'] = (df['dist_to_resistance'] < 0.5).astype(int)
    df['at_support'] = (df['dist_to_support'] < 0.5).astype(int)

    # Position in swing range (0 = at low, 1 = at high)
    df['swing_position'] = ((df['Close'].shift(1) - df['swing_low_20']) /
                            (swing_range.shift(1) + 0.001))

    # ========== ENHANCED CALENDAR FEATURES ==========
    # Week of month
    df['week_of_month'] = (df.index.day - 1) // 7 + 1

    # Quarter boundaries
    df['is_quarter_end'] = ((df.index.month % 3 == 0) & (df['day_of_month'] >= 25)).astype(int)
    df['is_quarter_start'] = ((df.index.month % 3 == 1) & (df['day_of_month'] <= 5)).astype(int)

    # OPEX week (third Friday of month - approximate)
    df['is_opex_week'] = ((df['day_of_month'] >= 15) & (df['day_of_month'] <= 21)).astype(int)

    # First/Last trading day effects
    df['is_first_5_days'] = (df['day_of_month'] <= 5).astype(int)
    df['is_last_5_days'] = (df['day_of_month'] >= 25).astype(int)

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

    # Predict high/low % moves.
    # Support BOTH model formats:
    # - New: {'high_models': {'xgb','gb','rf'}, 'low_models': {...}, 'weights': {...}}
    # - Legacy: {'high_model': regressor, 'low_model': regressor}
    if 'high_models' in model_data and 'low_models' in model_data:
        high_models = model_data['high_models']
        pred_high_pct = (
            high_models['xgb'].predict(X_scaled)[0] * weights.get('xgb', 0.4) +
            high_models['gb'].predict(X_scaled)[0] * weights.get('gb', 0.3) +
            high_models['rf'].predict(X_scaled)[0] * weights.get('rf', 0.3)
        ) + buffer

        low_models = model_data['low_models']
        pred_low_pct = (
            low_models['xgb'].predict(X_scaled)[0] * weights.get('xgb', 0.4) +
            low_models['gb'].predict(X_scaled)[0] * weights.get('gb', 0.3) +
            low_models['rf'].predict(X_scaled)[0] * weights.get('rf', 0.3)
        ) + buffer
    else:
        # Legacy single-model format
        high_model = model_data.get('high_model')
        low_model = model_data.get('low_model')
        if high_model is None or low_model is None:
            raise KeyError('high_model')

        pred_high_pct = float(high_model.predict(X_scaled)[0]) + float(buffer)
        pred_low_pct = float(low_model.predict(X_scaled)[0]) + float(buffer)

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
    """Check if market is currently open (9:30 AM - 4:00 PM ET)

    SPEC:
    - 09:30:00 ET = OPEN (market opens)
    - 16:00:00 ET = CLOSED (market closes AT 4:00 PM, not after)
    """
    import pytz
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    # Check if weekend
    if now_et.weekday() >= 5:
        return False

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # SPEC: 9:30 AM is open, 4:00 PM is closed (use < not <=)
    return market_open <= now_et < market_close


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

            # Get ATR for dynamic target extension
            atr = float(latest['atr_14']) if not pd.isna(latest.get('atr_14', None)) else current_price * 0.01

            # Dynamically extend targets if price has already moved past them
            # For SELL signals: if price below target_low, extend target lower based on momentum
            # For BUY signals: if price above target_high, extend target higher
            original_target_low = target_low
            original_target_high = target_high

            if current_price <= target_low:
                # Price has already hit/passed the low target - extend it
                # Use remaining momentum: extend by how much we've already exceeded + buffer
                overshoot = target_low - current_price
                # Extend target by 50% of ATR or the overshoot amount, whichever is larger
                extension = max(overshoot + atr * 0.3, atr * 0.5)
                target_low = round(current_price - extension, 2)

            if current_price >= target_high:
                # Price has already hit/passed the high target - extend it
                overshoot = current_price - target_high
                extension = max(overshoot + atr * 0.3, atr * 0.5)
                target_high = round(current_price + extension, 2)

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

            # Calculate remaining potential from current price to targets
            remaining_upside = round(((target_high - current_price) / current_price) * 100, 2) if target_high > current_price else 0
            remaining_downside = round(((current_price - target_low) / current_price) * 100, 2) if target_low < current_price else 0
            remaining_upside_dollars = round(target_high - current_price, 2) if target_high > current_price else 0
            remaining_downside_dollars = round(current_price - target_low, 2) if target_low < current_price else 0

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
                'remaining_potential': {
                    'upside_pct': remaining_upside,
                    'downside_pct': remaining_downside,
                    'upside_dollars': remaining_upside_dollars,
                    'downside_dollars': remaining_downside_dollars,
                    'target_extended': bool(target_low != original_target_low or target_high != original_target_high),
                    'original_target_low': original_target_low,
                    'original_target_high': original_target_high,
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

    # Fetch VIX data for volatility context (use VIXY ETF as proxy since VIX index requires paid plan)
    try:
        # Try VIXY first (VIX ETF that tracks VIX futures)
        vix_data = fetch_polygon_data('VIXY', days=5)
        if vix_data is not None and len(vix_data) > 0:
            vix_latest = vix_data.iloc[-1]
            vix_prev = vix_data.iloc[-2] if len(vix_data) > 1 else vix_latest
            vix_current = float(vix_latest['Close'])
            vix_change = ((vix_current - float(vix_prev['Close'])) / float(vix_prev['Close'])) * 100

            # VIXY typically trades between $5-$50+ - determine volatility regime based on % change
            if vix_change >= 10:
                vix_regime = 'HIGH'
                vix_emoji = '🔴'
                vix_note = 'Extreme fear spike - expect large moves'
            elif vix_change >= 5:
                vix_regime = 'ELEVATED'
                vix_emoji = '🟠'
                vix_note = 'Above average volatility'
            elif vix_change >= -2:
                vix_regime = 'NORMAL'
                vix_emoji = '🟢'
                vix_note = 'Normal market conditions'
            else:
                vix_regime = 'LOW'
                vix_emoji = '🟢'
                vix_note = 'Low volatility - complacency'

            signals['vix'] = {
                'ticker': 'VIXY',
                'current': round(vix_current, 2),
                'change_pct': round(vix_change, 2),
                'regime': vix_regime,
                'emoji': vix_emoji,
                'note': vix_note,
            }
        else:
            signals['vix'] = None
    except Exception as e:
        signals['vix'] = {'error': str(e)}

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

            # Support both ensemble and legacy high/low model formats
            if 'high_models' in hl_model and 'low_models' in hl_model:
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
            else:
                high_pred = float(hl_model['high_model'].predict(X_hl_scaled)[0])
                low_pred = float(hl_model['low_model'].predict(X_hl_scaled)[0])

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


def fetch_intraday_ranges(ticker: str):
    """
    Fetch multi-timeframe price ranges for today.

    Returns:
    - aftermarket: Previous day 4 PM - 8 PM H/L
    - rolling_24h: Last 24 hours H/L
    - premarket: 4 AM - 9:30 AM ET H/L
    - first_30min: 9:30 AM - 10 AM H/L
    - current_session: Today's regular session H/L so far
    """
    import pytz

    if not POLYGON_API_KEY:
        return None

    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    today_str = now_et.strftime('%Y-%m-%d')
    yesterday = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')

    ranges = {
        'ticker': ticker,
        'timestamp': now_et.isoformat(),
    }

    try:
        # Fetch today's minute data
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today_str}/{today_str}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute

            # Pre-market: 4 AM - 9:30 AM
            premarket = df[
                ((df['hour'] >= 4) & (df['hour'] < 9)) |
                ((df['hour'] == 9) & (df['minute'] < 30))
            ]
            if len(premarket) > 0:
                ranges['premarket'] = {
                    'high': float(premarket['h'].max()),
                    'low': float(premarket['l'].min()),
                    'open': float(premarket.iloc[0]['o']),
                    'close': float(premarket.iloc[-1]['c']),
                    'color': '#9333ea',  # Purple
                }
            else:
                ranges['premarket'] = None

            # First 30 minutes: 9:30 AM - 10:00 AM
            first_30 = df[
                ((df['hour'] == 9) & (df['minute'] >= 30)) |
                ((df['hour'] == 10) & (df['minute'] == 0))
            ]
            if len(first_30) > 0:
                ranges['first_30min'] = {
                    'high': float(first_30['h'].max()),
                    'low': float(first_30['l'].min()),
                    'range': float(first_30['h'].max() - first_30['l'].min()),
                    'color': '#f59e0b',  # Amber
                }
            else:
                ranges['first_30min'] = None

            # Current session so far: 9:30 AM - now
            session = df[
                ((df['hour'] == 9) & (df['minute'] >= 30)) |
                ((df['hour'] >= 10) & (df['hour'] < 16))
            ]
            if len(session) > 0:
                ranges['current_session'] = {
                    'high': float(session['h'].max()),
                    'low': float(session['l'].min()),
                    'open': float(session.iloc[0]['o']),
                    'last': float(session.iloc[-1]['c']),
                    'color': '#3b82f6',  # Blue
                }
            else:
                ranges['current_session'] = None

        # Fetch yesterday's data for aftermarket
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{yesterday}/{yesterday}"
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000, 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('results'):
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
            df['hour'] = df['timestamp'].dt.hour

            # After-market: 4 PM - 8 PM
            aftermarket = df[(df['hour'] >= 16) & (df['hour'] < 20)]
            if len(aftermarket) > 0:
                ranges['aftermarket'] = {
                    'high': float(aftermarket['h'].max()),
                    'low': float(aftermarket['l'].min()),
                    'color': '#ef4444',  # Red
                }
            else:
                ranges['aftermarket'] = None

            # Yesterday's regular session for reference
            yesterday_session = df[
                ((df['hour'] == 9) & (df['minute'] >= 30)) |
                ((df['hour'] >= 10) & (df['hour'] < 16))
            ]
            if len(yesterday_session) > 0:
                ranges['yesterday_session'] = {
                    'high': float(yesterday_session['h'].max()),
                    'low': float(yesterday_session['l'].min()),
                    'close': float(yesterday_session.iloc[-1]['c']),
                    'color': '#6b7280',  # Gray
                }

        # Calculate 24-hour rolling H/L (combine yesterday + today)
        if ranges.get('current_session') and ranges.get('yesterday_session'):
            ranges['rolling_24h'] = {
                'high': max(
                    ranges['current_session']['high'],
                    ranges.get('aftermarket', {}).get('high', 0),
                    ranges.get('premarket', {}).get('high', 0),
                    ranges['yesterday_session']['high']
                ),
                'low': min(
                    ranges['current_session']['low'],
                    ranges.get('aftermarket', {}).get('low', float('inf')),
                    ranges.get('premarket', {}).get('low', float('inf')),
                    ranges['yesterday_session']['low']
                ),
                'color': '#22c55e',  # Green
            }

    except Exception as e:
        ranges['error'] = str(e)

    return ranges


@app.route('/price_ranges', methods=['GET'])
def price_ranges():
    """
    Get multi-timeframe price ranges for all tickers.

    Returns colored price ranges for:
    - Aftermarket H/L (previous day 4 PM - 8 PM) - Red
    - 24-hour rolling H/L - Green
    - Pre-market H/L (4 AM - 9:30 AM) - Purple
    - First 30 min H/L (9:30 AM - 10 AM) - Amber
    - Current session H/L - Blue
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    result = {
        'generated_at': datetime.now().isoformat(),
        'tickers': {}
    }

    for ticker in SUPPORTED_TICKERS:
        try:
            ranges = fetch_intraday_ranges(ticker)
            if ranges:
                # Add ML-refined targets if model available
                if ticker in target_models:
                    model_data = target_models[ticker]
                    ranges['ml_refined'] = {
                        'available': True,
                        'capture_rate': model_data['metrics'].get('both_capture_rate', 0),
                    }
                else:
                    ranges['ml_refined'] = {'available': False}

                result['tickers'][ticker] = ranges
            else:
                result['tickers'][ticker] = {'error': 'No data available'}
        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    return jsonify(result)


# ============================================================
# TRADING DIRECTIONS ENDPOINT - V6 Model + EV Allocator
# ============================================================

def get_v6_prediction(ticker, hourly_bars, daily_bars, current_hour):
    """
    Get prediction from V6 time-split model.

    Returns: (prob_a, prob_b, session, price_11am) or (None, None, None, None) if not available
    """
    if ticker not in intraday_v6_models:
        return None, None, None, None

    model_data = intraday_v6_models[ticker]
    feature_cols = model_data['feature_cols']

    # Get today's data
    if len(hourly_bars) < 1 or len(daily_bars) < 3:
        return None, None, None, None

    # CRITICAL: Use daily bar open (9:30 AM regular market open) to match training
    # Training uses daily_df['Open'] which is the regular market open, NOT pre-market
    # hourly_bars[0]['o'] is the 4 AM pre-market open - WRONG for V6 model
    today_open = daily_bars[-1]['o']  # Today's daily bar open = 9:30 AM regular market open
    current_close = hourly_bars[-1]['c']
    current_high = max(b['h'] for b in hourly_bars)
    current_low = min(b['l'] for b in hourly_bars)

    # Previous days
    prev_day = daily_bars[-2] if len(daily_bars) >= 2 else daily_bars[-1]
    prev_prev_day = daily_bars[-3] if len(daily_bars) >= 3 else prev_day

    # Get 11 AM price if available
    price_11am = None
    for bar in hourly_bars:
        bar_hour = pd.Timestamp(bar['t'], unit='ms', tz='America/New_York').hour
        if bar_hour == 11:
            price_11am = bar['c']
            break

    # Build features - MUST MATCH training exactly (train_time_split.py)
    gap = (today_open - prev_day['c']) / prev_day['c']
    range_so_far = max(current_high - current_low, 0.0001)

    # Calculate time_pct like training: (hours since 9 AM) / 6.5
    # Current hour from last bar timestamp
    last_bar_time = pd.Timestamp(hourly_bars[-1]['t'], unit='ms', tz='America/New_York')
    hours_since_open = (last_bar_time.hour - 9) + (last_bar_time.minute / 60)
    time_pct = min(max(hours_since_open / 6.5, 0), 1)

    features = {
        # Core features
        'gap': gap,
        'gap_size': abs(gap),
        'gap_direction': 1 if gap > 0 else (-1 if gap < 0 else 0),

        # Previous day features (CORRECT NAMES to match training)
        'prev_return': (prev_day['c'] - prev_prev_day['c']) / prev_prev_day['c'],
        'prev_range': (prev_day['h'] - prev_day['l']) / prev_day['c'],
        'prev_body': (prev_day['c'] - prev_day['o']) / prev_day['o'],
        'prev_bullish': 1 if prev_day['c'] > prev_day['o'] else 0,

        # Current position features
        'current_vs_open': (current_close - today_open) / today_open,
        'current_vs_open_direction': 1 if current_close > today_open else (-1 if current_close < today_open else 0),
        'above_open': 1 if current_close > today_open else 0,
        'position_in_range': (current_close - current_low) / range_so_far,
        'range_so_far_pct': range_so_far / today_open,

        # Near high - match training: (high - current) < (current - low)
        'near_high': 1 if (current_high - current_close) < (current_close - current_low) else 0,

        # Gap status
        'gap_filled': 1 if (gap > 0 and current_low <= prev_day['c']) or (gap <= 0 and current_high >= prev_day['c']) else 0,
        'morning_reversal': 1 if (gap > 0 and current_close < today_open) or (gap < 0 and current_close > today_open) else 0,

        # Time and momentum
        'time_pct': time_pct,
        'first_hour_return': (hourly_bars[0]['c'] - today_open) / today_open if len(hourly_bars) >= 1 else 0,
        'last_hour_return': (hourly_bars[-1]['c'] - hourly_bars[-2]['c']) / hourly_bars[-2]['c'] if len(hourly_bars) >= 2 else 0,
        'bullish_bar_ratio': sum(1 for b in hourly_bars if b['c'] > b['o']) / len(hourly_bars) if hourly_bars else 0.5,

        # Day of week
        'is_monday': 1 if last_bar_time.dayofweek == 0 else 0,
        'is_friday': 1 if last_bar_time.dayofweek == 4 else 0,

        # Mean reversion signal (simple version)
        'mean_reversion_signal': -1 if gap > 0.01 else (1 if gap < -0.01 else 0),
    }

    # 11 AM features
    if price_11am is not None and current_hour >= 11:
        features['current_vs_11am'] = (current_close - price_11am) / price_11am
        features['above_11am'] = 1 if current_close > price_11am else 0
    else:
        features['current_vs_11am'] = 0
        features['above_11am'] = 0

    # Multi-day features (simplified - use last few days from daily_bars)
    if len(daily_bars) >= 6:
        features['return_3d'] = (daily_bars[-2]['c'] - daily_bars[-5]['c']) / daily_bars[-5]['c']
        features['return_5d'] = (daily_bars[-2]['c'] - daily_bars[-7]['c']) / daily_bars[-7]['c']
        returns = [(daily_bars[i]['c'] - daily_bars[i-1]['c']) / daily_bars[i-1]['c'] for i in range(-5, 0)]
        features['volatility_5d'] = np.std(returns) if returns else 0.01
    else:
        features['return_3d'] = 0
        features['return_5d'] = 0
        features['volatility_5d'] = 0.01

    # Consecutive days
    features['consecutive_up'] = 0
    features['consecutive_down'] = 0
    for i in range(1, min(4, len(daily_bars))):
        if daily_bars[-i]['c'] > daily_bars[-i]['o']:
            features['consecutive_up'] += 1
        else:
            break
    for i in range(1, min(4, len(daily_bars))):
        if daily_bars[-i]['c'] < daily_bars[-i]['o']:
            features['consecutive_down'] += 1
        else:
            break

    # Create feature array
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Determine session and get prediction
    # SPEC: Early session is current_hour < 11, Late session is current_hour >= 11
    if current_hour < 11:
        session = 'early'
        X_scaled = model_data['scaler_early'].transform(X)
        models = model_data['models_early']
        weights = model_data['weights_early']

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights[name] for name, m in models.items())
        prob_b = 0.5  # No Target B in early session
    else:
        session = 'late'
        X_scaled = model_data['scaler_late'].transform(X)
        models_a = model_data['models_late_a']
        models_b = model_data['models_late_b']
        weights_a = model_data['weights_late_a']
        weights_b = model_data['weights_late_b']

        prob_a = sum(m.predict_proba(X_scaled)[0][1] * weights_a[name] for name, m in models_a.items())
        prob_b = sum(m.predict_proba(X_scaled)[0][1] * weights_b[name] for name, m in models_b.items())

    return prob_a, prob_b, session, price_11am


def get_probability_bucket(prob):
    """
    Classify probability into bucket.

    SPEC: 25-75% is neutral zone (no trade)
    - >= 85%: very_strong_bull (highest confidence)
    - 75-85%: strong_bull (actionable)
    - 25-75%: neutral (NO TRADE)
    - 15-25%: strong_bear (actionable)
    - <= 15%: very_strong_bear (highest confidence)
    """
    if prob >= 0.85:
        return 'very_strong_bull'
    elif prob >= 0.75:
        return 'strong_bull'
    elif prob > 0.25:
        return 'neutral'
    elif prob >= 0.15:
        return 'strong_bear'
    else:
        return 'very_strong_bear'


def get_time_multiplier(hour):
    """Position size multiplier by time of day"""
    if hour < 12:
        return 0.7
    elif hour == 12:
        return 1.0
    elif hour in [13, 14]:
        return 1.2  # Peak accuracy
    elif hour == 15:
        return 0.8
    else:
        return 0.5


def get_signal_agreement_multiplier(prob_a, prob_b):
    """Multiplier when Target A and Target B agree

    SPEC (from TRADING_ENGINE_SPEC.md):
    - Both > 0.5: aligned_bullish => 1.2
    - Both < 0.5: aligned_bearish => 1.2
    - Conflicting (one > 0.5, one < 0.5): => 0.6
    """
    if prob_a > 0.5 and prob_b > 0.5:
        return 1.2  # aligned_bullish
    elif prob_a < 0.5 and prob_b < 0.5:
        return 1.2  # aligned_bearish
    elif (prob_a > 0.5 and prob_b < 0.5) or (prob_a < 0.5 and prob_b > 0.5):
        return 0.6  # conflicting
    else:
        return 1.0  # neutral (exactly 0.5)


@app.route('/trading_directions', methods=['GET'])
def trading_directions():
    """
    Get actionable trading directions for the day.

    Uses V6 time-split model with EV-optimized allocator logic.

    Returns:
    - Per-ticker: action (LONG/SHORT/NO_TRADE), size, confidence, targets
    - Best ticker recommendation
    - Time-based guidance
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    # Get current time in ET
    from datetime import timezone
    import pytz
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    current_hour = now_et.hour
    current_minute = now_et.minute

    # Check market hours
    is_open = is_market_hours()

    # SPEC: Pre-market (before 9:30 AM) and After-hours (after 4 PM) => NO_TRADE
    # Hard gate to prevent stale signals being shown as actionable
    if not is_open:
        return jsonify({
            'generated_at': now_et.isoformat(),
            'current_time_et': now_et.strftime('%I:%M %p ET'),
            'current_hour': current_hour,
            'market_open': False,
            'action': 'NO_TRADE',
            'reason': 'Market is closed (9:30 AM - 4:00 PM ET)',
            'tickers': {},
            'best_ticker': None,
            'summary': {
                'actionable_tickers': [],
                'best_opportunity': None,
                'recommendation': 'Market is closed. No trades allowed.'
            }
        })

    result = {
        'generated_at': now_et.isoformat(),
        'current_time_et': now_et.strftime('%I:%M %p ET'),
        'current_hour': current_hour,
        'market_open': is_open,
        'session': 'early' if current_hour < 11 else 'late',  # SPEC: early < 11, late >= 11
        'time_multiplier': get_time_multiplier(current_hour),
        'tickers': {},
        'best_ticker': None,
        'trading_rules': {
            'entry': [
                'LONG only when probability >= 75%',
                'SHORT only when probability <= 25%',
                'NO TRADE in neutral zone (25-75%)',
                'Size up when Target A & B agree',
                'Peak accuracy: 1-3 PM ET'
            ],
            'sizing': {
                'very_strong': '100% of max (prob >= 85% or <= 15%)',
                'strong': '75% of max (prob 75-85% or 15-25%)',
                'neutral': 'NO TRADE (prob 25-75%)'
            },
            'exit': {
                'stop_loss': 'Ticker-specific (SPY: 0.33%, QQQ: 0.45%, IWM: 0.60%)',
                'take_profit': 'Ticker-specific (SPY: 0.25%, QQQ: 0.34%, IWM: 0.45%)',
                'time_stop': '3:50 PM ET'
            }
        },
        'model_accuracy': {
            'late_target_a': '89-92% (full year)',
            'late_target_b': '79-82% (full year)',
            'peak_hours': '1-3 PM (100% recent)'
        }
    }

    if not is_open:
        result['message'] = 'Market is closed. Predictions based on last available data.'

    # Fetch data for each ticker
    best_score = -999
    best_ticker = None

    for ticker in SUPPORTED_TICKERS:
        try:
            # Fetch hourly data for today
            today = now_et.strftime('%Y-%m-%d')
            yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

            hourly_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{today}/{today}"
            hourly_resp = requests.get(hourly_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': POLYGON_API_KEY
            })
            hourly_data = hourly_resp.json()
            hourly_bars = hourly_data.get('results', [])

            # Fetch daily data for context
            daily_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{yesterday}/{today}"
            daily_resp = requests.get(daily_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 10,
                'apiKey': POLYGON_API_KEY
            })
            daily_data = daily_resp.json()
            daily_bars = daily_data.get('results', [])

            if len(hourly_bars) < 1:
                result['tickers'][ticker] = {
                    'error': 'No hourly data available',
                    'action': 'NO_TRADE'
                }
                continue

            # Get V6 prediction
            prob_a, prob_b, session, price_11am = get_v6_prediction(ticker, hourly_bars, daily_bars, current_hour)

            if prob_a is None:
                result['tickers'][ticker] = {
                    'error': 'V6 model not available',
                    'action': 'NO_TRADE'
                }
                continue

            # Get current price
            current_price = hourly_bars[-1]['c']
            # CRITICAL: Use daily bar open (9:30 AM regular market) to match V6 training
            # SPEC: NEVER use hourly_bars[0]['o'] (pre-market 4 AM) - causes training/serving skew
            if not daily_bars:
                result['tickers'][ticker] = {
                    'action': 'NO_TRADE',
                    'reason': 'Daily open unavailable - abort to prevent skew'
                }
                continue
            today_open = daily_bars[-1]['o']
            today_change = (current_price - today_open) / today_open * 100

            # Calculate allocation
            # In late session, use prob_b (Close > 11 AM) for action direction
            # In early session, use prob_a (Close > Open)
            action_prob = prob_b if session == 'late' else prob_a
            bucket = get_probability_bucket(action_prob)

            # Check neutral zone - SPEC: 25-75% threshold (matches frontend)
            # Only generate LONG/SHORT signals at the extremes for high confidence
            if 0.25 < action_prob < 0.75:
                action = 'NO_TRADE'
                reason = f'Neutral zone ({int(action_prob * 100)}%) - wait for stronger signal'
                position_pct = 0
                confidence = 0
            else:
                # Determine direction based on session-appropriate probability
                # LONG when prob >= 75% (strongly bullish)
                # SHORT when prob <= 25% (strongly bearish)
                action = 'LONG' if action_prob >= 0.75 else 'SHORT'
                target_label = 'Target B (vs 11AM)' if session == 'late' else 'Target A (vs Open)'
                strength = 'Strong' if (action_prob >= 0.85 or action_prob <= 0.15) else 'Moderate'
                reason = f"{strength} {action.lower()} signal ({int(action_prob * 100)}%) - {target_label}"

                # Calculate position size based on action_prob (session-appropriate)
                base_pct = 25  # Base 25% of capital for actionable signals
                prob_factor = 0.5 + abs(action_prob - 0.5)  # 0.5 to 1.0
                agreement = get_signal_agreement_multiplier(prob_a, prob_b)
                time_mult = get_time_multiplier(current_hour)

                # Size by bucket (simplified - only strong and very_strong actionable)
                if bucket in ['very_strong_bull', 'very_strong_bear']:
                    size_mult = 1.0  # Full size for >= 85% or <= 15%
                else:
                    size_mult = 0.75  # Reduced for 75-85% or 15-25%

                position_pct = base_pct * prob_factor * agreement * time_mult * size_mult
                position_pct = min(position_pct, 50)  # Cap at 50%

                # Confidence as percentage (0-100)
                # 75% prob = 50 confidence, 100% prob = 100 confidence
                confidence = int(abs(action_prob - 0.5) * 200)

            # Calculate targets - based on historical median moves from 11AM to close
            # These are the MEDIAN moves when Target B is correct (2025 data)
            # Stop is set at 1.5x the target to maintain good risk/reward
            ticker_targets = {
                'SPY': {'profit': 0.0025, 'stop': 0.0033},   # TP 0.25% (median), SL 0.33%
                'QQQ': {'profit': 0.0034, 'stop': 0.0045},   # TP 0.34% (median), SL 0.45%
                'IWM': {'profit': 0.0045, 'stop': 0.0060},   # TP 0.45% (median), SL 0.60%
            }
            targets = ticker_targets.get(ticker, {'profit': 0.0025, 'stop': 0.0033})

            if action in ['LONG', 'SHORT']:
                if action == 'LONG':
                    stop_loss = current_price * (1 - targets['stop'])
                    take_profit = current_price * (1 + targets['profit'])
                else:
                    stop_loss = current_price * (1 + targets['stop'])
                    take_profit = current_price * (1 - targets['profit'])
            else:
                stop_loss = None
                take_profit = None

            # Score for best ticker selection (use action_prob for consistency)
            ev = abs(action_prob - 0.5) * (1 if action != 'NO_TRADE' else 0)
            score = ev * get_signal_agreement_multiplier(prob_a, prob_b) * get_time_multiplier(current_hour)

            if score > best_score and action != 'NO_TRADE':
                best_score = score
                best_ticker = ticker

            # Determine confidence tier for frontend display
            if action_prob >= 0.85 or action_prob <= 0.15:
                confidence_tier = 'very_high'
            elif action_prob >= 0.75 or action_prob <= 0.25:
                confidence_tier = 'high'
            elif action_prob >= 0.60 or action_prob <= 0.40:
                confidence_tier = 'medium'
            else:
                confidence_tier = 'low'

            result['tickers'][ticker] = {
                'action': action,
                'reason': reason,
                'probability_a': round(prob_a, 3),
                'probability_b': round(prob_b, 3),
                'bucket': bucket,
                'confidence_tier': confidence_tier,
                'position_pct': round(position_pct, 1),
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'today_open': round(today_open, 2),
                'price_11am': round(price_11am, 2) if price_11am else None,
                'today_change_pct': round(today_change, 2),
                'stop_loss': round(stop_loss, 2) if stop_loss else None,
                'take_profit': round(take_profit, 2) if take_profit else None,
                'session': session,
                'thresholds': {
                    'long': 0.75,
                    'short': 0.25,
                    'neutral_low': 0.25,
                    'neutral_high': 0.75
                },
                'multipliers': {
                    'time': get_time_multiplier(current_hour),
                    'agreement': get_signal_agreement_multiplier(prob_a, prob_b)
                },
                'model_accuracy': {
                    'early': round(intraday_v6_models[ticker].get('acc_early', 0), 3),
                    'late_a': round(intraday_v6_models[ticker].get('acc_late_a', 0), 3),
                    'late_b': round(intraday_v6_models[ticker].get('acc_late_b', 0), 3)
                }
            }

        except Exception as e:
            result['tickers'][ticker] = {
                'error': str(e),
                'action': 'NO_TRADE'
            }

    result['best_ticker'] = best_ticker

    # Add summary guidance with detailed reasoning
    actionable = [t for t, d in result['tickers'].items() if d.get('action') in ['LONG', 'SHORT']]
    if actionable:
        # Build detailed recommendation
        details = []
        for t in actionable:
            td = result['tickers'][t]
            action_prob = td['probability_b'] if td['session'] == 'late' else td['probability_a']
            details.append(f"{t} {td['action']} ({int(action_prob * 100)}%)")

        result['summary'] = {
            'actionable_tickers': actionable,
            'best_opportunity': best_ticker,
            'recommendation': f"Actionable: {', '.join(details)}. Best: {best_ticker}" if best_ticker else f"Signals: {', '.join(details)}"
        }
    else:
        # Explain why no trades
        neutral_info = []
        for t, td in result['tickers'].items():
            if td.get('action') == 'NO_TRADE' and not td.get('error'):
                action_prob = td.get('probability_b', td.get('probability_a', 0.5))
                neutral_info.append(f"{t}: {int(action_prob * 100)}%")

        result['summary'] = {
            'actionable_tickers': [],
            'best_opportunity': None,
            'recommendation': f"All in neutral zone (25-75%). {', '.join(neutral_info)}. Wait for stronger signals."
        }

    return jsonify(result)


# ============================================================
# NORTHSTAR PHASE PIPELINE ENDPOINT
# ============================================================

# Import the Northstar pipeline
try:
    from rpe.northstar_pipeline import NorthstarPipeline, analyze_market
    NORTHSTAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Northstar pipeline not available: {e}")
    NORTHSTAR_AVAILABLE = False


@app.route('/northstar', methods=['GET'])
def northstar_analysis():
    """
    Northstar Phase Pipeline Analysis.

    4-Phase market structure analysis:
    - Phase 1: TRUTH (RealityState) - Immutable market structure
    - Phase 2: HEALTH_GATE (SignalHealthState) - Risk assessment
    - Phase 3: DENSITY_CONTROL (SignalDensityState) - Spam control
    - Phase 4: EXECUTION_PERMISSION (ExecutionState) - Final trade permission

    Query params:
    - ticker: SPY, QQQ, or IWM (default: all)
    """
    if not NORTHSTAR_AVAILABLE:
        return jsonify({'error': 'Northstar pipeline not available'}), 500

    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    # Get ticker parameter
    ticker_param = request.args.get('ticker', 'all').upper()
    tickers = [ticker_param] if ticker_param in SUPPORTED_TICKERS else SUPPORTED_TICKERS

    # Get current time in ET
    import pytz
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    current_hour = now_et.hour

    # Check market hours
    is_open = is_market_hours()

    result = {
        'generated_at': now_et.isoformat(),
        'current_time_et': now_et.strftime('%I:%M %p ET'),
        'current_hour': current_hour,
        'market_open': is_open,
        'tickers': {},
        'pipeline_version': '1.0',
        'phase_descriptions': {
            'phase1': 'TRUTH - Immutable market structure (direction, acceptance, range, MTF)',
            'phase2': 'HEALTH_GATE - Signal health scoring (structural integrity, participation)',
            'phase3': 'DENSITY_CONTROL - Spam/clustering control (throttle state)',
            'phase4': 'EXECUTION_PERMISSION - Final trade permission (bias, mode, risk)'
        }
    }

    # Initialize pipeline
    pipeline = NorthstarPipeline()

    for ticker in tickers:
        try:
            # Fetch 1-minute bars for today
            today = now_et.strftime('%Y-%m-%d')
            yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

            # Fetch 1-minute data (for intraday analysis)
            minute_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}"
            minute_resp = requests.get(minute_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': POLYGON_API_KEY
            })
            minute_data = minute_resp.json()
            minute_bars = minute_data.get('results', [])

            # Fetch daily data for context
            daily_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{yesterday}/{today}"
            daily_resp = requests.get(daily_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 10,
                'apiKey': POLYGON_API_KEY
            })
            daily_data = daily_resp.json()
            daily_bars_raw = daily_data.get('results', [])

            if len(minute_bars) < 30:
                result['tickers'][ticker] = {
                    'error': 'Insufficient minute data',
                    'bars_available': len(minute_bars)
                }
                continue

            # Convert to pandas DataFrames
            bars_1m = pd.DataFrame(minute_bars)
            bars_1m = bars_1m.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            daily_df = None
            if len(daily_bars_raw) >= 2:
                daily_df = pd.DataFrame(daily_bars_raw)
                daily_df = daily_df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            # Run the Northstar pipeline
            analysis = pipeline.run(
                symbol=ticker,
                bars_1m=bars_1m,
                daily_bars=daily_df,
                signals_last_10m=0,  # Can be enhanced with signal tracking
                time_since_acceptance_minutes=0  # Can be enhanced with state tracking
            )

            # Add current price info
            current_price = minute_bars[-1]['c']
            # SPEC: Use daily bar open (9:30 AM) not minute_bars[0]['o'] (pre-market)
            today_open = daily_bars_raw[-1]['o'] if daily_bars_raw else None
            today_change = ((current_price - today_open) / today_open * 100) if today_open else None

            analysis['current_price'] = round(current_price, 2)
            analysis['today_open'] = round(today_open, 2) if today_open else None
            analysis['today_change_pct'] = round(today_change, 2) if today_change else None
            analysis['bars_analyzed'] = len(minute_bars)

            result['tickers'][ticker] = analysis

        except Exception as e:
            result['tickers'][ticker] = {
                'error': str(e),
                'phase1': None,
                'phase2': None,
                'phase3': None,
                'phase4': None
            }

    # Add summary across all tickers
    allowed_tickers = []
    for t, data in result['tickers'].items():
        if data.get('phase4', {}).get('allowed', False):
            allowed_tickers.append({
                'ticker': t,
                'bias': data['phase4']['bias'],
                'mode': data['phase4']['execution_mode'],
                'risk': data['phase4']['risk_state']
            })

    result['summary'] = {
        'execution_allowed': len(allowed_tickers) > 0,
        'allowed_tickers': allowed_tickers,
        'recommendation': f"Execute on {allowed_tickers[0]['ticker']} ({allowed_tickers[0]['bias']})" if allowed_tickers else "No execution permission - stand down"
    }

    return jsonify(result)


# ============================================================
# REPLAY MODE - Time Travel Testing
# ============================================================

@app.route('/replay', methods=['GET'])
def replay_mode():
    """
    Replay Mode - Simulate any historical trading day at any point in time.

    Query params:
    - date: YYYY-MM-DD (required) - The day to replay
    - time: HH:MM (required) - Simulated time in ET (e.g., "10:30", "14:15")
    - ticker: SPY, QQQ, IWM, or 'all' (default: all)

    Returns both V6 trading directions AND Northstar analysis
    as they would have appeared at that exact moment.
    """
    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    if not NORTHSTAR_AVAILABLE:
        return jsonify({'error': 'Northstar pipeline not available'}), 500

    # Parse parameters
    replay_date = request.args.get('date')
    replay_time = request.args.get('time')
    ticker_param = request.args.get('ticker', 'all').upper()

    if not replay_date or not replay_time:
        return jsonify({
            'error': 'Missing required parameters',
            'usage': '/replay?date=2025-12-20&time=14:30&ticker=SPY'
        }), 400

    # Parse time
    try:
        hour, minute = map(int, replay_time.split(':'))
        if hour < 4 or hour > 20:
            return jsonify({'error': 'Time must be between 04:00 and 20:00 ET'}), 400
    except:
        return jsonify({'error': 'Invalid time format. Use HH:MM (e.g., 14:30)'}), 400

    # Determine tickers
    tickers = [ticker_param] if ticker_param in SUPPORTED_TICKERS else SUPPORTED_TICKERS

    # Calculate simulated timestamp cutoff (in milliseconds)
    import pytz
    from datetime import datetime as dt
    et_tz = pytz.timezone('America/New_York')

    # Create the simulated datetime
    simulated_dt = et_tz.localize(dt.strptime(f"{replay_date} {replay_time}", "%Y-%m-%d %H:%M"))
    simulated_ts = int(simulated_dt.timestamp() * 1000)

    # Determine session and market status
    is_market_open = 9 * 60 + 30 <= hour * 60 + minute <= 16 * 60
    session = 'early' if hour < 11 else 'late'  # SPEC: early < 11, late >= 11

    result = {
        'mode': 'REPLAY',
        'replay_date': replay_date,
        'replay_time': replay_time,
        'simulated_time_et': simulated_dt.strftime('%I:%M %p ET'),
        'simulated_hour': hour,
        'simulated_minute': minute,
        'market_open': is_market_open,
        'session': session,
        'tickers': {},
        'v6_signals': {},
        'northstar': {},
        'summary': {}
    }

    # Get previous day for daily context
    from datetime import timedelta
    prev_date = (dt.strptime(replay_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")

    pipeline = NorthstarPipeline()
    best_ticker = None
    best_score = -999

    for ticker in tickers:
        try:
            # Fetch minute data for the replay date
            minute_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{replay_date}/{replay_date}"
            minute_resp = requests.get(minute_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': POLYGON_API_KEY
            })
            minute_data = minute_resp.json()
            all_minute_bars = minute_data.get('results', [])

            # Filter to only bars BEFORE simulated time
            minute_bars = [b for b in all_minute_bars if b['t'] <= simulated_ts]

            if len(minute_bars) < 10:
                result['tickers'][ticker] = {
                    'error': f'Insufficient data at {replay_time}',
                    'bars_available': len(minute_bars)
                }
                continue

            # Fetch hourly data for V6 (also filtered)
            hourly_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{replay_date}/{replay_date}"
            hourly_resp = requests.get(hourly_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': POLYGON_API_KEY
            })
            hourly_data = hourly_resp.json()
            all_hourly_bars = hourly_data.get('results', [])
            hourly_bars = [b for b in all_hourly_bars if b['t'] <= simulated_ts]

            # Fetch daily data for context
            daily_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{prev_date}/{replay_date}"
            daily_resp = requests.get(daily_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 10,
                'apiKey': POLYGON_API_KEY
            })
            daily_data = daily_resp.json()
            daily_bars = daily_data.get('results', [])

            # Current price at simulated time
            current_price = minute_bars[-1]['c']

            # CRITICAL: Use daily bar open (9:30 AM regular market) to match V6 training
            # Training uses daily_df['Open'] which is the regular market open
            today_open = daily_bars[-1]['o'] if daily_bars else None

            # Find 11 AM price for Target B
            price_11am = None
            for bar in minute_bars:
                bar_time = dt.fromtimestamp(bar['t'] / 1000, tz=et_tz)
                bar_hour = bar_time.hour
                bar_minute = bar_time.minute

                # 11 AM price for Target B
                if bar_hour == 11 and bar_minute == 0 and price_11am is None:
                    price_11am = bar['c']
                    break

            # SPEC: NEVER use minute_bars[0]['o'] as fallback - causes training/serving skew
            # If daily open unavailable, we cannot compute accurate features
            if today_open is None:
                result['tickers'][ticker] = {
                    'error': 'Daily open unavailable - cannot compute V6 features',
                    'v6_action': 'NO_TRADE',
                    'v6_reason': 'Daily open unavailable - abort to prevent skew'
                }
                continue

            today_change = (current_price - today_open) / today_open * 100

            # ==================
            # V6 PREDICTIONS
            # ==================
            prob_a, prob_b = None, None
            if len(hourly_bars) >= 1 and ticker in intraday_v6_models:
                try:
                    prob_a, prob_b, _, _ = get_v6_prediction(ticker, hourly_bars, daily_bars, hour)
                except:
                    pass

            # Calculate V6 action
            v6_action = 'NO_TRADE'
            v6_reason = 'Insufficient data'
            if prob_a is not None and prob_b is not None:
                action_prob = prob_b if session == 'late' else prob_a
                if 0.45 <= action_prob <= 0.55:
                    v6_action = 'NO_TRADE'
                    v6_reason = 'Neutral zone'
                else:
                    v6_action = 'LONG' if action_prob > 0.5 else 'SHORT'
                    bucket = get_probability_bucket(action_prob)
                    v6_reason = f"{bucket.replace('_', ' ').title()}"

            result['v6_signals'][ticker] = {
                'action': v6_action,
                'reason': v6_reason,
                'probability_a': round(prob_a, 3) if prob_a else None,
                'probability_b': round(prob_b, 3) if prob_b else None,
                'session': session,
                'price_11am': round(price_11am, 2) if price_11am else None
            }

            # ==================
            # NORTHSTAR ANALYSIS
            # ==================
            bars_1m = pd.DataFrame(minute_bars)
            bars_1m = bars_1m.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            daily_df = None
            if len(daily_bars) >= 2:
                daily_df = pd.DataFrame(daily_bars)
                daily_df = daily_df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            northstar_analysis = pipeline.run(
                symbol=ticker,
                bars_1m=bars_1m,
                daily_bars=daily_df,
                signals_last_10m=0,
                time_since_acceptance_minutes=0
            )

            result['northstar'][ticker] = {
                'phase1': northstar_analysis['phase1'],
                'phase2': northstar_analysis['phase2'],
                'phase3': northstar_analysis['phase3'],
                'phase4': northstar_analysis['phase4']
            }

            # ==================
            # COMBINED TICKER DATA
            # ==================
            result['tickers'][ticker] = {
                'current_price': round(current_price, 2),
                'today_open': round(today_open, 2),
                'today_change_pct': round(today_change, 2),
                'price_11am': round(price_11am, 2) if price_11am else None,
                'bars_analyzed': len(minute_bars),
                'v6': result['v6_signals'][ticker],
                'northstar': result['northstar'][ticker]
            }

            # Score for best ticker
            if v6_action != 'NO_TRADE' and result['northstar'][ticker]['phase4']['allowed']:
                action_prob = prob_b if session == 'late' else prob_a
                score = abs(action_prob - 0.5) * 100
                if score > best_score:
                    best_score = score
                    best_ticker = ticker

        except Exception as e:
            result['tickers'][ticker] = {'error': str(e)}

    # Summary
    allowed_tickers = [t for t, d in result['tickers'].items()
                       if not d.get('error') and d.get('northstar', {}).get('phase4', {}).get('allowed')]

    result['summary'] = {
        'best_ticker': best_ticker,
        'allowed_tickers': allowed_tickers,
        'v6_actionable': [t for t, d in result.get('v6_signals', {}).items() if d.get('action') in ['LONG', 'SHORT']],
        'recommendation': f"Best: {best_ticker}" if best_ticker else "No clear opportunity"
    }

    return jsonify(result)


# ============================================================
# REALITY PROOF ENGINE (RPE) - 5-Phase Pipeline
# ============================================================

# Import the RPE pipeline
try:
    from rpe.rpe_engine import RPEPipeline, analyze_market_rpe
    RPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RPE engine not available: {e}")
    RPE_AVAILABLE = False


@app.route('/rpe', methods=['GET'])
def rpe_analysis():
    """
    Reality Proof Engine (RPE) - 5-Phase Market Structure Analysis.

    5-Phase pipeline with strict layering invariants:
    - Phase 1: TRUTH - Core market structure (immutable, no ML)
    - Phase 2: SIGNAL_HEALTH - Data integrity and conditions
    - Phase 3: SIGNAL_DENSITY - Mode, cooldown, budget, material change
    - Phase 4: EXECUTION_POSTURE - Bias, play type, confidence, invalidation
    - Phase 5: LEARNING_FORECASTING - Predictions, forecasts, entry/exit (ONLY phase with ML)

    INVARIANTS:
    - truth_never_depends_on_decisions: Phase 1 is pure observation
    - no_repainting: All calculations use only bars where t < now
    - predictions_only_in_phase5: No ML predictions in phases 1-4

    Query params:
    - ticker: SPY, QQQ, or IWM (default: all)
    """
    if not RPE_AVAILABLE:
        return jsonify({'error': 'RPE engine not available'}), 500

    if not POLYGON_API_KEY:
        return jsonify({'error': 'POLYGON_API_KEY not configured'}), 500

    # Get ticker parameter
    ticker_param = request.args.get('ticker', 'all').upper()
    tickers = [ticker_param] if ticker_param in SUPPORTED_TICKERS else SUPPORTED_TICKERS

    # Get current time in ET
    import pytz
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    current_hour = now_et.hour

    # Check market hours
    is_open = is_market_hours()

    result = {
        'generated_at': now_et.isoformat(),
        'current_time_et': now_et.strftime('%I:%M %p ET'),
        'current_hour': current_hour,
        'market_open': is_open,
        'tickers': {},
        'contract_version': '2.0',
        'phase_descriptions': {
            'phase1_truth': 'Core market structure (immutable, no ML predictions)',
            'phase2_signal_health': 'Data integrity + signal health assessment',
            'phase3_signal_density': 'Mode, cooldown, budget, material change tracking',
            'phase4_execution_posture': 'Bias, play type, confidence, invalidation',
            'phase5_learning_forecasting': 'ONLY phase with ML - forecasts, entry/exit, triple barrier'
        }
    }

    # Initialize pipeline
    pipeline = RPEPipeline()

    for ticker in tickers:
        try:
            # Fetch 1-minute bars for today
            today = now_et.strftime('%Y-%m-%d')
            yesterday = (now_et - timedelta(days=5)).strftime('%Y-%m-%d')

            # Fetch 1-minute data (for intraday analysis)
            minute_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}"
            minute_resp = requests.get(minute_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': POLYGON_API_KEY
            })
            minute_data = minute_resp.json()
            minute_bars = minute_data.get('results', [])

            # Fetch daily data for context
            daily_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{yesterday}/{today}"
            daily_resp = requests.get(daily_url, params={
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 10,
                'apiKey': POLYGON_API_KEY
            })
            daily_data = daily_resp.json()
            daily_bars_raw = daily_data.get('results', [])

            if len(minute_bars) < 30:
                result['tickers'][ticker] = {
                    'error': 'Insufficient minute data',
                    'bars_available': len(minute_bars)
                }
                continue

            # Convert to pandas DataFrames
            bars_1m = pd.DataFrame(minute_bars)
            bars_1m = bars_1m.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            daily_df = None
            if len(daily_bars_raw) >= 2:
                daily_df = pd.DataFrame(daily_bars_raw)
                daily_df = daily_df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'})

            # Get V6 predictions for Phase 5 using proper V6 model
            v6_preds = None
            try:
                # Fetch hourly bars for V6 model (same as trading_directions)
                hourly_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{today}/{today}"
                hourly_resp = requests.get(hourly_url, params={
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50000,
                    'apiKey': POLYGON_API_KEY
                })
                hourly_bars = hourly_resp.json().get('results', [])

                if len(hourly_bars) >= 1 and len(daily_bars_raw) >= 3:
                    # Use proper get_v6_prediction function
                    prob_a, prob_b, session, price_11am = get_v6_prediction(
                        ticker, hourly_bars, daily_bars_raw, current_hour
                    )
                    if prob_a is not None and prob_b is not None:
                        v6_preds = {
                            'target_a_prob': float(prob_a),
                            'target_b_prob': float(prob_b),
                            'session': session,
                            'price_11am': float(price_11am) if price_11am else None
                        }
                    else:
                        print(f"V6 returned None for {ticker}")
            except Exception as e:
                pass  # V6 predictions are optional

            # Run the RPE pipeline
            analysis = pipeline.run(
                symbol=ticker,
                bars_1m=bars_1m,
                daily_bars=daily_df,
                signals_last_10m=0,
                time_since_acceptance_minutes=0,
                v6_predictions=v6_preds
            )

            # Add current price info
            current_price = minute_bars[-1]['c']
            # SPEC: Use daily bar open (9:30 AM regular market) - NEVER minute_bars[0]['o']
            today_open = daily_bars_raw[-1]['o'] if daily_bars_raw else None
            today_change = ((current_price - today_open) / today_open * 100) if today_open else None

            analysis['current_price'] = round(current_price, 2)
            analysis['today_open'] = round(today_open, 2) if today_open else None
            analysis['today_change_pct'] = round(today_change, 2) if today_change else None
            analysis['bars_analyzed'] = len(minute_bars)

            # Add V6 Target A/B predictions to output
            if v6_preds and 'target_a_prob' in v6_preds:
                analysis['v6_signals'] = {
                    'target_a_prob': round(v6_preds['target_a_prob'], 3),
                    'target_b_prob': round(v6_preds['target_b_prob'], 3),
                    'session': v6_preds.get('session'),
                    'price_11am': round(v6_preds.get('price_11am'), 2) if v6_preds.get('price_11am') else None,
                    'target_a_action': 'LONG' if v6_preds['target_a_prob'] > 0.5 else 'SHORT',
                    'target_b_action': 'LONG' if v6_preds['target_b_prob'] > 0.5 else 'SHORT'
                }

            result['tickers'][ticker] = analysis

        except Exception as e:
            import traceback
            result['tickers'][ticker] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    # Add summary across all tickers
    allowed_tickers = []
    for t, data in result['tickers'].items():
        if data.get('phase4_execution_posture', {}).get('allowed', False):
            allowed_tickers.append({
                'ticker': t,
                'bias': data['phase4_execution_posture']['bias']['bias'],
                'play_type': data['phase4_execution_posture']['play_type']['play_type'],
                'confidence': data['phase4_execution_posture']['confidence_risk']['confidence']
            })

    result['summary'] = {
        'execution_allowed': len(allowed_tickers) > 0,
        'allowed_tickers': allowed_tickers,
        'recommendation': f"Execute on {allowed_tickers[0]['ticker']} ({allowed_tickers[0]['bias']})" if allowed_tickers else "No execution permission - stand down"
    }

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
        print(f"  GET  /northstar     - 4-phase Northstar analysis")
        print(f"  GET  /rpe           - 5-phase Reality Proof Engine")
        print(f"  GET  /replay        - Time-travel replay mode")
        print("="*50)
        print("\nRECOMMENDATION: Only trade when win_probability >= 0.7")
        print("="*50 + "\n")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Error: No models could be loaded.")
        print("Run upgrade_production_models.py first to train models.")
