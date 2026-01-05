"""
Legacy Prediction Endpoints

FVG and other legacy prediction endpoints (for backwards compatibility).
"""

from flask import Blueprint, jsonify, request
import numpy as np
import pandas as pd

from ..config import SUPPORTED_TICKERS, CATEGORICAL_MAPPINGS
from ..models.store import models, combined_model

bp = Blueprint('predictions', __name__)


def engineer_features(data):
    """Add engineered features to the data dict (legacy FVG model)"""
    features = data.copy()

    # Size ratios
    fvg_size = data.get('fvg_size', 0)
    atr = data.get('atr', 1)
    features['fvg_atr_ratio'] = fvg_size / max(atr, 0.01)

    body_size = data.get('body_size', 0)
    features['body_atr_ratio'] = body_size / max(atr, 0.01)

    # Wick analysis
    upper_wick = data.get('upper_wick', 0)
    lower_wick = data.get('lower_wick', 0)
    total_wick = upper_wick + lower_wick
    features['wick_ratio'] = upper_wick / max(lower_wick, 0.01)
    features['body_to_wick_ratio'] = body_size / max(total_wick, 0.01)

    # Volume strength
    volume = data.get('volume', 0)
    avg_volume = data.get('avg_volume', 1)
    features['relative_volume'] = volume / max(avg_volume, 1)

    # Trend strength
    ema_20 = data.get('ema_20', 0)
    ema_50 = data.get('ema_50', 0)
    close = data.get('close', 0)
    features['ema_spread'] = (ema_20 - ema_50) / max(close, 1) * 100
    features['price_vs_ema20'] = (close - ema_20) / max(close, 1) * 100

    # RSI divergence
    rsi = data.get('rsi', 50)
    features['rsi_extreme'] = 1 if rsi > 70 or rsi < 30 else 0
    features['rsi_distance_from_50'] = abs(rsi - 50)

    # MACD strength
    macd = data.get('macd', 0)
    features['macd_strength'] = abs(macd) / max(atr, 0.01)

    # FVG quality score
    quality_score = 0
    if fvg_size > 0.5:
        quality_score += 1
    if features['relative_volume'] > 1.2:
        quality_score += 1
    if rsi > 30 and rsi < 70:
        quality_score += 1
    features['fvg_quality_score'] = quality_score

    # Gap characteristics
    gap_pct = data.get('gap_pct', 0)
    features['gap_strength'] = abs(gap_pct)
    features['gap_direction'] = 1 if gap_pct > 0 else -1 if gap_pct < 0 else 0

    return features


def build_features(data, feature_cols):
    """Build feature array for prediction (legacy FVG model)"""
    features = []

    for col in feature_cols:
        if col in CATEGORICAL_MAPPINGS:
            val = data.get(col, 'unknown')
            mapping = CATEGORICAL_MAPPINGS[col]
            features.append(mapping.get(val, mapping.get('unknown', 0)))
        else:
            val = data.get(col, 0)
            if isinstance(val, (int, float)):
                features.append(val)
            else:
                features.append(0)

    return np.array([features])


@bp.route('/predict', methods=['POST'])
def predict():
    """Single FVG prediction endpoint (legacy)"""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    ticker = data.get('ticker', 'SPY').upper()

    # Get model
    model_data = models.get(ticker) or combined_model
    if not model_data:
        return jsonify({'error': 'No model available'}), 500

    try:
        # Engineer features
        features = engineer_features(data)

        # Build feature array
        feature_cols = model_data['feature_cols']
        X = build_features(features, feature_cols)

        # Predict
        proba = model_data['ensemble'].predict_proba(X)[0]
        prediction = 1 if proba[1] >= 0.5 else 0

        return jsonify({
            'prediction': int(prediction),
            'confidence': float(max(proba)),
            'probability_bullish': float(proba[1]),
            'probability_bearish': float(proba[0]),
            'ticker': ticker,
            'model_used': 'ticker_specific' if ticker in models else 'combined'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch FVG prediction endpoint (legacy)"""
    data = request.get_json()

    if not data or 'fvgs' not in data:
        return jsonify({'error': 'No FVGs provided'}), 400

    fvgs = data['fvgs']
    results = []

    for fvg in fvgs:
        ticker = fvg.get('ticker', 'SPY').upper()
        model_data = models.get(ticker) or combined_model

        if not model_data:
            results.append({'error': 'No model available', 'ticker': ticker})
            continue

        try:
            features = engineer_features(fvg)
            feature_cols = model_data['feature_cols']
            X = build_features(features, feature_cols)

            proba = model_data['ensemble'].predict_proba(X)[0]
            prediction = 1 if proba[1] >= 0.5 else 0

            results.append({
                'prediction': int(prediction),
                'confidence': float(max(proba)),
                'probability_bullish': float(proba[1]),
                'ticker': ticker
            })

        except Exception as e:
            results.append({'error': str(e), 'ticker': ticker})

    return jsonify({'predictions': results})
