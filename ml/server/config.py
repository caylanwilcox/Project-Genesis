"""
Server Configuration

Single source of truth for all server settings.
"""

import os

# =============================================================================
# API KEYS
# =============================================================================
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '').strip()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'v6_models')

# =============================================================================
# TICKERS
# =============================================================================
SUPPORTED_TICKERS = ['SPY', 'QQQ', 'IWM']

# =============================================================================
# MODEL SETTINGS
# =============================================================================
ACTIVE_MODEL_VERSION = os.environ.get('MODEL_VERSION', 'standard')

# =============================================================================
# SIGNAL SETTINGS
# =============================================================================
SIGNAL_LOCK_MINUTES = 60  # Lock signals for 1 hour to prevent flip-flopping

# Neutral zone - probabilities in this range = NO_TRADE
# Model accuracy is only reliable at extremes (>75% or <25%)
NEUTRAL_ZONE_LOW = 0.25
NEUTRAL_ZONE_HIGH = 0.75

# =============================================================================
# SESSION BOUNDARIES (SPEC COMPLIANT)
# =============================================================================
# Early session: hour < 11 (9:30 AM - 10:59 AM)
# Late session: hour >= 11 (11:00 AM - 4:00 PM)
EARLY_SESSION_END_HOUR = 11

# Market hours
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# =============================================================================
# VOLATILITY THRESHOLDS
# =============================================================================
LOW_VOL_THRESHOLD = 0.30
HIGH_VOL_THRESHOLD = 0.70

# =============================================================================
# CATEGORICAL MAPPINGS
# =============================================================================
CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}

# =============================================================================
# V6 FEATURE SCHEMA
# =============================================================================
# NOTE: V6 models are SELF-DESCRIBING - feature_cols is stored inside the pickle file
# and loaded at runtime. This ensures training/serving feature alignment.
#
# The model's feature_cols (29 features) are defined in ml/server/v6/features.py
# and saved into the pickle during training (ml/train_time_split.py:367).
#
# At prediction time, we load feature_cols from the model itself:
#   feature_cols = model_data['feature_cols']  # See ml/server/v6/predictions.py:31
#
# DO NOT define V6_FEATURE_COLS here - it would create a second source of truth
# that could drift from the actual trained model.
