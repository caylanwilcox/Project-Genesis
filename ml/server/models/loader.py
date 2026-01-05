"""
Model Loader

Loads all ML models at server startup.
"""

import os
import pickle
from ..config import MODELS_DIR, SUPPORTED_TICKERS, ACTIVE_MODEL_VERSION
from . import store


def load_all_models():
    """Load all available models on startup"""
    print("Loading ML models...")

    # Load V6 time-split models (primary production models)
    print("\nLoading V6 Time-Split Intraday models...")
    for ticker in SUPPORTED_TICKERS:
        v6_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
        if os.path.exists(v6_path):
            try:
                with open(v6_path, 'rb') as f:
                    store.intraday_v6_models[ticker] = pickle.load(f)
                acc_early = store.intraday_v6_models[ticker].get('acc_early', 0)
                acc_late_a = store.intraday_v6_models[ticker].get('acc_late_a', 0)
                acc_late_b = store.intraday_v6_models[ticker].get('acc_late_b', 0)
                print(f"  ✓ {ticker} V6 model loaded")
                print(f"      Early: {acc_early:.1%}, Late A: {acc_late_a:.1%}, Late B: {acc_late_b:.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} V6 model failed to load: {e}")
        else:
            print(f"  - {ticker} V6 model not found")

    # Load V6.1 SWING models (multi-day predictions) - upgraded with CatBoost, VIX, cross-asset
    print("\nLoading V6.1 SWING models...")
    for ticker in SUPPORTED_TICKERS:
        # Try V6.1 first (upgraded), fall back to V6
        swing_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_v6_1.pkl')
        version = 'v6.1'
        if not os.path.exists(swing_path):
            swing_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_v6.pkl')
            version = 'v6'

        if os.path.exists(swing_path):
            try:
                with open(swing_path, 'rb') as f:
                    store.swing_v6_models[ticker] = pickle.load(f)
                acc_5d = store.swing_v6_models[ticker].get('acc_5d', 0)
                acc_10d = store.swing_v6_models[ticker].get('acc_10d', 0)
                print(f"  ✓ {ticker} SWING {version} model loaded")
                print(f"      5-Day: {acc_5d:.1%}, 10-Day: {acc_10d:.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} SWING model failed to load: {e}")
        else:
            print(f"  - {ticker} SWING model not found")

    # Load 3-Day SWING models
    print("\nLoading 3-Day SWING models...")
    for ticker in SUPPORTED_TICKERS:
        swing_3d_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_3d.pkl')
        if os.path.exists(swing_3d_path):
            try:
                with open(swing_3d_path, 'rb') as f:
                    store.swing_3d_models[ticker] = pickle.load(f)
                acc_3d = store.swing_3d_models[ticker].get('acc_3d', 0)
                print(f"  ✓ {ticker} 3-Day model loaded: {acc_3d:.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} 3-Day model failed to load: {e}")
        else:
            print(f"  - {ticker} 3-Day model not found")

    # Load 1-Day SWING models
    print("\nLoading 1-Day SWING models...")
    for ticker in SUPPORTED_TICKERS:
        swing_1d_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_swing_1d.pkl')
        if os.path.exists(swing_1d_path):
            try:
                with open(swing_1d_path, 'rb') as f:
                    store.swing_1d_models[ticker] = pickle.load(f)
                acc_1d = store.swing_1d_models[ticker].get('acc_1d', 0)
                print(f"  ✓ {ticker} 1-Day model loaded: {acc_1d:.1%}")
            except Exception as e:
                print(f"  ✗ {ticker} 1-Day model failed to load: {e}")
        else:
            print(f"  - {ticker} 1-Day model not found")

    # Load legacy models (for backwards compatibility)
    _load_legacy_models()

    total = len(store.intraday_v6_models)
    swing_total = len(store.swing_v6_models)
    print(f"\nV6 Intraday models loaded: {total}")
    print(f"V6 SWING models loaded: {swing_total}")
    print(f"Active Model Version: {ACTIVE_MODEL_VERSION}")
    return total > 0 or swing_total > 0


def _load_legacy_models():
    """Load legacy models (FVG, daily, highlow, etc.)"""

    # FVG models
    for ticker in SUPPORTED_TICKERS:
        model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_fvg_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    store.models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Combined model
    combined_path = os.path.join(MODELS_DIR, 'combined_fvg_model.pkl')
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'rb') as f:
                store.combined_model = pickle.load(f)
        except Exception:
            pass

    # Daily direction models
    for ticker in SUPPORTED_TICKERS:
        daily_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_daily_model.pkl')
        if os.path.exists(daily_path):
            try:
                with open(daily_path, 'rb') as f:
                    store.daily_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # High/Low models
    for ticker in SUPPORTED_TICKERS:
        highlow_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_highlow_model.pkl')
        if os.path.exists(highlow_path):
            try:
                with open(highlow_path, 'rb') as f:
                    store.highlow_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Shrinking range models
    for ticker in SUPPORTED_TICKERS:
        shrink_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_shrinking_model.pkl')
        if os.path.exists(shrink_path):
            try:
                with open(shrink_path, 'rb') as f:
                    store.shrinking_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Regime models
    for ticker in SUPPORTED_TICKERS:
        regime_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_regime_model.pkl')
        if os.path.exists(regime_path):
            try:
                with open(regime_path, 'rb') as f:
                    store.regime_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Intraday models
    for ticker in SUPPORTED_TICKERS:
        intraday_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_model.pkl')
        if os.path.exists(intraday_path):
            try:
                with open(intraday_path, 'rb') as f:
                    store.intraday_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Target models
    for ticker in SUPPORTED_TICKERS:
        target_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_target_model.pkl')
        if os.path.exists(target_path):
            try:
                with open(target_path, 'rb') as f:
                    store.target_models[ticker] = pickle.load(f)
            except Exception:
                pass

    # Enhanced v3 models
    for ticker in SUPPORTED_TICKERS:
        enhanced_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_enhanced_v3_model.pkl')
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, 'rb') as f:
                    store.enhanced_v3_models[ticker] = pickle.load(f)
            except Exception:
                pass


def get_v6_model(ticker: str):
    """Get V6 model for a ticker"""
    return store.intraday_v6_models.get(ticker.upper())


def get_model_for_ticker(ticker: str):
    """Get the best model for a given ticker (legacy)"""
    ticker = ticker.upper()
    if ticker in store.models:
        return store.models[ticker]
    return store.combined_model
