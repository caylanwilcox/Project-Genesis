"""
Configuration for SPY Daily Range Plan Prediction System
"""
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """LightGBM model configuration"""
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42

@dataclass
class DataConfig:
    """Data configuration"""
    symbol: str = "SPY"
    start_year: int = 2004
    end_year: int = 2024

    # Session times (Eastern)
    market_open: str = "09:30"
    market_close: str = "16:00"
    or_end: str = "09:45"  # Opening Range end (15 min)

    # ATR lookback
    atr_period: int = 14

    # Minimum bars for valid session
    min_session_bars: int = 100

@dataclass
class TargetConfig:
    """Target level configuration in ATR units"""
    t1_units: float = 0.5   # First target
    t2_units: float = 1.0   # Second target
    t3_units: float = 1.5   # Third target
    sl_units: float = 1.25  # Stop loss

@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration"""
    train_start_year: int = 2004
    train_end_offset: int = 1  # Train up to test_year - 1
    test_years: List[int] = None

    def __post_init__(self):
        if self.test_years is None:
            self.test_years = list(range(2015, 2025))

# Feature groups
FEATURE_GROUPS = {
    "price_action": [
        "open_to_vwap_pct",
        "or_high_to_vwap_pct",
        "or_low_to_vwap_pct",
        "or_range_pct",
        "prev_close_to_open_gap_pct",
    ],
    "volatility": [
        "atr_14",
        "atr_5",
        "atr_ratio_5_14",
        "or_atr_ratio",
        "prev_day_range_pct",
        "prev_day_body_pct",
    ],
    "momentum": [
        "rsi_14",
        "price_vs_sma_20",
        "price_vs_sma_50",
        "macd_hist",
        "adx_14",
    ],
    "volume": [
        "volume_ratio_vs_20d_avg",
        "or_volume_ratio",
    ],
    "time": [
        "day_of_week",
        "month",
        "days_since_month_start",
        "is_opex_week",
        "is_fomc_day",
    ],
    "market_regime": [
        "vix_level",
        "vix_change_1d",
        "spy_20d_return",
        "spy_5d_return",
    ],
}

# All features flattened
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Target labels
TARGET_LABELS = [
    "touch_t1_long",   # Price touched T1 above VWAP
    "touch_t2_long",
    "touch_t3_long",
    "touch_sl_long",   # Price touched SL below VWAP
    "touch_t1_short",  # Price touched T1 below VWAP
    "touch_t2_short",
    "touch_t3_short",
    "touch_sl_short",  # Price touched SL above VWAP
]

FIRST_TOUCH_CLASSES = [
    "t1_long", "t2_long", "t3_long", "sl_long",
    "t1_short", "t2_short", "t3_short", "sl_short",
    "none"
]
