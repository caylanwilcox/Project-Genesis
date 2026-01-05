"""
Centralized Configuration for ML Training
Ensures consistent train/test periods across all models to prevent data leakage.
"""

# =============================================================================
# STANDARD TRAINING/TESTING PERIODS
# =============================================================================
# These periods MUST be used by ALL training scripts to ensure:
# 1. No data leakage (test data never appears in training)
# 2. Valid model comparison (all models tested on same period)
# 3. Consistent backtest results
#
# IMPORTANT: Do NOT modify these dates without updating ALL training scripts

STANDARD_PERIODS = {
    'train_start': '2000-01-01',
    'train_end': '2024-12-31',
    'test_start': '2025-01-01',
    'test_end': '2025-12-31',
}

# Convenience exports
TRAIN_START = STANDARD_PERIODS['train_start']
TRAIN_END = STANDARD_PERIODS['train_end']
TEST_START = STANDARD_PERIODS['test_start']
TEST_END = STANDARD_PERIODS['test_end']

# =============================================================================
# DATA AVAILABILITY NOTES
# =============================================================================
# Some tickers have limited historical data. Handle gracefully:
#
# | Ticker | Earliest Data |
# |--------|---------------|
# | SPY    | 1993          |
# | QQQ    | 1999          |
# | IWM    | 2000          |
# | UVXY   | 2011          |
#
# Training scripts should use TRAIN_START but handle missing data gracefully
# for tickers that don't go back that far.

# =============================================================================
# TICKERS
# =============================================================================
DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM']
INTRADAY_TICKERS = ['SPY', 'QQQ', 'IWM']
SWING_TICKERS = ['SPY', 'QQQ', 'IWM']

# =============================================================================
# MODEL SETTINGS
# =============================================================================
RANDOM_STATE = 42
N_JOBS = -1  # Use all cores

# XGBoost defaults
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
}

# Random Forest defaults
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,
}

# =============================================================================
# VALIDATION
# =============================================================================
def validate_periods():
    """Validate that periods are correctly configured."""
    from datetime import datetime

    train_start = datetime.strptime(TRAIN_START, '%Y-%m-%d')
    train_end = datetime.strptime(TRAIN_END, '%Y-%m-%d')
    test_start = datetime.strptime(TEST_START, '%Y-%m-%d')
    test_end = datetime.strptime(TEST_END, '%Y-%m-%d')

    assert train_end < test_start, "Train end must be before test start (data leakage!)"
    assert train_start < train_end, "Train start must be before train end"
    assert test_start < test_end, "Test start must be before test end"

    return True


if __name__ == "__main__":
    validate_periods()
    print("Standard Training Periods:")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Test:  {TEST_START} to {TEST_END}")
