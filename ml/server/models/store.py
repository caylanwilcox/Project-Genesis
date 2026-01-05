"""
Model Storage

Global model dictionaries - loaded once at startup.
"""

# V6 time-split models (primary production models)
intraday_v6_models = {}  # ticker -> V6 model data

# V6 SWING models (multi-day predictions)
swing_v6_models = {}  # ticker -> V6 SWING model data (5d, 10d)
swing_3d_models = {}  # ticker -> 3-day swing model data
swing_1d_models = {}  # ticker -> 1-day swing model data

# Legacy models (kept for backwards compatibility)
models = {}              # ticker -> FVG model
combined_model = None    # Combined FVG model
daily_models = {}        # ticker -> daily direction model
highlow_models = {}      # ticker -> high/low model
shrinking_models = {}    # ticker -> shrinking range model
regime_models = {}       # ticker -> volatility regime model
intraday_models = {}     # ticker -> intraday session model
target_models = {}       # ticker -> target refinement model
enhanced_v3_models = {}  # ticker -> enhanced v3 model
