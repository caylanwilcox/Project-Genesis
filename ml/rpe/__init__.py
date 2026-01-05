# Phase 1: Structural Context Layer
# Version 3.0

from .vwap import calculate_vwap
from .acceptance import check_acceptance, check_swing_acceptance
from .auction_state import classify_auction_state
from .levels import calculate_intraday_levels, calculate_swing_levels
from .failures import detect_failures
from .beware import generate_beware_alerts, aggregate_risk_level
from .compute import compute_intraday_context, compute_swing_context
