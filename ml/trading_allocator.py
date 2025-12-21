"""
Trading Allocator V1 - EV-Optimized Capital Deployment

Key principles:
1. EV-weighted signals (not just accuracy)
2. Magnitude classification (Target C)
3. Conditional position sizing
4. Volatility filtering
5. Capital concentration
6. Proper metrics (R-multiple, profit per bucket)

This transforms the V6 direction model into an actual trading system.
"""

import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


# ============================================================
# SECTION 1: EV-BASED SIGNAL SCORING
# ============================================================

# Historical performance by probability bucket (from backtest)
# Format: (win_rate, avg_return_when_correct, avg_loss_when_wrong)
PROBABILITY_BUCKETS = {
    'very_strong_bull': {'range': (0.90, 1.00), 'win_rate': 1.00, 'avg_win': 0.0045, 'avg_loss': 0.0025},
    'strong_bull':      {'range': (0.70, 0.90), 'win_rate': 0.80, 'avg_win': 0.0035, 'avg_loss': 0.0030},
    'moderate_bull':    {'range': (0.60, 0.70), 'win_rate': 0.65, 'avg_win': 0.0025, 'avg_loss': 0.0030},
    'neutral':          {'range': (0.40, 0.60), 'win_rate': 0.50, 'avg_win': 0.0020, 'avg_loss': 0.0020},
    'moderate_bear':    {'range': (0.30, 0.40), 'win_rate': 0.65, 'avg_win': 0.0025, 'avg_loss': 0.0030},
    'strong_bear':      {'range': (0.10, 0.30), 'win_rate': 0.80, 'avg_win': 0.0035, 'avg_loss': 0.0030},
    'very_strong_bear': {'range': (0.00, 0.10), 'win_rate': 1.00, 'avg_win': 0.0045, 'avg_loss': 0.0025},
}


def get_probability_bucket(prob: float) -> str:
    """Classify probability into bucket"""
    if prob >= 0.90:
        return 'very_strong_bull'
    elif prob >= 0.70:
        return 'strong_bull'
    elif prob >= 0.60:
        return 'moderate_bull'
    elif prob >= 0.40:
        return 'neutral'
    elif prob >= 0.30:
        return 'moderate_bear'
    elif prob >= 0.10:
        return 'strong_bear'
    else:
        return 'very_strong_bear'


def calculate_ev(prob: float, direction: str = 'bull') -> float:
    """
    Calculate Expected Value for a signal

    EV = (win_rate × avg_win) - ((1 - win_rate) × avg_loss)

    This replaces simple accuracy-weighted voting.
    """
    bucket_name = get_probability_bucket(prob)
    bucket = PROBABILITY_BUCKETS[bucket_name]

    # For bear signals, we're betting on close < open
    if direction == 'bear' or prob < 0.5:
        # Invert probability for bear calculation
        effective_prob = 1 - prob
    else:
        effective_prob = prob

    win_rate = bucket['win_rate']
    avg_win = bucket['avg_win']
    avg_loss = bucket['avg_loss']

    ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    return ev


# ============================================================
# SECTION 2: MAGNITUDE CLASSIFICATION (TARGET C)
# ============================================================

def classify_magnitude(volatility_5d: float, gap: float, first_hour_return: float) -> str:
    """
    Classify expected move size

    Buckets:
    - small: < 0.3% expected
    - medium: 0.3-0.7% expected
    - large: > 0.7% expected

    Uses volatility regime and early price action.
    """
    # Combine signals for magnitude estimate
    vol_signal = volatility_5d * 100  # Convert to percentage
    gap_signal = abs(gap) * 100
    momentum_signal = abs(first_hour_return) * 100

    # Weighted estimate
    expected_move = (vol_signal * 0.5) + (gap_signal * 0.3) + (momentum_signal * 0.2)

    if expected_move < 0.3:
        return 'small'
    elif expected_move < 0.7:
        return 'medium'
    else:
        return 'large'


def get_magnitude_multiplier(magnitude: str) -> float:
    """Position size multiplier based on expected magnitude"""
    return {
        'small': 0.5,   # Half size on grind days
        'medium': 1.0,  # Normal size
        'large': 1.5,   # Increase on large-move days
    }.get(magnitude, 1.0)


# ============================================================
# SECTION 3: CONDITIONAL POSITION SIZING (CPS)
# ============================================================

def get_time_multiplier(hour: int) -> float:
    """
    Time-of-day position multiplier

    Based on backtest accuracy patterns:
    - 12-1 PM: 1.0 (baseline)
    - 1-3 PM: 1.2 (peak accuracy zone)
    - 3-4 PM: 0.8 (protect gains, avoid EOD noise)
    """
    if hour < 12:
        return 0.7  # Early session is less reliable
    elif hour == 12:
        return 1.0
    elif hour in [13, 14]:
        return 1.2  # Peak accuracy
    elif hour == 15:
        return 0.8  # Protect gains
    else:
        return 0.5  # After hours


def get_signal_agreement_multiplier(prob_a: float, prob_b: float) -> float:
    """
    Multiplier when Target A and Target B agree

    If both signals point same direction with high confidence,
    increase position size.
    """
    # Both bullish
    if prob_a > 0.6 and prob_b > 0.6:
        return 1.25
    # Both bearish
    elif prob_a < 0.4 and prob_b < 0.4:
        return 1.25
    # Disagreement
    elif (prob_a > 0.6 and prob_b < 0.4) or (prob_a < 0.4 and prob_b > 0.6):
        return 0.5
    # One neutral
    else:
        return 1.0


def calculate_position_size(
    base_size: float,
    prob_a: float,
    prob_b: float,
    hour: int,
    magnitude: str
) -> float:
    """
    Final Position Size = Base × Probability Factor × Agreement × Time × Magnitude
    """
    # Probability factor (0.5 to 1.0 based on distance from 0.5)
    prob_factor = 0.5 + abs(prob_a - 0.5)

    # Get multipliers
    agreement = get_signal_agreement_multiplier(prob_a, prob_b)
    time_mult = get_time_multiplier(hour)
    mag_mult = get_magnitude_multiplier(magnitude)

    final_size = base_size * prob_factor * agreement * time_mult * mag_mult

    # Cap at 2x base
    return min(final_size, base_size * 2.0)


# ============================================================
# SECTION 4: VOLATILITY FILTER
# ============================================================

def should_trade(
    current_range: float,
    avg_range_20d: float,
    pre_1pm_range: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Volatility gate to filter out dead capital situations

    Returns: (should_trade, size_multiplier, reason)
    """
    # Calculate range ratio
    if avg_range_20d > 0:
        range_ratio = current_range / avg_range_20d
    else:
        range_ratio = 1.0

    # Too compressed - no edge
    if range_ratio < 0.6:
        return False, 0.0, "Range too compressed (< 60% of 20d avg)"

    # Pre-1 PM compression check
    if pre_1pm_range is not None and avg_range_20d > 0:
        pre_range_ratio = pre_1pm_range / avg_range_20d
        if pre_range_ratio < 0.4:
            return True, 0.5, "Pre-1PM compression (half size)"

    # Very extended - reduce size to protect
    if range_ratio > 1.5:
        return True, 0.75, "Extended range (reduced size)"

    # Normal conditions
    return True, 1.0, "Normal volatility"


# ============================================================
# SECTION 5: CAPITAL CONCENTRATION
# ============================================================

def select_best_ticker(signals: Dict[str, Dict]) -> Tuple[str, Dict]:
    """
    Select the single best ticker to trade

    Criteria (in order):
    1. Highest EV signal
    2. Cleanest agreement (A & B aligned)
    3. Structural advantage (IWM late session bias)

    Returns: (ticker, signal_data)
    """
    scored = []

    for ticker, data in signals.items():
        prob_a = data['prob_a']
        prob_b = data['prob_b']
        hour = data['hour']

        # Calculate EV
        ev = calculate_ev(prob_a)

        # Agreement bonus
        agreement = get_signal_agreement_multiplier(prob_a, prob_b)

        # IWM structural bonus (from backtest: 100% late session)
        ticker_bonus = 1.2 if ticker == 'IWM' and hour >= 12 else 1.0

        # Time bonus
        time_mult = get_time_multiplier(hour)

        # Composite score
        score = ev * agreement * ticker_bonus * time_mult * 1000

        scored.append({
            'ticker': ticker,
            'score': score,
            'ev': ev,
            'data': data
        })

    # Sort by score
    scored.sort(key=lambda x: x['score'], reverse=True)

    if scored:
        best = scored[0]
        return best['ticker'], best['data']

    return None, None


# ============================================================
# SECTION 6: TRADING METRICS
# ============================================================

class TradingMetrics:
    """Track proper trading metrics"""

    def __init__(self):
        self.trades = []
        self.r_unit = 0.0025  # 0.25% as 1R

    def add_trade(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        prob_bucket: str,
        hour: int,
        date: str
    ):
        """Record a trade"""
        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        r_multiple = pnl / self.r_unit

        self.trades.append({
            'date': date,
            'ticker': ticker,
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'size': size,
            'pnl': pnl,
            'r_multiple': r_multiple,
            'prob_bucket': prob_bucket,
            'hour': hour,
            'won': pnl > 0
        })

    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)

        # Overall metrics
        total_r = df['r_multiple'].sum()
        avg_r = df['r_multiple'].mean()
        win_rate = df['won'].mean()

        # Per bucket metrics
        bucket_stats = df.groupby('prob_bucket').agg({
            'r_multiple': ['mean', 'sum', 'count'],
            'won': 'mean'
        }).round(3)

        # Per ticker metrics
        ticker_stats = df.groupby('ticker').agg({
            'r_multiple': ['mean', 'sum', 'count'],
            'won': 'mean'
        }).round(3)

        # Drawdown
        cumulative = df['r_multiple'].cumsum()
        peak = cumulative.expanding().max()
        drawdown = cumulative - peak
        max_drawdown = drawdown.min()

        return {
            'total_r': total_r,
            'avg_r_per_trade': avg_r,
            'win_rate': win_rate,
            'total_trades': len(df),
            'max_drawdown_r': max_drawdown,
            'bucket_stats': bucket_stats,
            'ticker_stats': ticker_stats,
            'profit_factor': abs(df[df['r_multiple'] > 0]['r_multiple'].sum() /
                                 df[df['r_multiple'] < 0]['r_multiple'].sum()) if (df['r_multiple'] < 0).any() else float('inf')
        }


# ============================================================
# SECTION 7: FULL ALLOCATION DECISION
# ============================================================

def generate_allocation(
    ticker: str,
    prob_a: float,
    prob_b: float,
    hour: int,
    volatility_5d: float,
    gap: float,
    first_hour_return: float,
    current_range: float,
    avg_range_20d: float,
    base_capital: float = 100000,
    max_position_pct: float = 0.20
) -> Dict:
    """
    Generate complete allocation decision

    Combines all modules into single recommendation.
    """
    # 1. Volatility filter
    can_trade, vol_mult, vol_reason = should_trade(current_range, avg_range_20d)

    if not can_trade:
        return {
            'action': 'NO_TRADE',
            'reason': vol_reason,
            'position_size': 0,
            'confidence': 0
        }

    # 2. Get probability bucket
    bucket = get_probability_bucket(prob_a)

    # 3. Check if neutral zone (tighter: 45-55%)
    if 0.45 <= prob_a <= 0.55:
        return {
            'action': 'NO_TRADE',
            'reason': 'Neutral probability zone (45-55%)',
            'position_size': 0,
            'confidence': 0
        }

    # 4. Calculate EV
    ev = calculate_ev(prob_a)

    # 5. Classify magnitude
    magnitude = classify_magnitude(volatility_5d, gap, first_hour_return)

    # 6. Calculate position size
    base_size = base_capital * max_position_pct
    position_size = calculate_position_size(base_size, prob_a, prob_b, hour, magnitude)
    position_size *= vol_mult  # Apply volatility adjustment

    # 7. Determine direction
    direction = 'LONG' if prob_a > 0.5 else 'SHORT'

    # 8. Confidence score (0-100)
    confidence = int(abs(prob_a - 0.5) * 200)

    # 9. Build recommendation
    return {
        'action': direction,
        'ticker': ticker,
        'position_size': round(position_size, 2),
        'position_pct': round(position_size / base_capital * 100, 1),
        'confidence': confidence,
        'ev': round(ev * 10000, 2),  # In basis points
        'probability_bucket': bucket,
        'magnitude': magnitude,
        'hour': hour,
        'multipliers': {
            'volatility': vol_mult,
            'agreement': get_signal_agreement_multiplier(prob_a, prob_b),
            'time': get_time_multiplier(hour),
            'magnitude': get_magnitude_multiplier(magnitude)
        },
        'targets': {
            'stop_loss_pct': -0.25,  # Fixed fractional stop
            'take_profit_pct': 0.50,  # 2R target
            'trailing_activation': 0.25  # Start trailing after 1R
        },
        'reason': f"{bucket.replace('_', ' ').title()} signal, {magnitude} magnitude day"
    }


# ============================================================
# SECTION 8: LIVE SIGNAL GENERATION
# ============================================================

def load_model(ticker: str) -> Dict:
    """Load V6 model for ticker"""
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def generate_live_signals(base_capital: float = 100000) -> Dict:
    """
    Generate live trading signals for all tickers

    Returns allocation recommendation.
    """
    signals = {}

    for ticker in ['SPY', 'QQQ', 'IWM']:
        try:
            model_data = load_model(ticker)
            # Would fetch live data and generate prediction here
            # For now, return structure
            signals[ticker] = {
                'model_loaded': True,
                'version': model_data.get('version', 'V6')
            }
        except Exception as e:
            signals[ticker] = {
                'error': str(e)
            }

    return signals


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_allocation():
    """Example of how to use the allocator"""

    print("="*70)
    print("  TRADING ALLOCATOR - EXAMPLE")
    print("="*70)

    # Example scenario: QQQ at 2 PM with strong bull signal
    allocation = generate_allocation(
        ticker='QQQ',
        prob_a=0.92,  # Strong bullish
        prob_b=0.78,  # Also bullish (agreement)
        hour=14,       # 2 PM (peak zone)
        volatility_5d=0.012,
        gap=0.003,
        first_hour_return=0.004,
        current_range=0.008,
        avg_range_20d=0.010,
        base_capital=100000
    )

    print("\n  SCENARIO: QQQ at 2:00 PM")
    print("  Probability A: 92%, Probability B: 78%")
    print("-"*50)
    print(f"\n  ACTION: {allocation['action']}")
    print(f"  Position Size: ${allocation['position_size']:,.0f}")
    print(f"  Position %: {allocation['position_pct']}%")
    print(f"  Confidence: {allocation['confidence']}%")
    print(f"  Expected Value: {allocation['ev']} bps")
    print(f"  Bucket: {allocation['probability_bucket']}")
    print(f"  Magnitude: {allocation['magnitude']}")

    print("\n  MULTIPLIERS:")
    for k, v in allocation['multipliers'].items():
        print(f"    {k}: {v}x")

    print("\n  TARGETS:")
    print(f"    Stop Loss: {allocation['targets']['stop_loss_pct']}%")
    print(f"    Take Profit: {allocation['targets']['take_profit_pct']}%")
    print(f"    Trailing After: {allocation['targets']['trailing_activation']}%")

    # Example 2: Low confidence scenario
    print("\n" + "="*70)
    print("  SCENARIO: SPY at 10 AM with weak signal")
    print("-"*50)

    allocation2 = generate_allocation(
        ticker='SPY',
        prob_a=0.55,  # Weak
        prob_b=0.48,  # Neutral
        hour=10,
        volatility_5d=0.008,
        gap=0.001,
        first_hour_return=0.002,
        current_range=0.004,
        avg_range_20d=0.010,
        base_capital=100000
    )

    print(f"\n  ACTION: {allocation2['action']}")
    print(f"  Reason: {allocation2.get('reason', 'N/A')}")

    # Example 3: Capital concentration
    print("\n" + "="*70)
    print("  CAPITAL CONCENTRATION EXAMPLE")
    print("-"*50)

    all_signals = {
        'SPY': {'prob_a': 0.75, 'prob_b': 0.68, 'hour': 14},
        'QQQ': {'prob_a': 0.82, 'prob_b': 0.79, 'hour': 14},
        'IWM': {'prob_a': 0.94, 'prob_b': 0.88, 'hour': 14}
    }

    best_ticker, best_data = select_best_ticker(all_signals)
    print(f"\n  Best Ticker: {best_ticker}")
    print(f"  Prob A: {best_data['prob_a']:.0%}")
    print(f"  Prob B: {best_data['prob_b']:.0%}")
    print("  (Concentrating capital here instead of splitting)")


if __name__ == '__main__':
    example_allocation()
