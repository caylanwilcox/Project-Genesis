"""
Northstar Phase Pipeline
========================
A 4-phase market structure analysis system.

Phase 1: TRUTH - RealityState (immutable market structure truth)
Phase 2: HEALTH_GATE - SignalHealthState (risk assessment)
Phase 3: DENSITY_CONTROL - SignalDensityState (spam/clustering control)
Phase 4: EXECUTION_PERMISSION - ExecutionState (final trade permission)

Principles:
- Phase 1 is immutable truth (no ML override)
- Phase 2/3 are gates (risk + spam control)
- Phase 4 is permission + framing (no entries/targets/stops)
- No prediction leakage between phases
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np


def to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.bool_, np.generic)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ============================================================
# ENUMS
# ============================================================

class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    BALANCED = "BALANCED"

class DominantTimeframe(Enum):
    INTRADAY = "INTRADAY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

class ConfidenceBand(Enum):
    NO_TRADE = "NO_TRADE"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    STRUCTURAL_EDGE = "STRUCTURAL_EDGE"

class AcceptanceStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"

class RangeState(Enum):
    TREND = "TREND"
    BALANCE = "BALANCE"
    FAILED_EXPANSION = "FAILED_EXPANSION"

class ExpansionQuality(Enum):
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    NONE = "NONE"

class Conviction(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Memory(Enum):
    PERSISTENT = "PERSISTENT"
    RESET = "RESET"

class GapBehavior(Enum):
    FILL = "FILL"
    HOLD = "HOLD"
    EXPAND = "EXPAND"

class HealthTier(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNSTABLE = "UNSTABLE"

class Throttle(Enum):
    OPEN = "OPEN"
    LIMITED = "LIMITED"
    BLOCKED = "BLOCKED"

class Bias(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class ExecutionMode(Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    MEAN_REVERSION = "MEAN_REVERSION"
    SCALP = "SCALP"
    NO_TRADE = "NO_TRADE"

class RiskState(Enum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    DEFENSIVE = "DEFENSIVE"


# ============================================================
# PHASE 1: REALITY STATE (TRUTH)
# ============================================================

@dataclass
class Acceptance:
    accepted: bool = False
    acceptance_strength: AcceptanceStrength = AcceptanceStrength.WEAK
    acceptance_reason: str = ""  # Specific reason WHY (e.g., "7/10 bars above $590.50 mid")
    failed_levels: List[str] = field(default_factory=list)

@dataclass
class Range:
    state: RangeState = RangeState.BALANCE
    rotation_complete: bool = False
    expansion_quality: ExpansionQuality = ExpansionQuality.NONE

@dataclass
class MTF:
    aligned: bool = False
    dominant_tf: DominantTimeframe = DominantTimeframe.INTRADAY
    conflict_tf: Optional[str] = None

@dataclass
class Participation:
    conviction: Conviction = Conviction.LOW
    effort_result_match: bool = False

@dataclass
class MemoryState:
    memory: Memory = Memory.RESET
    gap_behavior: Optional[GapBehavior] = None

@dataclass
class Failure:
    present: bool = False
    failure_types: List[str] = field(default_factory=list)

@dataclass
class KeyLevels:
    """Key price levels for support/resistance"""
    recent_high: float = 0.0
    recent_low: float = 0.0
    mid_point: float = 0.0
    pivot: float = 0.0
    pivot_r1: float = 0.0
    pivot_s1: float = 0.0
    current_price: float = 0.0
    today_open: float = 0.0
    prev_close: float = 0.0
    # Retest levels - where price will likely retest on range failure
    retest_high: float = 0.0  # Next resistance if range high fails
    retest_low: float = 0.0   # Next support if range low fails

@dataclass
class VolatilityExpansion:
    """Volatility expansion prediction"""
    probability: float = 0.0           # 0.0 - 1.0
    signal: str = "NONE"               # NONE, WEAK, MODERATE, STRONG
    expansion_likely: bool = False
    reasons: List[str] = field(default_factory=list)

@dataclass
class RealityState:
    """Phase 1 Output: Immutable market structure truth"""
    resolved: bool = False
    direction: Direction = Direction.BALANCED
    dominant_timeframe: DominantTimeframe = DominantTimeframe.INTRADAY
    confidence_band: ConfidenceBand = ConfidenceBand.NO_TRADE
    acceptance: Acceptance = field(default_factory=Acceptance)
    range: Range = field(default_factory=Range)
    mtf: MTF = field(default_factory=MTF)
    participation: Participation = field(default_factory=Participation)
    memory: MemoryState = field(default_factory=MemoryState)
    failure: Failure = field(default_factory=Failure)
    key_levels: KeyLevels = field(default_factory=KeyLevels)
    volatility_expansion: VolatilityExpansion = field(default_factory=VolatilityExpansion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resolved': to_python_type(self.resolved),
            'direction': self.direction.value,
            'dominant_timeframe': self.dominant_timeframe.value,
            'confidence_band': self.confidence_band.value,
            'acceptance': {
                'accepted': to_python_type(self.acceptance.accepted),
                'acceptance_strength': self.acceptance.acceptance_strength.value,
                'acceptance_reason': self.acceptance.acceptance_reason,
                'failed_levels': self.acceptance.failed_levels
            },
            'range': {
                'state': self.range.state.value,
                'rotation_complete': to_python_type(self.range.rotation_complete),
                'expansion_quality': self.range.expansion_quality.value
            },
            'mtf': {
                'aligned': to_python_type(self.mtf.aligned),
                'dominant_tf': self.mtf.dominant_tf.value,
                'conflict_tf': self.mtf.conflict_tf
            },
            'participation': {
                'conviction': self.participation.conviction.value,
                'effort_result_match': to_python_type(self.participation.effort_result_match)
            },
            'memory': {
                'memory': self.memory.memory.value,
                'gap_behavior': self.memory.gap_behavior.value if self.memory.gap_behavior else None
            },
            'failure': {
                'present': to_python_type(self.failure.present),
                'failure_types': self.failure.failure_types
            },
            'key_levels': {
                'recent_high': to_python_type(self.key_levels.recent_high),
                'recent_low': to_python_type(self.key_levels.recent_low),
                'mid_point': to_python_type(self.key_levels.mid_point),
                'pivot': to_python_type(self.key_levels.pivot),
                'pivot_r1': to_python_type(self.key_levels.pivot_r1),
                'pivot_s1': to_python_type(self.key_levels.pivot_s1),
                'current_price': to_python_type(self.key_levels.current_price),
                'today_open': to_python_type(self.key_levels.today_open),
                'prev_close': to_python_type(self.key_levels.prev_close),
                'retest_high': to_python_type(self.key_levels.retest_high),
                'retest_low': to_python_type(self.key_levels.retest_low)
            },
            'volatility_expansion': {
                'probability': to_python_type(self.volatility_expansion.probability),
                'signal': self.volatility_expansion.signal,
                'expansion_likely': to_python_type(self.volatility_expansion.expansion_likely),
                'reasons': self.volatility_expansion.reasons
            }
        }


# ============================================================
# PHASE 2: SIGNAL HEALTH STATE (HEALTH GATE)
# ============================================================

@dataclass
class SignalHealthState:
    """Phase 2 Output: Health assessment of the signal"""
    health_score: int = 0
    tier: HealthTier = HealthTier.UNSTABLE
    stand_down: bool = True
    reasons: List[str] = field(default_factory=list)

    # Individual dimension scores
    structural_integrity: int = 100
    time_persistence: int = 100
    volatility_alignment: int = 100
    participation_consistency: int = 100
    failure_risk: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'health_score': to_python_type(self.health_score),
            'tier': self.tier.value,
            'stand_down': to_python_type(self.stand_down),
            'reasons': self.reasons,
            'dimensions': {
                'structural_integrity': to_python_type(self.structural_integrity),
                'time_persistence': to_python_type(self.time_persistence),
                'volatility_alignment': to_python_type(self.volatility_alignment),
                'participation_consistency': to_python_type(self.participation_consistency),
                'failure_risk': to_python_type(self.failure_risk)
            }
        }


# ============================================================
# PHASE 3: SIGNAL DENSITY STATE (DENSITY CONTROL)
# ============================================================

@dataclass
class SignalDensityState:
    """Phase 3 Output: Spam/clustering control"""
    density_score: int = 0
    throttle: Throttle = Throttle.BLOCKED
    allowed_signals: int = 0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'density_score': to_python_type(self.density_score),
            'throttle': self.throttle.value,
            'allowed_signals': to_python_type(self.allowed_signals),
            'reasons': self.reasons
        }


# ============================================================
# PHASE 4: EXECUTION STATE (EXECUTION PERMISSION)
# ============================================================

@dataclass
class ExecutionState:
    """Phase 4 Output: Final execution permission"""
    allowed: bool = False
    bias: Bias = Bias.NEUTRAL
    execution_mode: ExecutionMode = ExecutionMode.NO_TRADE
    risk_state: RiskState = RiskState.DEFENSIVE
    invalidation_context: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'allowed': to_python_type(self.allowed),
            'bias': self.bias.value,
            'execution_mode': self.execution_mode.value,
            'risk_state': self.risk_state.value,
            'invalidation_context': self.invalidation_context
        }


# ============================================================
# PHASE 1 ENGINE: REALITY STATE CALCULATOR
# ============================================================

class Phase1Engine:
    """
    Phase 1: TRUTH - Calculates RealityState from raw market data.
    This is immutable truth - no ML can override these outputs.
    """

    def __init__(self):
        self.lookback_periods = {
            'intraday': 390,  # 1 trading day in minutes
            'daily': 20,      # 20 days
            'weekly': 12,     # 12 weeks
            'monthly': 6      # 6 months
        }

    def calculate(self, symbol: str, bars_1m: pd.DataFrame,
                  daily_bars: pd.DataFrame = None) -> RealityState:
        """
        Calculate RealityState from market data.

        Args:
            symbol: Ticker symbol
            bars_1m: 1-minute bars DataFrame with columns [open, high, low, close, volume, timestamp]
            daily_bars: Daily bars for multi-timeframe analysis
        """
        state = RealityState()

        if bars_1m is None or len(bars_1m) < 30:
            return state

        # Resolve the state
        state.resolved = True

        # 1. Determine Direction
        state.direction = self._calculate_direction(bars_1m)

        # 2. Determine Dominant Timeframe
        state.dominant_timeframe = self._calculate_dominant_timeframe(bars_1m, daily_bars)

        # 3. Calculate Acceptance
        state.acceptance = self._calculate_acceptance(bars_1m)

        # 4. Calculate Range State
        state.range = self._calculate_range(bars_1m)

        # 5. Calculate MTF Alignment
        state.mtf = self._calculate_mtf(bars_1m, daily_bars)

        # 6. Calculate Participation/Conviction
        state.participation = self._calculate_participation(bars_1m)

        # 7. Calculate Memory State
        state.memory = self._calculate_memory(bars_1m, daily_bars)

        # 8. Detect Failure Patterns
        state.failure = self._detect_failures(bars_1m)

        # 9. Calculate Key Price Levels
        state.key_levels = self._calculate_key_levels(bars_1m, daily_bars)

        # 10. Calculate Volatility Expansion Probability
        state.volatility_expansion = self._calculate_volatility_expansion(bars_1m, daily_bars)

        # 11. Determine Confidence Band
        state.confidence_band = self._calculate_confidence_band(state)

        return state

    def _calculate_direction(self, bars: pd.DataFrame) -> Direction:
        """Determine market direction from price action"""
        if len(bars) < 20:
            return Direction.BALANCED

        # Use multiple timeframes for direction
        close = bars['close'].values

        # Short-term (last 20 bars)
        short_sma = np.mean(close[-20:])
        # Medium-term (last 60 bars)
        medium_sma = np.mean(close[-60:]) if len(close) >= 60 else short_sma
        # Current price
        current = close[-1]

        # Higher highs / lower lows
        recent_high = np.max(bars['high'].values[-20:])
        recent_low = np.min(bars['low'].values[-20:])
        prior_high = np.max(bars['high'].values[-40:-20]) if len(bars) >= 40 else recent_high
        prior_low = np.min(bars['low'].values[-40:-20]) if len(bars) >= 40 else recent_low

        # Determine direction
        higher_high = recent_high > prior_high
        higher_low = recent_low > prior_low
        lower_high = recent_high < prior_high
        lower_low = recent_low < prior_low

        if higher_high and higher_low and current > short_sma:
            return Direction.UP
        elif lower_high and lower_low and current < short_sma:
            return Direction.DOWN
        else:
            return Direction.BALANCED

    def _calculate_dominant_timeframe(self, bars_1m: pd.DataFrame,
                                       daily_bars: pd.DataFrame = None) -> DominantTimeframe:
        """Determine which timeframe is driving price action"""
        # Calculate ATR-like volatility at different scales
        intraday_vol = self._calculate_volatility(bars_1m[-60:]) if len(bars_1m) >= 60 else 0

        if daily_bars is not None and len(daily_bars) >= 5:
            daily_vol = self._calculate_volatility(daily_bars[-5:])
            # Compare volatilities
            if daily_vol > intraday_vol * 2:
                return DominantTimeframe.DAILY

        return DominantTimeframe.INTRADAY

    def _calculate_acceptance(self, bars: pd.DataFrame) -> Acceptance:
        """Calculate price acceptance at key levels"""
        acceptance = Acceptance()

        if len(bars) < 30:
            return acceptance

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Find key levels (using simple pivot points)
        recent_high = np.max(high[-30:])
        recent_low = np.min(low[-30:])
        mid_point = (recent_high + recent_low) / 2
        current = close[-1]

        # Check if price has accepted above/below mid
        bars_above_mid = np.sum(close[-10:] > mid_point)
        bars_below_mid = np.sum(close[-10:] < mid_point)

        if bars_above_mid >= 7:
            acceptance.accepted = True
            acceptance.acceptance_strength = AcceptanceStrength.STRONG
            acceptance.acceptance_reason = f"{bars_above_mid}/10 bars above ${mid_point:.2f} mid → price accepted ABOVE range"
        elif bars_above_mid >= 5:
            acceptance.accepted = True
            acceptance.acceptance_strength = AcceptanceStrength.MODERATE
            acceptance.acceptance_reason = f"{bars_above_mid}/10 bars above ${mid_point:.2f} mid → building acceptance"
        elif bars_below_mid >= 7:
            acceptance.accepted = True
            acceptance.acceptance_strength = AcceptanceStrength.STRONG
            acceptance.acceptance_reason = f"{bars_below_mid}/10 bars below ${mid_point:.2f} mid → price accepted BELOW range"
        elif bars_below_mid >= 5:
            acceptance.accepted = True
            acceptance.acceptance_strength = AcceptanceStrength.MODERATE
            acceptance.acceptance_reason = f"{bars_below_mid}/10 bars below ${mid_point:.2f} mid → building acceptance"
        else:
            acceptance.acceptance_reason = f"Only {max(bars_above_mid, bars_below_mid)}/10 bars on one side of ${mid_point:.2f} mid → no clear acceptance"

        # Track failed levels with detailed price context
        if current < recent_high * 0.995 and np.max(close[-5:]) >= recent_high * 0.998:
            rejection_pct = ((recent_high - current) / current) * 100
            acceptance.failed_levels.append(
                f"Rejected at ${recent_high:.2f} resistance (now ${current:.2f}, -{rejection_pct:.2f}% off high)"
            )
        if current > recent_low * 1.005 and np.min(close[-5:]) <= recent_low * 1.002:
            bounce_pct = ((current - recent_low) / recent_low) * 100
            acceptance.failed_levels.append(
                f"Bounced off ${recent_low:.2f} support (now ${current:.2f}, +{bounce_pct:.2f}% off low)"
            )

        return acceptance

    def _calculate_range(self, bars: pd.DataFrame) -> Range:
        """Calculate range state (trend/balance/failed expansion)"""
        range_state = Range()

        if len(bars) < 30:
            return range_state

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Calculate range expansion
        range_30 = np.max(high[-30:]) - np.min(low[-30:])
        range_10 = np.max(high[-10:]) - np.min(low[-10:])

        # Determine if trending or in balance
        price_change_30 = (close[-1] - close[-30]) / close[-30] if len(close) >= 30 else 0

        if abs(price_change_30) > 0.005:  # >0.5% move
            range_state.state = RangeState.TREND
            range_state.expansion_quality = ExpansionQuality.CLEAN if range_10 > range_30 * 0.5 else ExpansionQuality.DIRTY
        elif range_10 < range_30 * 0.3:
            range_state.state = RangeState.BALANCE
            range_state.expansion_quality = ExpansionQuality.NONE
        else:
            # Check for failed expansion
            if range_10 > range_30 * 0.7 and abs(price_change_30) < 0.002:
                range_state.state = RangeState.FAILED_EXPANSION
                range_state.expansion_quality = ExpansionQuality.DIRTY

        # Check rotation
        recent_highs = high[-10:]
        recent_lows = low[-10:]
        range_state.rotation_complete = (np.argmax(recent_highs) != len(recent_highs)-1 and
                                          np.argmin(recent_lows) != len(recent_lows)-1)

        return range_state

    def _calculate_mtf(self, bars_1m: pd.DataFrame,
                       daily_bars: pd.DataFrame = None) -> MTF:
        """Calculate multi-timeframe alignment"""
        mtf = MTF()

        intraday_dir = self._calculate_direction(bars_1m)
        mtf.dominant_tf = DominantTimeframe.INTRADAY

        if daily_bars is not None and len(daily_bars) >= 5:
            daily_dir = self._calculate_direction(daily_bars)

            if intraday_dir == daily_dir:
                mtf.aligned = True
            else:
                mtf.aligned = False
                mtf.conflict_tf = f"DAILY={daily_dir.value} vs INTRADAY={intraday_dir.value}"

            # Determine dominant
            daily_vol = self._calculate_volatility(daily_bars[-5:])
            intraday_vol = self._calculate_volatility(bars_1m[-60:]) if len(bars_1m) >= 60 else 0

            if daily_vol > intraday_vol * 1.5:
                mtf.dominant_tf = DominantTimeframe.DAILY
        else:
            mtf.aligned = True  # Can't determine conflict without daily data

        return mtf

    def _calculate_participation(self, bars: pd.DataFrame) -> Participation:
        """Calculate market participation/conviction"""
        participation = Participation()

        if len(bars) < 20 or 'volume' not in bars.columns:
            return participation

        volume = bars['volume'].values
        close = bars['close'].values

        # Average volume
        avg_vol = np.mean(volume[-20:])
        recent_vol = np.mean(volume[-5:])

        # Volume conviction
        if recent_vol > avg_vol * 1.5:
            participation.conviction = Conviction.HIGH
        elif recent_vol > avg_vol:
            participation.conviction = Conviction.MEDIUM
        else:
            participation.conviction = Conviction.LOW

        # Effort vs Result
        price_move = abs(close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        # High volume should produce movement
        if vol_ratio > 1.3 and price_move > 0.002:
            participation.effort_result_match = True
        elif vol_ratio < 0.8 and price_move < 0.001:
            participation.effort_result_match = True
        elif vol_ratio > 1.3 and price_move < 0.001:
            participation.effort_result_match = False  # High effort, no result
        else:
            participation.effort_result_match = True

        return participation

    def _calculate_memory(self, bars_1m: pd.DataFrame,
                          daily_bars: pd.DataFrame = None) -> MemoryState:
        """Calculate market memory state"""
        memory = MemoryState()

        if daily_bars is None or len(daily_bars) < 2:
            return memory

        # Check for gap
        prev_close = daily_bars['close'].values[-2]
        today_open = daily_bars['open'].values[-1]
        gap_pct = (today_open - prev_close) / prev_close

        if abs(gap_pct) > 0.002:  # >0.2% gap
            current = bars_1m['close'].values[-1] if len(bars_1m) > 0 else today_open

            # Determine gap behavior
            if gap_pct > 0:  # Gap up
                if current < today_open:
                    memory.gap_behavior = GapBehavior.FILL
                elif current > today_open * 1.002:
                    memory.gap_behavior = GapBehavior.EXPAND
                else:
                    memory.gap_behavior = GapBehavior.HOLD
            else:  # Gap down
                if current > today_open:
                    memory.gap_behavior = GapBehavior.FILL
                elif current < today_open * 0.998:
                    memory.gap_behavior = GapBehavior.EXPAND
                else:
                    memory.gap_behavior = GapBehavior.HOLD

        # Memory persistence
        if daily_bars is not None and len(daily_bars) >= 3:
            # Check if recent days have similar direction
            recent_dirs = []
            for i in range(-3, 0):
                if daily_bars['close'].values[i] > daily_bars['open'].values[i]:
                    recent_dirs.append(1)
                else:
                    recent_dirs.append(-1)

            if all(d == recent_dirs[0] for d in recent_dirs):
                memory.memory = Memory.PERSISTENT
            else:
                memory.memory = Memory.RESET

        return memory

    def _detect_failures(self, bars: pd.DataFrame) -> Failure:
        """Detect failure patterns with specific price levels"""
        failure = Failure()

        if len(bars) < 20:
            return failure

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        current_price = close[-1]

        # Failed breakout detection
        recent_high = np.max(high[-10:])
        recent_low = np.min(low[-10:])
        prior_high = np.max(high[-20:-10]) if len(high) >= 20 else recent_high
        prior_low = np.min(low[-20:-10]) if len(low) >= 20 else recent_low

        # Failed high breakout - include price levels
        if recent_high > prior_high and current_price < prior_high:
            failure.present = True
            failure.failure_types.append(
                f"FAILED_HIGH_BREAKOUT @ ${recent_high:.2f} (rejected, now ${current_price:.2f})"
            )

        # Failed low breakout - include price levels
        if recent_low < prior_low and current_price > prior_low:
            failure.present = True
            failure.failure_types.append(
                f"FAILED_LOW_BREAKOUT @ ${recent_low:.2f} (rejected, now ${current_price:.2f})"
            )

        # Reversal candle patterns with price context
        if len(bars) >= 3:
            # Check for reversal after trend
            if high[-2] > high[-3] and close[-1] < low[-2]:
                failure.present = True
                failure.failure_types.append(
                    f"BEARISH_REVERSAL from ${high[-2]:.2f} high"
                )
            if low[-2] < low[-3] and close[-1] > high[-2]:
                failure.present = True
                failure.failure_types.append(
                    f"BULLISH_REVERSAL from ${low[-2]:.2f} low"
                )

        return failure

    def _calculate_key_levels(self, bars: pd.DataFrame,
                               daily_bars: pd.DataFrame = None) -> KeyLevels:
        """Calculate key support/resistance levels"""
        levels = KeyLevels()

        if len(bars) < 15:
            return levels

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Current price
        levels.current_price = round(float(close[-1]), 2)

        # Recent high/low from intraday bars (15 bar lookback = 15 minutes)
        levels.recent_high = round(float(np.max(high[-15:])), 2)
        levels.recent_low = round(float(np.min(low[-15:])), 2)
        levels.mid_point = round((levels.recent_high + levels.recent_low) / 2, 2)

        # Retest levels - where price will retest on range failure
        # Look at prior range (bars 30-15) to find next support/resistance
        if len(bars) >= 30:
            prior_high = round(float(np.max(high[-30:-15])), 2)
            prior_low = round(float(np.min(low[-30:-15])), 2)
            # If current range high breaks up, retest target is prior high
            # If current range low breaks down, retest target is prior low
            levels.retest_high = prior_high
            levels.retest_low = prior_low
        else:
            # Not enough data - use pivot levels as fallback
            levels.retest_high = levels.recent_high
            levels.retest_low = levels.recent_low

        # Daily pivots if daily data available
        if daily_bars is not None and len(daily_bars) >= 2:
            prev_day = daily_bars.iloc[-2]  # Previous day for pivot calculation
            today = daily_bars.iloc[-1]  # Today

            prev_high = float(prev_day['high'])
            prev_low = float(prev_day['low'])
            prev_close = float(prev_day['close'])
            today_open = float(today['open'])

            # Classic floor trader pivot points
            levels.pivot = round((prev_high + prev_low + prev_close) / 3, 2)
            levels.pivot_r1 = round(2 * levels.pivot - prev_low, 2)
            levels.pivot_s1 = round(2 * levels.pivot - prev_high, 2)
            levels.today_open = round(today_open, 2)
            levels.prev_close = round(prev_close, 2)

        return levels

    def _calculate_volatility_expansion(self, bars: pd.DataFrame,
                                        daily_bars: pd.DataFrame = None) -> VolatilityExpansion:
        """
        Predict probability of imminent volatility expansion.
        Uses ML model trained on compression -> expansion patterns.
        """
        expansion = VolatilityExpansion()

        try:
            from .volatility_expansion import predict_expansion
            result = predict_expansion(bars, daily_bars)
            expansion.probability = round(result.probability, 3)
            expansion.signal = result.signal.value
            expansion.expansion_likely = result.expansion_likely
            expansion.reasons = result.reasons
        except Exception as e:
            # Fallback to simple rule-based approach
            expansion.probability = 0.0
            expansion.signal = "NONE"
            expansion.expansion_likely = False
            expansion.reasons = [f"Model unavailable: {str(e)}"]

        return expansion

    def _calculate_confidence_band(self, state: RealityState) -> ConfidenceBand:
        """Determine overall confidence band"""
        score = 0

        # Acceptance contributes
        if state.acceptance.accepted:
            if state.acceptance.acceptance_strength == AcceptanceStrength.STRONG:
                score += 30
            elif state.acceptance.acceptance_strength == AcceptanceStrength.MODERATE:
                score += 20
            else:
                score += 10

        # MTF alignment contributes
        if state.mtf.aligned:
            score += 25

        # Range state contributes
        if state.range.state == RangeState.TREND:
            score += 20
        elif state.range.state == RangeState.BALANCE:
            score += 10

        # Participation contributes
        if state.participation.conviction == Conviction.HIGH:
            score += 15
        elif state.participation.conviction == Conviction.MEDIUM:
            score += 10

        if state.participation.effort_result_match:
            score += 10

        # Failure detracts
        if state.failure.present:
            score -= 30

        # Determine band
        if score >= 60:
            return ConfidenceBand.STRUCTURAL_EDGE
        elif score >= 30:
            return ConfidenceBand.CONTEXT_ONLY
        else:
            return ConfidenceBand.NO_TRADE

    def _calculate_volatility(self, bars: pd.DataFrame) -> float:
        """Calculate volatility (ATR-like)"""
        if len(bars) < 2:
            return 0

        high = bars['high'].values
        low = bars['low'].values
        close = bars['close'].values

        tr = np.maximum(high[1:] - low[1:],
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        return np.mean(tr) if len(tr) > 0 else 0


# ============================================================
# PHASE 2 ENGINE: SIGNAL HEALTH SCORER
# ============================================================

class Phase2Engine:
    """
    Phase 2: HEALTH_GATE - Scores signal health across multiple dimensions.
    Deterministic scoring with configurable weights.
    """

    def __init__(self):
        self.weights = {
            'structural_integrity': 0.30,
            'time_persistence': 0.15,
            'volatility_alignment': 0.15,
            'participation_consistency': 0.20,
            'failure_risk': 0.20
        }

    def calculate(self, reality: RealityState,
                  recent_snapshots: List[RealityState] = None,
                  time_since_acceptance_minutes: int = 0,
                  stall_flags: bool = False,
                  vol_dislocation: bool = False) -> SignalHealthState:
        """
        Calculate SignalHealthState from RealityState.
        """
        health = SignalHealthState()
        reasons = []

        # 1. Structural Integrity (start at 100, subtract penalties)
        struct_score = 100
        if not reality.acceptance.accepted:
            struct_score -= 25
            reasons.append("No acceptance")
        if not reality.mtf.aligned:
            struct_score -= 15
            reasons.append("MTF conflict")
        if reality.failure.present:
            struct_score -= 30
            reasons.append("Failure signals present")
        if reality.range.state == RangeState.FAILED_EXPANSION:
            struct_score -= 20
            reasons.append("Failed expansion")
        health.structural_integrity = max(0, struct_score)

        # 2. Time Persistence
        time_score = 100
        if stall_flags:
            time_score -= 20
            reasons.append("Stall detected")
        if time_since_acceptance_minutes > 60:
            time_score -= 10
            reasons.append("Long time since acceptance")
        health.time_persistence = max(0, time_score)

        # 3. Volatility Alignment
        vol_score = 100
        if vol_dislocation:
            vol_score -= 25
            reasons.append("Volatility dislocation")
        health.volatility_alignment = max(0, vol_score)

        # 4. Participation Consistency
        part_score = 100
        if reality.participation.conviction == Conviction.LOW:
            part_score -= 15
            reasons.append("Low conviction")
        if not reality.participation.effort_result_match:
            part_score -= 20
            reasons.append("Effort/result mismatch")
        health.participation_consistency = max(0, part_score)

        # 5. Failure Risk
        fail_score = 100
        if reality.failure.present:
            fail_score -= 40
            reasons.append("Active failure patterns")
        health.failure_risk = max(0, fail_score)

        # Aggregate with weights
        health.health_score = int(
            health.structural_integrity * self.weights['structural_integrity'] +
            health.time_persistence * self.weights['time_persistence'] +
            health.volatility_alignment * self.weights['volatility_alignment'] +
            health.participation_consistency * self.weights['participation_consistency'] +
            health.failure_risk * self.weights['failure_risk']
        )

        # Determine tier
        if health.health_score >= 75:
            health.tier = HealthTier.HEALTHY
            health.stand_down = False
        elif health.health_score >= 45:
            health.tier = HealthTier.DEGRADED
            health.stand_down = False
        else:
            health.tier = HealthTier.UNSTABLE
            health.stand_down = True

        health.reasons = reasons
        return health


# ============================================================
# PHASE 3 ENGINE: SIGNAL DENSITY CONTROLLER
# ============================================================

class Phase3Engine:
    """
    Phase 3: DENSITY_CONTROL - Controls signal spam and clustering.
    """

    def __init__(self):
        self.signal_history: List[Dict] = []

    def calculate(self, reality: RealityState,
                  health: SignalHealthState,
                  signals_last_10m: int = 0,
                  distinct_level_count: int = 0,
                  active_timeframes_firing: List[str] = None) -> SignalDensityState:
        """
        Calculate SignalDensityState for spam control.
        """
        density = SignalDensityState()
        reasons = []

        density_score = 100

        # 1. Clustering check
        if signals_last_10m > 3 and distinct_level_count <= 1:
            density_score -= 40
            reasons.append("Same-level spam")

        # 2. Timeframe saturation
        if active_timeframes_firing and len(active_timeframes_firing) >= 3:
            density_score -= 30
            reasons.append("Too many TFs firing")

        # 3. Regime compatibility
        if reality.range.state == RangeState.BALANCE and signals_last_10m > 4:
            density_score -= 30
            reasons.append("Noise in balance")

        density.density_score = max(0, density_score)
        density.reasons = reasons

        # Determine throttle
        if density.density_score >= 70:
            density.throttle = Throttle.OPEN
            density.allowed_signals = 999
        elif density.density_score >= 40:
            density.throttle = Throttle.LIMITED
            density.allowed_signals = 1
        else:
            density.throttle = Throttle.BLOCKED
            density.allowed_signals = 0

        return density


# ============================================================
# PHASE 4 ENGINE: EXECUTION PERMISSION
# ============================================================

class Phase4Engine:
    """
    Phase 4: EXECUTION_PERMISSION - Final gate for trade execution.
    """

    def calculate(self, reality: RealityState,
                  health: SignalHealthState,
                  density: SignalDensityState) -> ExecutionState:
        """
        Calculate ExecutionState - final permission decision.
        """
        execution = ExecutionState()
        invalidation = []

        # Eligibility Gate - DENY if any condition met
        deny = False

        if reality.confidence_band != ConfidenceBand.STRUCTURAL_EDGE:
            deny = True
            invalidation.append(f"Confidence band: {reality.confidence_band.value}")

        if health.stand_down:
            deny = True
            invalidation.append("Health stand_down active")

        if density.throttle == Throttle.BLOCKED:
            deny = True
            invalidation.append("Density throttle BLOCKED")

        execution.invalidation_context = invalidation

        if deny:
            execution.allowed = False
            execution.execution_mode = ExecutionMode.NO_TRADE
            execution.bias = Bias.NEUTRAL
            execution.risk_state = RiskState.DEFENSIVE
            return execution

        # Passed gate - determine framing
        execution.allowed = True

        # Bias from direction
        if reality.direction == Direction.UP:
            execution.bias = Bias.LONG
        elif reality.direction == Direction.DOWN:
            execution.bias = Bias.SHORT
        else:
            execution.bias = Bias.NEUTRAL

        # Execution mode
        if reality.range.state == RangeState.TREND and health.tier == HealthTier.HEALTHY:
            execution.execution_mode = ExecutionMode.TREND_CONTINUATION
        elif reality.range.state == RangeState.BALANCE and health.tier != HealthTier.UNSTABLE:
            execution.execution_mode = ExecutionMode.MEAN_REVERSION
        else:
            execution.execution_mode = ExecutionMode.SCALP

        # Risk posture
        if health.tier == HealthTier.HEALTHY and density.throttle == Throttle.OPEN:
            execution.risk_state = RiskState.NORMAL
        elif health.tier == HealthTier.DEGRADED or density.throttle == Throttle.LIMITED:
            execution.risk_state = RiskState.REDUCED
        else:
            execution.risk_state = RiskState.DEFENSIVE

        # Add invalidation context
        invalidation.append("Break acceptance")
        invalidation.append("MTF conflict")
        invalidation.append("Failure pattern activation")
        execution.invalidation_context = invalidation

        return execution


# ============================================================
# FULL PIPELINE
# ============================================================

class NorthstarPipeline:
    """
    Complete Northstar Phase Pipeline.
    Orchestrates all 4 phases.
    """

    def __init__(self):
        self.phase1 = Phase1Engine()
        self.phase2 = Phase2Engine()
        self.phase3 = Phase3Engine()
        self.phase4 = Phase4Engine()

    def run(self, symbol: str, bars_1m: pd.DataFrame,
            daily_bars: pd.DataFrame = None,
            signals_last_10m: int = 0,
            time_since_acceptance_minutes: int = 0) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Dict with phase1, phase2, phase3, phase4 outputs
        """
        # Phase 1: Truth
        reality = self.phase1.calculate(symbol, bars_1m, daily_bars)

        # Phase 2: Health
        health = self.phase2.calculate(
            reality,
            time_since_acceptance_minutes=time_since_acceptance_minutes
        )

        # Phase 3: Density
        density = self.phase3.calculate(
            reality, health,
            signals_last_10m=signals_last_10m
        )

        # Phase 4: Execution
        execution = self.phase4.calculate(reality, health, density)

        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'phase1': reality.to_dict(),
            'phase2': health.to_dict(),
            'phase3': density.to_dict(),
            'phase4': execution.to_dict()
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def analyze_market(symbol: str, bars_1m: pd.DataFrame,
                   daily_bars: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Convenience function to run the full Northstar pipeline.
    """
    pipeline = NorthstarPipeline()
    return pipeline.run(symbol, bars_1m, daily_bars)
