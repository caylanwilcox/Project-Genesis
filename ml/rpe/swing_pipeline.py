"""
RPE SWING Pipeline
==================
Northstar-style 4-phase analysis for swing trades using daily/weekly timeframes.

Phase 1: REALITY (Market Structure Truth)
- Weekly trend direction
- Daily trend alignment
- Support/resistance levels

Phase 2: HEALTH (Signal Quality)
- Trend consistency across timeframes
- Volume confirmation
- Failure risk assessment

Phase 3: DENSITY (Signal Filtering)
- Avoid choppy periods
- Wait for clean setups

Phase 4: EXECUTION (Trade Permission)
- Bias: LONG / SHORT / NEUTRAL
- Mode: TREND / MEAN_REVERSION / NO_TRADE
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np


def to_python_type(val):
    """Convert numpy types to Python native types."""
    if isinstance(val, (np.bool_, np.generic)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ============================================================
# ENUMS
# ============================================================

class SwingDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class SwingTimeframe(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

class SwingTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"

class SwingConfidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NO_TRADE = "NO_TRADE"

class SwingMode(Enum):
    TREND_FOLLOW = "TREND_FOLLOW"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    NO_TRADE = "NO_TRADE"

class SwingHealth(Enum):
    HEALTHY = "HEALTHY"
    MIXED = "MIXED"
    WEAK = "WEAK"

class SwingBias(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


# ============================================================
# PHASE 1: SWING REALITY STATE
# ============================================================

@dataclass
class SwingLevels:
    """Key swing trade levels"""
    weekly_high: float = 0.0
    weekly_low: float = 0.0
    weekly_mid: float = 0.0
    daily_high_5d: float = 0.0
    daily_low_5d: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    current_price: float = 0.0
    atr_14: float = 0.0

@dataclass
class TrendAlignment:
    """Multi-timeframe trend alignment"""
    weekly_trend: SwingTrend = SwingTrend.SIDEWAYS
    daily_trend: SwingTrend = SwingTrend.SIDEWAYS
    aligned: bool = False
    conflict_reason: str = ""

@dataclass
class SwingMomentum:
    """Momentum indicators"""
    rsi_14: float = 50.0
    weekly_rsi: float = 50.0
    macd_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    momentum_5d: float = 0.0
    momentum_10d: float = 0.0

@dataclass
class SwingVolume:
    """Volume analysis"""
    volume_trend: str = "NORMAL"  # INCREASING, DECREASING, NORMAL
    volume_confirmation: bool = False
    avg_volume_20d: float = 0.0

@dataclass
class SwingRealityState:
    """Phase 1 Output: Swing market structure truth"""
    resolved: bool = False
    direction: SwingDirection = SwingDirection.NEUTRAL
    dominant_timeframe: SwingTimeframe = SwingTimeframe.DAILY
    confidence: SwingConfidence = SwingConfidence.NO_TRADE
    levels: SwingLevels = field(default_factory=SwingLevels)
    trend_alignment: TrendAlignment = field(default_factory=TrendAlignment)
    momentum: SwingMomentum = field(default_factory=SwingMomentum)
    volume: SwingVolume = field(default_factory=SwingVolume)
    structure_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resolved': to_python_type(self.resolved),
            'direction': self.direction.value,
            'dominant_timeframe': self.dominant_timeframe.value,
            'confidence': self.confidence.value,
            'levels': {
                'weekly_high': to_python_type(self.levels.weekly_high),
                'weekly_low': to_python_type(self.levels.weekly_low),
                'weekly_mid': to_python_type(self.levels.weekly_mid),
                'daily_high_5d': to_python_type(self.levels.daily_high_5d),
                'daily_low_5d': to_python_type(self.levels.daily_low_5d),
                'sma_20': to_python_type(self.levels.sma_20),
                'sma_50': to_python_type(self.levels.sma_50),
                'current_price': to_python_type(self.levels.current_price),
                'atr_14': to_python_type(self.levels.atr_14)
            },
            'trend_alignment': {
                'weekly_trend': self.trend_alignment.weekly_trend.value,
                'daily_trend': self.trend_alignment.daily_trend.value,
                'aligned': to_python_type(self.trend_alignment.aligned),
                'conflict_reason': self.trend_alignment.conflict_reason
            },
            'momentum': {
                'rsi_14': to_python_type(self.momentum.rsi_14),
                'weekly_rsi': to_python_type(self.momentum.weekly_rsi),
                'macd_signal': self.momentum.macd_signal,
                'momentum_5d': to_python_type(self.momentum.momentum_5d),
                'momentum_10d': to_python_type(self.momentum.momentum_10d)
            },
            'volume': {
                'volume_trend': self.volume.volume_trend,
                'volume_confirmation': to_python_type(self.volume.volume_confirmation),
                'avg_volume_20d': to_python_type(self.volume.avg_volume_20d)
            },
            'structure_notes': self.structure_notes
        }


# ============================================================
# PHASE 2: SWING HEALTH STATE
# ============================================================

@dataclass
class SwingHealthState:
    """Phase 2 Output: Signal health assessment"""
    health_score: int = 0
    health_tier: SwingHealth = SwingHealth.WEAK
    stand_down: bool = True
    reasons: List[str] = field(default_factory=list)

    # Dimension scores
    trend_strength: int = 0
    momentum_quality: int = 0
    volume_support: int = 0
    failure_risk: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'health_score': to_python_type(self.health_score),
            'health_tier': self.health_tier.value,
            'stand_down': to_python_type(self.stand_down),
            'reasons': self.reasons,
            'dimensions': {
                'trend_strength': to_python_type(self.trend_strength),
                'momentum_quality': to_python_type(self.momentum_quality),
                'volume_support': to_python_type(self.volume_support),
                'failure_risk': to_python_type(self.failure_risk)
            }
        }


# ============================================================
# PHASE 3: SWING DENSITY STATE
# ============================================================

@dataclass
class SwingDensityState:
    """Phase 3 Output: Signal density control"""
    density_score: int = 0
    is_clean_setup: bool = False
    wait_for: str = ""  # What to wait for if not clean
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'density_score': to_python_type(self.density_score),
            'is_clean_setup': to_python_type(self.is_clean_setup),
            'wait_for': self.wait_for,
            'reasons': self.reasons
        }


# ============================================================
# PHASE 4: SWING EXECUTION STATE
# ============================================================

@dataclass
class SwingExecutionState:
    """Phase 4 Output: Final trade permission"""
    allowed: bool = False
    bias: SwingBias = SwingBias.NEUTRAL
    mode: SwingMode = SwingMode.NO_TRADE
    holding_period: str = "N/A"  # "5-DAY", "10-DAY", etc.
    invalidation_levels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'allowed': to_python_type(self.allowed),
            'bias': self.bias.value,
            'mode': self.mode.value,
            'holding_period': self.holding_period,
            'invalidation_levels': self.invalidation_levels
        }


# ============================================================
# PHASE 1 ENGINE
# ============================================================

class SwingPhase1Engine:
    """Calculate swing reality state from daily/weekly data."""

    def calculate(self, daily_bars: pd.DataFrame,
                  weekly_bars: pd.DataFrame = None) -> SwingRealityState:
        state = SwingRealityState()

        if daily_bars is None or len(daily_bars) < 30:
            return state

        state.resolved = True
        notes = []

        # Current price
        current_price = float(daily_bars['close'].iloc[-1])
        state.levels.current_price = round(current_price, 2)

        # Calculate levels
        state.levels = self._calculate_levels(daily_bars, weekly_bars)

        # Calculate trends
        state.trend_alignment = self._calculate_trends(daily_bars, weekly_bars)

        # Calculate momentum
        state.momentum = self._calculate_momentum(daily_bars, weekly_bars)

        # Calculate volume
        state.volume = self._calculate_volume(daily_bars)

        # Determine direction
        state.direction = self._determine_direction(state)

        # Determine confidence
        state.confidence = self._determine_confidence(state)

        # Determine dominant timeframe
        if weekly_bars is not None and len(weekly_bars) >= 4:
            weekly_move = abs(weekly_bars['close'].iloc[-1] - weekly_bars['close'].iloc[-4]) / weekly_bars['close'].iloc[-4]
            daily_move = abs(daily_bars['close'].iloc[-1] - daily_bars['close'].iloc[-20]) / daily_bars['close'].iloc[-20]
            if weekly_move > daily_move * 1.5:
                state.dominant_timeframe = SwingTimeframe.WEEKLY

        # Add notes
        if state.trend_alignment.aligned:
            notes.append(f"Trends aligned: {state.trend_alignment.weekly_trend.value}")
        else:
            notes.append(f"Trend conflict: {state.trend_alignment.conflict_reason}")

        if state.momentum.rsi_14 > 70:
            notes.append("RSI overbought - caution for longs")
        elif state.momentum.rsi_14 < 30:
            notes.append("RSI oversold - caution for shorts")

        if state.volume.volume_confirmation:
            notes.append("Volume confirms move")

        state.structure_notes = notes

        return state

    def _calculate_levels(self, daily_bars: pd.DataFrame,
                          weekly_bars: pd.DataFrame = None) -> SwingLevels:
        levels = SwingLevels()

        close = daily_bars['close'].values
        high = daily_bars['high'].values
        low = daily_bars['low'].values

        levels.current_price = round(float(close[-1]), 2)
        levels.daily_high_5d = round(float(np.max(high[-5:])), 2)
        levels.daily_low_5d = round(float(np.min(low[-5:])), 2)

        if len(daily_bars) >= 20:
            levels.sma_20 = round(float(np.mean(close[-20:])), 2)
        if len(daily_bars) >= 50:
            levels.sma_50 = round(float(np.mean(close[-50:])), 2)

        # ATR
        if len(daily_bars) >= 14:
            tr_list = []
            for i in range(-14, 0):
                h = daily_bars.iloc[i]['high']
                l = daily_bars.iloc[i]['low']
                pc = daily_bars.iloc[i-1]['close'] if i > -14 else daily_bars.iloc[i]['open']
                tr = max(h - l, abs(h - pc), abs(l - pc))
                tr_list.append(tr)
            levels.atr_14 = round(float(np.mean(tr_list)), 2)

        # Weekly levels
        if weekly_bars is not None and len(weekly_bars) >= 1:
            levels.weekly_high = round(float(weekly_bars['high'].iloc[-1]), 2)
            levels.weekly_low = round(float(weekly_bars['low'].iloc[-1]), 2)
            levels.weekly_mid = round((levels.weekly_high + levels.weekly_low) / 2, 2)

        return levels

    def _calculate_trends(self, daily_bars: pd.DataFrame,
                          weekly_bars: pd.DataFrame = None) -> TrendAlignment:
        alignment = TrendAlignment()

        # Daily trend
        close = daily_bars['close'].values
        if len(daily_bars) >= 20:
            sma_5 = np.mean(close[-5:])
            sma_20 = np.mean(close[-20:])
            if sma_5 > sma_20 and close[-1] > sma_5:
                alignment.daily_trend = SwingTrend.UPTREND
            elif sma_5 < sma_20 and close[-1] < sma_5:
                alignment.daily_trend = SwingTrend.DOWNTREND
            else:
                alignment.daily_trend = SwingTrend.SIDEWAYS

        # Weekly trend
        if weekly_bars is not None and len(weekly_bars) >= 4:
            weekly_close = weekly_bars['close'].values
            sma_2w = np.mean(weekly_close[-2:])
            sma_4w = np.mean(weekly_close[-4:])
            if sma_2w > sma_4w and weekly_close[-1] > sma_2w:
                alignment.weekly_trend = SwingTrend.UPTREND
            elif sma_2w < sma_4w and weekly_close[-1] < sma_2w:
                alignment.weekly_trend = SwingTrend.DOWNTREND
            else:
                alignment.weekly_trend = SwingTrend.SIDEWAYS

        # Check alignment
        if alignment.daily_trend == alignment.weekly_trend:
            alignment.aligned = True
        elif alignment.weekly_trend == SwingTrend.SIDEWAYS or alignment.daily_trend == SwingTrend.SIDEWAYS:
            alignment.aligned = False
            alignment.conflict_reason = "One timeframe sideways"
        else:
            alignment.aligned = False
            alignment.conflict_reason = f"Daily={alignment.daily_trend.value} vs Weekly={alignment.weekly_trend.value}"

        return alignment

    def _calculate_momentum(self, daily_bars: pd.DataFrame,
                            weekly_bars: pd.DataFrame = None) -> SwingMomentum:
        momentum = SwingMomentum()

        close = daily_bars['close'].values

        # RSI
        if len(daily_bars) >= 14:
            changes = pd.Series(close).diff().dropna()[-14:]
            gains = changes[changes > 0].sum()
            losses = abs(changes[changes < 0].sum())
            if losses > 0:
                rs = gains / losses
                momentum.rsi_14 = round(100 - (100 / (1 + rs)), 1)
            else:
                momentum.rsi_14 = 100.0

        # Weekly RSI
        if weekly_bars is not None and len(weekly_bars) >= 4:
            weekly_close = weekly_bars['close'].values
            changes = pd.Series(weekly_close).diff().dropna()[-4:]
            gains = changes[changes > 0].sum()
            losses = abs(changes[changes < 0].sum())
            if losses > 0:
                rs = gains / losses
                momentum.weekly_rsi = round(100 - (100 / (1 + rs)), 1)
            else:
                momentum.weekly_rsi = 100.0

        # Momentum
        if len(close) >= 5:
            momentum.momentum_5d = round((close[-1] - close[-5]) / close[-5] * 100, 2)
        if len(close) >= 10:
            momentum.momentum_10d = round((close[-1] - close[-10]) / close[-10] * 100, 2)

        # MACD signal (simplified)
        if len(daily_bars) >= 26:
            ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            if macd > 0:
                momentum.macd_signal = "BULLISH"
            elif macd < 0:
                momentum.macd_signal = "BEARISH"

        return momentum

    def _calculate_volume(self, daily_bars: pd.DataFrame) -> SwingVolume:
        vol = SwingVolume()

        if 'volume' not in daily_bars.columns or len(daily_bars) < 20:
            return vol

        volume = daily_bars['volume'].values
        vol.avg_volume_20d = float(np.mean(volume[-20:]))

        # Volume trend
        recent_vol = np.mean(volume[-5:])
        older_vol = np.mean(volume[-10:-5]) if len(volume) >= 10 else recent_vol

        if recent_vol > older_vol * 1.2:
            vol.volume_trend = "INCREASING"
        elif recent_vol < older_vol * 0.8:
            vol.volume_trend = "DECREASING"
        else:
            vol.volume_trend = "NORMAL"

        # Volume confirmation
        close = daily_bars['close'].values
        price_up = close[-1] > close[-5]
        if (price_up and vol.volume_trend == "INCREASING") or \
           (not price_up and vol.volume_trend == "INCREASING"):
            vol.volume_confirmation = True

        return vol

    def _determine_direction(self, state: SwingRealityState) -> SwingDirection:
        score = 0

        # Trend alignment
        if state.trend_alignment.weekly_trend == SwingTrend.UPTREND:
            score += 2
        elif state.trend_alignment.weekly_trend == SwingTrend.DOWNTREND:
            score -= 2

        if state.trend_alignment.daily_trend == SwingTrend.UPTREND:
            score += 1
        elif state.trend_alignment.daily_trend == SwingTrend.DOWNTREND:
            score -= 1

        # Momentum
        if state.momentum.momentum_5d > 1:
            score += 1
        elif state.momentum.momentum_5d < -1:
            score -= 1

        # Price vs MAs
        if state.levels.current_price > state.levels.sma_20 > 0:
            score += 1
        elif state.levels.current_price < state.levels.sma_20:
            score -= 1

        if score >= 2:
            return SwingDirection.BULLISH
        elif score <= -2:
            return SwingDirection.BEARISH
        else:
            return SwingDirection.NEUTRAL

    def _determine_confidence(self, state: SwingRealityState) -> SwingConfidence:
        score = 0

        if state.trend_alignment.aligned:
            score += 30

        if state.volume.volume_confirmation:
            score += 20

        if 30 < state.momentum.rsi_14 < 70:
            score += 15  # Not overbought/oversold

        if state.direction != SwingDirection.NEUTRAL:
            score += 20

        if score >= 65:
            return SwingConfidence.HIGH
        elif score >= 45:
            return SwingConfidence.MEDIUM
        elif score >= 25:
            return SwingConfidence.LOW
        else:
            return SwingConfidence.NO_TRADE


# ============================================================
# PHASE 2 ENGINE
# ============================================================

class SwingPhase2Engine:
    """Calculate swing health state."""

    def calculate(self, reality: SwingRealityState) -> SwingHealthState:
        health = SwingHealthState()
        reasons = []

        # Trend strength (0-100)
        trend_score = 50
        if reality.trend_alignment.aligned:
            trend_score = 80
        elif reality.trend_alignment.weekly_trend == SwingTrend.SIDEWAYS:
            trend_score = 40
            reasons.append("Weekly trend unclear")
        else:
            trend_score = 30
            reasons.append("Timeframe conflict")
        health.trend_strength = trend_score

        # Momentum quality (0-100)
        mom_score = 50
        if 40 <= reality.momentum.rsi_14 <= 60:
            mom_score = 70  # Good entry zone
        elif reality.momentum.rsi_14 > 70:
            mom_score = 30
            reasons.append("RSI overbought")
        elif reality.momentum.rsi_14 < 30:
            mom_score = 30
            reasons.append("RSI oversold")

        if reality.momentum.macd_signal == "BULLISH" and reality.direction == SwingDirection.BULLISH:
            mom_score += 20
        elif reality.momentum.macd_signal == "BEARISH" and reality.direction == SwingDirection.BEARISH:
            mom_score += 20
        health.momentum_quality = min(100, mom_score)

        # Volume support (0-100)
        vol_score = 50
        if reality.volume.volume_confirmation:
            vol_score = 80
        elif reality.volume.volume_trend == "DECREASING":
            vol_score = 30
            reasons.append("Volume declining")
        health.volume_support = vol_score

        # Failure risk (0-100, higher = less risk)
        fail_score = 70
        if not reality.trend_alignment.aligned:
            fail_score -= 20
        if reality.momentum.rsi_14 > 75 or reality.momentum.rsi_14 < 25:
            fail_score -= 15
        health.failure_risk = max(0, fail_score)

        # Aggregate
        health.health_score = int(
            health.trend_strength * 0.35 +
            health.momentum_quality * 0.25 +
            health.volume_support * 0.20 +
            health.failure_risk * 0.20
        )

        # Determine tier
        if health.health_score >= 65:
            health.health_tier = SwingHealth.HEALTHY
            health.stand_down = False
        elif health.health_score >= 45:
            health.health_tier = SwingHealth.MIXED
            health.stand_down = False
        else:
            health.health_tier = SwingHealth.WEAK
            health.stand_down = True

        health.reasons = reasons
        return health


# ============================================================
# PHASE 3 ENGINE
# ============================================================

class SwingPhase3Engine:
    """Calculate swing density state."""

    def calculate(self, reality: SwingRealityState,
                  health: SwingHealthState) -> SwingDensityState:
        density = SwingDensityState()
        reasons = []

        score = 100

        # Chop detection
        if reality.trend_alignment.weekly_trend == SwingTrend.SIDEWAYS:
            score -= 30
            reasons.append("Weekly chop")

        if reality.trend_alignment.daily_trend == SwingTrend.SIDEWAYS:
            score -= 20
            reasons.append("Daily chop")

        # Overbought/oversold extremes need pullback
        if reality.momentum.rsi_14 > 80:
            score -= 25
            reasons.append("Extremely overbought")
            density.wait_for = "RSI pullback below 70"
        elif reality.momentum.rsi_14 < 20:
            score -= 25
            reasons.append("Extremely oversold")
            density.wait_for = "RSI bounce above 30"

        density.density_score = max(0, score)
        density.reasons = reasons

        if density.density_score >= 60 and not density.wait_for:
            density.is_clean_setup = True
        else:
            density.is_clean_setup = False

        return density


# ============================================================
# PHASE 4 ENGINE
# ============================================================

class SwingPhase4Engine:
    """Calculate swing execution state."""

    def calculate(self, reality: SwingRealityState,
                  health: SwingHealthState,
                  density: SwingDensityState) -> SwingExecutionState:
        execution = SwingExecutionState()
        invalidations = []

        # Check gates
        deny = False

        if reality.confidence == SwingConfidence.NO_TRADE:
            deny = True
            invalidations.append("No trade confidence")

        if health.stand_down:
            deny = True
            invalidations.append("Health stand-down")

        if not density.is_clean_setup:
            deny = True
            invalidations.append(f"Not clean: {density.wait_for or 'choppy conditions'}")

        if deny:
            execution.allowed = False
            execution.bias = SwingBias.NEUTRAL
            execution.mode = SwingMode.NO_TRADE
            execution.invalidation_levels = invalidations
            return execution

        # Permission granted
        execution.allowed = True

        # Determine bias
        if reality.direction == SwingDirection.BULLISH:
            execution.bias = SwingBias.LONG
        elif reality.direction == SwingDirection.BEARISH:
            execution.bias = SwingBias.SHORT
        else:
            execution.bias = SwingBias.NEUTRAL

        # Determine mode
        if reality.trend_alignment.aligned and reality.trend_alignment.weekly_trend != SwingTrend.SIDEWAYS:
            execution.mode = SwingMode.TREND_FOLLOW
            execution.holding_period = "5-10 DAYS"
        elif reality.momentum.rsi_14 < 35 or reality.momentum.rsi_14 > 65:
            execution.mode = SwingMode.MEAN_REVERSION
            execution.holding_period = "3-5 DAYS"
        else:
            execution.mode = SwingMode.BREAKOUT
            execution.holding_period = "5 DAYS"

        # Invalidation levels
        if execution.bias == SwingBias.LONG:
            invalidations.append(f"Stop below {reality.levels.daily_low_5d}")
            if reality.levels.sma_50 > 0:
                invalidations.append(f"Invalidate below SMA50: {reality.levels.sma_50}")
        elif execution.bias == SwingBias.SHORT:
            invalidations.append(f"Stop above {reality.levels.daily_high_5d}")
            if reality.levels.sma_50 > 0:
                invalidations.append(f"Invalidate above SMA50: {reality.levels.sma_50}")

        execution.invalidation_levels = invalidations
        return execution


# ============================================================
# FULL SWING PIPELINE
# ============================================================

class SwingRPEPipeline:
    """Complete RPE SWING Pipeline for multi-timeframe swing analysis."""

    def __init__(self):
        self.phase1 = SwingPhase1Engine()
        self.phase2 = SwingPhase2Engine()
        self.phase3 = SwingPhase3Engine()
        self.phase4 = SwingPhase4Engine()

    def run(self, symbol: str,
            daily_bars: pd.DataFrame,
            weekly_bars: pd.DataFrame = None,
            v6_swing_predictions: Dict = None) -> Dict[str, Any]:
        """
        Run complete swing analysis pipeline.

        Args:
            symbol: Ticker symbol
            daily_bars: Daily OHLCV DataFrame
            weekly_bars: Weekly OHLCV DataFrame (optional)
            v6_swing_predictions: V6 SWING model predictions (optional)

        Returns:
            Dict with all 4 phases + optional V6 signals
        """
        # Phase 1: Reality
        reality = self.phase1.calculate(daily_bars, weekly_bars)

        # Phase 2: Health
        health = self.phase2.calculate(reality)

        # Phase 3: Density
        density = self.phase3.calculate(reality, health)

        # Phase 4: Execution
        execution = self.phase4.calculate(reality, health, density)

        result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': 'SWING',
            'phase1': reality.to_dict(),
            'phase2': health.to_dict(),
            'phase3': density.to_dict(),
            'phase4': execution.to_dict()
        }

        # Add V6 swing predictions if available
        if v6_swing_predictions:
            result['v6_swing'] = v6_swing_predictions

        return result


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def analyze_swing(symbol: str,
                  daily_bars: pd.DataFrame,
                  weekly_bars: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Convenience function to run swing analysis.
    """
    pipeline = SwingRPEPipeline()
    return pipeline.run(symbol, daily_bars, weekly_bars)
