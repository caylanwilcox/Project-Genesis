"""
Reality Proof Engine (RPE) - 5-Phase Market Structure Analysis
==============================================================
Contract Version: 2.0

A 5-phase market structure analysis system with strict layering invariants.

Phase 1: TRUTH - Core market structure truth (immutable, no ML override)
Phase 2: SIGNAL_HEALTH - Data integrity and signal health assessment
Phase 3: SIGNAL_DENSITY - Mode, cooldown, budget, material change tracking
Phase 4: EXECUTION_POSTURE - Bias, play type, confidence, invalidation
Phase 5: LEARNING_FORECASTING - Predictions, forecasts, entry/exit (ONLY phase with predictions)

LAYERING INVARIANTS:
- truth_never_depends_on_decisions: Phase 1 is pure observation
- no_repainting: All calculations use only bars where t < now
- predictions_only_in_phase5: No ML predictions in phases 1-4
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.bool_, np.generic)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, Enum):
        return val.value
    return val


# ============================================================
# META & CONTRACT
# ============================================================

@dataclass
class RPEMeta:
    """RPE Contract metadata"""
    contract_version: str = "2.0"
    timezone: str = "America/New_York"
    instrument_universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    bar_interval: str = "1min"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_version': self.contract_version,
            'timezone': self.timezone,
            'instrument_universe': self.instrument_universe,
            'bar_interval': self.bar_interval
        }


@dataclass
class LayeringInvariants:
    """Invariants that must hold across all phases"""
    truth_never_depends_on_decisions: bool = True
    no_repainting: bool = True
    predictions_only_in_phase5: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'truth_never_depends_on_decisions': self.truth_never_depends_on_decisions,
            'no_repainting': self.no_repainting,
            'predictions_only_in_phase5': self.predictions_only_in_phase5
        }


# ============================================================
# PHASE 1 ENUMS
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

class AcceptanceStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"

class RangeStateType(Enum):
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

class MemoryType(Enum):
    PERSISTENT = "PERSISTENT"
    RESET = "RESET"

class GapBehavior(Enum):
    FILL = "FILL"
    HOLD = "HOLD"
    EXPAND = "EXPAND"


# ============================================================
# PHASE 2 ENUMS
# ============================================================

class HealthTier(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNSTABLE = "UNSTABLE"

class RiskStateType(Enum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    DEFENSIVE = "DEFENSIVE"


# ============================================================
# PHASE 3 ENUMS
# ============================================================

class DensityMode(Enum):
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    NORMAL = "NORMAL"
    QUIET = "QUIET"

class ThrottleLevel(Enum):
    OPEN = "OPEN"
    LIMITED = "LIMITED"
    BLOCKED = "BLOCKED"


# ============================================================
# PHASE 4 ENUMS
# ============================================================

class Bias(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class PlayType(Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    SCALP = "SCALP"
    NO_TRADE = "NO_TRADE"


# ============================================================
# PHASE 5 ENUMS
# ============================================================

class RegimeType(Enum):
    TREND = "TREND"
    MEAN_REVERSION = "MR"
    CHOP = "CHOP"
    EVENT = "EVENT"

class TripleBarrierOutcome(Enum):
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    PENDING = "PENDING"


# ============================================================
# PHASE 1: TRUTH DATACLASSES
# ============================================================

@dataclass
class Core12:
    """Core 12 market structure indicators"""
    vwap: float = 0.0
    vwap_slope: float = 0.0
    poc: float = 0.0  # Point of Control
    val: float = 0.0  # Value Area Low
    vah: float = 0.0  # Value Area High
    vpoc_developing: bool = False
    delta: float = 0.0
    cumulative_delta: float = 0.0
    internals_advance_decline: float = 0.0
    internals_tick: float = 0.0
    atr_14: float = 0.0
    relative_volume: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: to_python_type(v) for k, v in {
            'vwap': self.vwap,
            'vwap_slope': self.vwap_slope,
            'poc': self.poc,
            'val': self.val,
            'vah': self.vah,
            'vpoc_developing': self.vpoc_developing,
            'delta': self.delta,
            'cumulative_delta': self.cumulative_delta,
            'internals_advance_decline': self.internals_advance_decline,
            'internals_tick': self.internals_tick,
            'atr_14': self.atr_14,
            'relative_volume': self.relative_volume
        }.items()}


@dataclass
class Acceptance:
    """Price acceptance state at key levels"""
    accepted: bool = False
    strength: AcceptanceStrength = AcceptanceStrength.WEAK
    at_level: Optional[str] = None
    time_held_bars: int = 0
    failed_levels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accepted': to_python_type(self.accepted),
            'strength': self.strength.value,
            'at_level': self.at_level,
            'time_held_bars': to_python_type(self.time_held_bars),
            'failed_levels': self.failed_levels
        }


@dataclass
class RangeState:
    """Range/trend state"""
    state: RangeStateType = RangeStateType.BALANCE
    rotation_complete: bool = False
    expansion_quality: ExpansionQuality = ExpansionQuality.NONE
    range_high: float = 0.0
    range_low: float = 0.0
    range_width_atr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'rotation_complete': to_python_type(self.rotation_complete),
            'expansion_quality': self.expansion_quality.value,
            'range_high': to_python_type(self.range_high),
            'range_low': to_python_type(self.range_low),
            'range_width_atr': to_python_type(self.range_width_atr)
        }


@dataclass
class MTFContinuity:
    """Multi-timeframe alignment state"""
    aligned: bool = False
    dominant_tf: DominantTimeframe = DominantTimeframe.INTRADAY
    conflict_tf: Optional[str] = None
    alignment_score: int = 0  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'aligned': to_python_type(self.aligned),
            'dominant_tf': self.dominant_tf.value,
            'conflict_tf': self.conflict_tf,
            'alignment_score': to_python_type(self.alignment_score)
        }


@dataclass
class Participation:
    """Market participation metrics"""
    conviction: Conviction = Conviction.LOW
    effort_result_match: bool = False
    volume_profile_shape: str = "NORMAL"  # NORMAL, SKEWED_HIGH, SKEWED_LOW
    institutional_footprint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'conviction': self.conviction.value,
            'effort_result_match': to_python_type(self.effort_result_match),
            'volume_profile_shape': self.volume_profile_shape,
            'institutional_footprint': to_python_type(self.institutional_footprint)
        }


@dataclass
class SessionMemory:
    """Session context and gap behavior"""
    memory: MemoryType = MemoryType.RESET
    gap_behavior: Optional[GapBehavior] = None
    gap_size_atr: float = 0.0
    overnight_range_atr: float = 0.0
    prior_day_close: float = 0.0
    prior_day_high: float = 0.0
    prior_day_low: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory': self.memory.value,
            'gap_behavior': self.gap_behavior.value if self.gap_behavior else None,
            'gap_size_atr': to_python_type(self.gap_size_atr),
            'overnight_range_atr': to_python_type(self.overnight_range_atr),
            'prior_day_close': to_python_type(self.prior_day_close),
            'prior_day_high': to_python_type(self.prior_day_high),
            'prior_day_low': to_python_type(self.prior_day_low)
        }


@dataclass
class Failures:
    """Failure pattern detection"""
    present: bool = False
    failure_types: List[str] = field(default_factory=list)
    most_recent_failure_bars_ago: Optional[int] = None
    failed_breakout_level: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'present': to_python_type(self.present),
            'failure_types': self.failure_types,
            'most_recent_failure_bars_ago': to_python_type(self.most_recent_failure_bars_ago) if self.most_recent_failure_bars_ago else None,
            'failed_breakout_level': to_python_type(self.failed_breakout_level) if self.failed_breakout_level else None
        }


@dataclass
class Phase1Aggregation:
    """Aggregated Phase 1 scores"""
    resolved: bool = False
    direction: Direction = Direction.BALANCED
    dominant_timeframe: DominantTimeframe = DominantTimeframe.INTRADAY
    truth_score: int = 0  # 0-100
    confidence_tier: str = "NO_TRADE"  # NO_TRADE, CONTEXT_ONLY, STRUCTURAL_EDGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resolved': to_python_type(self.resolved),
            'direction': self.direction.value,
            'dominant_timeframe': self.dominant_timeframe.value,
            'truth_score': to_python_type(self.truth_score),
            'confidence_tier': self.confidence_tier
        }


@dataclass
class Phase1Truth:
    """Phase 1 Complete Output: Immutable market structure truth"""
    core12: Core12 = field(default_factory=Core12)
    acceptance: Acceptance = field(default_factory=Acceptance)
    range_state: RangeState = field(default_factory=RangeState)
    mtf_continuity: MTFContinuity = field(default_factory=MTFContinuity)
    participation: Participation = field(default_factory=Participation)
    session_memory: SessionMemory = field(default_factory=SessionMemory)
    failures: Failures = field(default_factory=Failures)
    aggregation: Phase1Aggregation = field(default_factory=Phase1Aggregation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'core12': self.core12.to_dict(),
            'acceptance': self.acceptance.to_dict(),
            'range_state': self.range_state.to_dict(),
            'mtf_continuity': self.mtf_continuity.to_dict(),
            'participation': self.participation.to_dict(),
            'session_memory': self.session_memory.to_dict(),
            'failures': self.failures.to_dict(),
            'aggregation': self.aggregation.to_dict()
        }


# ============================================================
# PHASE 2: SIGNAL HEALTH DATACLASSES
# ============================================================

@dataclass
class DataIntegrity:
    """Data feed integrity checks"""
    missing_bars: int = 0
    duplicate_bars: int = 0
    time_alignment_ok: bool = True
    feed_latency_ms: float = 0.0
    data_quality_score: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'missing_bars': to_python_type(self.missing_bars),
            'duplicate_bars': to_python_type(self.duplicate_bars),
            'time_alignment_ok': to_python_type(self.time_alignment_ok),
            'feed_latency_ms': to_python_type(self.feed_latency_ms),
            'data_quality_score': to_python_type(self.data_quality_score)
        }


@dataclass
class SignalConditions:
    """Signal quality conditions"""
    structural_integrity: int = 100
    time_persistence: int = 100
    volatility_alignment: int = 100
    participation_consistency: int = 100
    failure_risk: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'structural_integrity': to_python_type(self.structural_integrity),
            'time_persistence': to_python_type(self.time_persistence),
            'volatility_alignment': to_python_type(self.volatility_alignment),
            'participation_consistency': to_python_type(self.participation_consistency),
            'failure_risk': to_python_type(self.failure_risk)
        }


@dataclass
class RiskState:
    """Current risk assessment"""
    state: RiskStateType = RiskStateType.DEFENSIVE
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    vix_level: Optional[float] = None
    correlation_breakdown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'volatility_regime': self.volatility_regime,
            'vix_level': to_python_type(self.vix_level) if self.vix_level else None,
            'correlation_breakdown': to_python_type(self.correlation_breakdown)
        }


@dataclass
class Phase2SignalHealth:
    """Phase 2 Complete Output: Signal health assessment"""
    integrity: DataIntegrity = field(default_factory=DataIntegrity)
    conditions: SignalConditions = field(default_factory=SignalConditions)
    risk_state: RiskState = field(default_factory=RiskState)
    health_score: int = 0  # 0-100
    tier: HealthTier = HealthTier.UNSTABLE
    blockers: List[str] = field(default_factory=list)
    stand_down: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'integrity': self.integrity.to_dict(),
            'conditions': self.conditions.to_dict(),
            'risk_state': self.risk_state.to_dict(),
            'health_score': to_python_type(self.health_score),
            'tier': self.tier.value,
            'blockers': self.blockers,
            'stand_down': to_python_type(self.stand_down)
        }


# ============================================================
# PHASE 3: SIGNAL DENSITY DATACLASSES
# ============================================================

@dataclass
class DensityModeState:
    """Current density mode"""
    mode: DensityMode = DensityMode.NORMAL
    regime_duration_bars: int = 0
    mode_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'regime_duration_bars': to_python_type(self.regime_duration_bars),
            'mode_confidence': to_python_type(self.mode_confidence)
        }


@dataclass
class Cooldown:
    """Signal cooldown tracking"""
    active: bool = False
    remaining_bars: int = 0
    last_signal_bars_ago: Optional[int] = None
    signals_in_window: int = 0
    window_size_bars: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            'active': to_python_type(self.active),
            'remaining_bars': to_python_type(self.remaining_bars),
            'last_signal_bars_ago': to_python_type(self.last_signal_bars_ago) if self.last_signal_bars_ago else None,
            'signals_in_window': to_python_type(self.signals_in_window),
            'window_size_bars': to_python_type(self.window_size_bars)
        }


@dataclass
class SignalBudget:
    """Signal budget/throttle tracking"""
    daily_budget: int = 10
    used_today: int = 0
    remaining: int = 10
    throttle: ThrottleLevel = ThrottleLevel.OPEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            'daily_budget': to_python_type(self.daily_budget),
            'used_today': to_python_type(self.used_today),
            'remaining': to_python_type(self.remaining),
            'throttle': self.throttle.value
        }


@dataclass
class MaterialChange:
    """Tracks material changes requiring new signal consideration"""
    detected: bool = False
    change_type: Optional[str] = None  # LEVEL_BREAK, REGIME_SHIFT, VOLUME_SPIKE
    magnitude: float = 0.0
    bars_since_change: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected': to_python_type(self.detected),
            'change_type': self.change_type,
            'magnitude': to_python_type(self.magnitude),
            'bars_since_change': to_python_type(self.bars_since_change)
        }


@dataclass
class Phase3SignalDensity:
    """Phase 3 Complete Output: Signal density control"""
    mode: DensityModeState = field(default_factory=DensityModeState)
    cooldown: Cooldown = field(default_factory=Cooldown)
    budget: SignalBudget = field(default_factory=SignalBudget)
    material_change: MaterialChange = field(default_factory=MaterialChange)
    density_score: int = 100  # 0-100
    allow_signal: bool = True
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.to_dict(),
            'cooldown': self.cooldown.to_dict(),
            'budget': self.budget.to_dict(),
            'material_change': self.material_change.to_dict(),
            'density_score': to_python_type(self.density_score),
            'allow_signal': to_python_type(self.allow_signal),
            'reasons': self.reasons
        }


# ============================================================
# PHASE 4: EXECUTION POSTURE DATACLASSES
# ============================================================

@dataclass
class ExecutionBias:
    """Trading bias determination"""
    bias: Bias = Bias.NEUTRAL
    strength: float = 0.0  # 0-1
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bias': self.bias.value,
            'strength': to_python_type(self.strength),
            'reasons': self.reasons
        }


@dataclass
class PlayTypeSelection:
    """Selected play type"""
    play_type: PlayType = PlayType.NO_TRADE
    suitability_score: int = 0
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'play_type': self.play_type.value,
            'suitability_score': to_python_type(self.suitability_score),
            'alternatives': self.alternatives
        }


@dataclass
class ConfidenceRisk:
    """Confidence and risk assessment"""
    confidence: int = 0  # 0-100
    risk_reward_ratio: float = 0.0
    position_size_modifier: float = 1.0
    max_loss_atr: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'confidence': to_python_type(self.confidence),
            'risk_reward_ratio': to_python_type(self.risk_reward_ratio),
            'position_size_modifier': to_python_type(self.position_size_modifier),
            'max_loss_atr': to_python_type(self.max_loss_atr)
        }


@dataclass
class Invalidation:
    """Invalidation levels and conditions"""
    price_levels: List[float] = field(default_factory=list)
    time_limit_bars: Optional[int] = None
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'price_levels': [to_python_type(p) for p in self.price_levels],
            'time_limit_bars': to_python_type(self.time_limit_bars) if self.time_limit_bars else None,
            'conditions': self.conditions
        }


@dataclass
class StandDownLogic:
    """Reasons for standing down"""
    active: bool = False
    reasons: List[str] = field(default_factory=list)
    duration_recommendation_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'active': to_python_type(self.active),
            'reasons': self.reasons,
            'duration_recommendation_bars': to_python_type(self.duration_recommendation_bars)
        }


@dataclass
class Phase4ExecutionPosture:
    """Phase 4 Complete Output: Execution posture"""
    bias: ExecutionBias = field(default_factory=ExecutionBias)
    play_type: PlayTypeSelection = field(default_factory=PlayTypeSelection)
    confidence_risk: ConfidenceRisk = field(default_factory=ConfidenceRisk)
    invalidation: Invalidation = field(default_factory=Invalidation)
    stand_down_logic: StandDownLogic = field(default_factory=StandDownLogic)
    allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bias': self.bias.to_dict(),
            'play_type': self.play_type.to_dict(),
            'confidence_risk': self.confidence_risk.to_dict(),
            'invalidation': self.invalidation.to_dict(),
            'stand_down_logic': self.stand_down_logic.to_dict(),
            'allowed': to_python_type(self.allowed)
        }


# ============================================================
# PHASE 5: LEARNING/FORECASTING DATACLASSES
# ============================================================

@dataclass
class ForecastHorizon:
    """Prediction for a specific time horizon"""
    horizon_name: str = "1h"
    horizon_bars: int = 60
    direction_prob: Dict[str, float] = field(default_factory=lambda: {"UP": 0.33, "DOWN": 0.33, "FLAT": 0.34})
    expected_move_pct: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'horizon_name': self.horizon_name,
            'horizon_bars': to_python_type(self.horizon_bars),
            'direction_prob': {k: to_python_type(v) for k, v in self.direction_prob.items()},
            'expected_move_pct': to_python_type(self.expected_move_pct),
            'confidence': to_python_type(self.confidence)
        }


@dataclass
class RegimeProbs:
    """Regime probability distribution"""
    trend: float = 0.25
    mean_reversion: float = 0.25
    chop: float = 0.25
    event: float = 0.25
    dominant: RegimeType = RegimeType.CHOP

    def to_dict(self) -> Dict[str, Any]:
        return {
            'TREND': to_python_type(self.trend),
            'MR': to_python_type(self.mean_reversion),
            'CHOP': to_python_type(self.chop),
            'EVENT': to_python_type(self.event),
            'dominant': self.dominant.value
        }


@dataclass
class Forecast:
    """ML-based regime and direction forecast"""
    regime_probs: RegimeProbs = field(default_factory=RegimeProbs)
    next_bar_direction: str = "FLAT"
    next_bar_confidence: float = 0.0
    trend_strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime_probs': self.regime_probs.to_dict(),
            'next_bar_direction': self.next_bar_direction,
            'next_bar_confidence': to_python_type(self.next_bar_confidence),
            'trend_strength': to_python_type(self.trend_strength)
        }


@dataclass
class TripleBarrier:
    """Triple barrier method entry/exit"""
    entry_price: float = 0.0
    profit_target: float = 0.0
    stop_loss: float = 0.0
    time_limit_bars: int = 0
    expected_outcome: TripleBarrierOutcome = TripleBarrierOutcome.PENDING
    outcome_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "PROFIT_TARGET": 0.33, "STOP_LOSS": 0.33, "TIME_EXIT": 0.34
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_price': to_python_type(self.entry_price),
            'profit_target': to_python_type(self.profit_target),
            'stop_loss': to_python_type(self.stop_loss),
            'time_limit_bars': to_python_type(self.time_limit_bars),
            'expected_outcome': self.expected_outcome.value,
            'outcome_probabilities': {k: to_python_type(v) for k, v in self.outcome_probabilities.items()}
        }


@dataclass
class EntryExit:
    """Entry and exit predictions"""
    optimal_entry_level: float = 0.0
    optimal_entry_type: str = "LIMIT"  # MARKET, LIMIT, STOP
    triple_barrier: TripleBarrier = field(default_factory=TripleBarrier)
    expected_hold_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimal_entry_level': to_python_type(self.optimal_entry_level),
            'optimal_entry_type': self.optimal_entry_type,
            'triple_barrier': self.triple_barrier.to_dict(),
            'expected_hold_bars': to_python_type(self.expected_hold_bars)
        }


@dataclass
class Phase5LearningForecasting:
    """Phase 5 Complete Output: Predictions and forecasting (ONLY phase with ML predictions)"""
    horizons: List[ForecastHorizon] = field(default_factory=list)
    forecast: Forecast = field(default_factory=Forecast)
    entry_exit: EntryExit = field(default_factory=EntryExit)
    model_version: str = "v6_time_split"
    last_training_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'horizons': [h.to_dict() for h in self.horizons],
            'forecast': self.forecast.to_dict(),
            'entry_exit': self.entry_exit.to_dict(),
            'model_version': self.model_version,
            'last_training_date': self.last_training_date
        }


# ============================================================
# PHASE 1 ENGINE
# ============================================================

class Phase1Engine:
    """
    Phase 1: TRUTH - Calculates immutable market structure truth.
    INVARIANT: No ML predictions, no decisions, pure observation.
    """

    def __init__(self):
        self.lookback_periods = {
            'short': 20,
            'medium': 60,
            'long': 120
        }

    def calculate(self, symbol: str, bars_1m: pd.DataFrame,
                  daily_bars: pd.DataFrame = None) -> Phase1Truth:
        """Calculate Phase 1 Truth from market data."""
        truth = Phase1Truth()

        if bars_1m is None or len(bars_1m) < 30:
            return truth

        # Calculate all components
        truth.core12 = self._calculate_core12(bars_1m)
        truth.acceptance = self._calculate_acceptance(bars_1m)
        truth.range_state = self._calculate_range_state(bars_1m)
        truth.mtf_continuity = self._calculate_mtf(bars_1m, daily_bars)
        truth.participation = self._calculate_participation(bars_1m)
        truth.session_memory = self._calculate_session_memory(bars_1m, daily_bars)
        truth.failures = self._detect_failures(bars_1m)
        truth.aggregation = self._aggregate(truth)

        return truth

    def _calculate_core12(self, bars: pd.DataFrame) -> Core12:
        """Calculate Core 12 indicators"""
        core = Core12()

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        volume = bars['volume'].values if 'volume' in bars.columns else np.ones(len(close))

        # VWAP
        typical_price = (high + low + close) / 3
        cum_vol = np.cumsum(volume)
        cum_vol_price = np.cumsum(typical_price * volume)
        core.vwap = cum_vol_price[-1] / cum_vol[-1] if cum_vol[-1] > 0 else close[-1]

        # VWAP slope (last 20 bars)
        if len(bars) >= 20:
            vwap_20 = cum_vol_price[-20:] / cum_vol[-20:]
            core.vwap_slope = (vwap_20[-1] - vwap_20[0]) / vwap_20[0] if vwap_20[0] != 0 else 0

        # POC/VAH/VAL (simplified volume profile)
        price_vol = list(zip(close, volume))
        sorted_pv = sorted(price_vol, key=lambda x: x[1], reverse=True)
        core.poc = sorted_pv[0][0] if sorted_pv else close[-1]

        total_vol = sum(volume)
        cumulative = 0
        val_set, vah_set = False, False
        sorted_by_price = sorted(price_vol, key=lambda x: x[0])

        for price, vol in sorted_by_price:
            cumulative += vol
            if cumulative >= total_vol * 0.16 and not val_set:
                core.val = price
                val_set = True
            if cumulative >= total_vol * 0.84 and not vah_set:
                core.vah = price
                vah_set = True

        # ATR-14
        if len(bars) >= 14:
            tr = np.maximum(high[1:] - low[1:],
                           np.maximum(np.abs(high[1:] - close[:-1]),
                                     np.abs(low[1:] - close[:-1])))
            core.atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        # Relative volume
        if len(volume) >= 20:
            avg_vol = np.mean(volume[-20:])
            core.relative_volume = np.mean(volume[-5:]) / avg_vol if avg_vol > 0 else 1.0

        # Delta (simplified - using close vs open)
        if len(bars) >= 5:
            recent_delta = sum((c - o) * v for c, o, v in zip(close[-5:], bars['open'].values[-5:], volume[-5:]))
            core.delta = recent_delta
            core.cumulative_delta = sum((c - o) * v for c, o, v in zip(close, bars['open'].values, volume))

        return core

    def _calculate_acceptance(self, bars: pd.DataFrame) -> Acceptance:
        """Calculate price acceptance state"""
        acceptance = Acceptance()

        if len(bars) < 30:
            return acceptance

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Key levels
        recent_high = np.max(high[-30:])
        recent_low = np.min(low[-30:])
        mid_point = (recent_high + recent_low) / 2
        current = close[-1]

        # Count bars above/below mid
        bars_above = np.sum(close[-10:] > mid_point)
        bars_below = np.sum(close[-10:] < mid_point)

        if bars_above >= 7:
            acceptance.accepted = True
            acceptance.strength = AcceptanceStrength.STRONG
            acceptance.at_level = f"Above mid {mid_point:.2f}"
            acceptance.time_held_bars = int(bars_above)
        elif bars_above >= 5:
            acceptance.accepted = True
            acceptance.strength = AcceptanceStrength.MODERATE
            acceptance.at_level = f"Above mid {mid_point:.2f}"
            acceptance.time_held_bars = int(bars_above)
        elif bars_below >= 7:
            acceptance.accepted = True
            acceptance.strength = AcceptanceStrength.STRONG
            acceptance.at_level = f"Below mid {mid_point:.2f}"
            acceptance.time_held_bars = int(bars_below)
        elif bars_below >= 5:
            acceptance.accepted = True
            acceptance.strength = AcceptanceStrength.MODERATE
            acceptance.at_level = f"Below mid {mid_point:.2f}"
            acceptance.time_held_bars = int(bars_below)

        # Track failed levels
        if current < recent_high * 0.995 and np.max(close[-5:]) >= recent_high * 0.998:
            acceptance.failed_levels.append(f"Failed high {recent_high:.2f}")
        if current > recent_low * 1.005 and np.min(close[-5:]) <= recent_low * 1.002:
            acceptance.failed_levels.append(f"Failed low {recent_low:.2f}")

        return acceptance

    def _calculate_range_state(self, bars: pd.DataFrame) -> RangeState:
        """Calculate range/trend state"""
        state = RangeState()

        if len(bars) < 30:
            return state

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        # Range calculations
        range_30 = np.max(high[-30:]) - np.min(low[-30:])
        range_10 = np.max(high[-10:]) - np.min(low[-10:])

        state.range_high = float(np.max(high[-30:]))
        state.range_low = float(np.min(low[-30:]))

        # ATR for normalization
        if len(bars) >= 14:
            tr = np.maximum(high[1:] - low[1:],
                           np.maximum(np.abs(high[1:] - close[:-1]),
                                     np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:])
            state.range_width_atr = range_30 / atr if atr > 0 else 0

        # Determine state
        price_change_30 = (close[-1] - close[-30]) / close[-30] if len(close) >= 30 else 0

        if abs(price_change_30) > 0.005:
            state.state = RangeStateType.TREND
            state.expansion_quality = ExpansionQuality.CLEAN if range_10 > range_30 * 0.5 else ExpansionQuality.DIRTY
        elif range_10 < range_30 * 0.3:
            state.state = RangeStateType.BALANCE
            state.expansion_quality = ExpansionQuality.NONE
        else:
            if range_10 > range_30 * 0.7 and abs(price_change_30) < 0.002:
                state.state = RangeStateType.FAILED_EXPANSION
                state.expansion_quality = ExpansionQuality.DIRTY

        # Rotation check
        recent_highs = high[-10:]
        recent_lows = low[-10:]
        state.rotation_complete = (np.argmax(recent_highs) != len(recent_highs)-1 and
                                   np.argmin(recent_lows) != len(recent_lows)-1)

        return state

    def _calculate_mtf(self, bars_1m: pd.DataFrame, daily_bars: pd.DataFrame = None) -> MTFContinuity:
        """Calculate multi-timeframe continuity"""
        mtf = MTFContinuity()

        intraday_dir = self._get_direction(bars_1m)

        if daily_bars is not None and len(daily_bars) >= 5:
            daily_dir = self._get_direction(daily_bars)

            if intraday_dir == daily_dir:
                mtf.aligned = True
                mtf.alignment_score = 80
            else:
                mtf.aligned = False
                mtf.conflict_tf = f"DAILY={daily_dir.value} vs INTRADAY={intraday_dir.value}"
                mtf.alignment_score = 30

            # Determine dominant timeframe
            daily_vol = self._calc_volatility(daily_bars[-5:])
            intraday_vol = self._calc_volatility(bars_1m[-60:]) if len(bars_1m) >= 60 else 0

            if daily_vol > intraday_vol * 1.5:
                mtf.dominant_tf = DominantTimeframe.DAILY
        else:
            mtf.aligned = True
            mtf.alignment_score = 50

        return mtf

    def _calculate_participation(self, bars: pd.DataFrame) -> Participation:
        """Calculate market participation"""
        part = Participation()

        if len(bars) < 20 or 'volume' not in bars.columns:
            return part

        volume = bars['volume'].values
        close = bars['close'].values

        avg_vol = np.mean(volume[-20:])
        recent_vol = np.mean(volume[-5:])

        # Conviction
        if recent_vol > avg_vol * 1.5:
            part.conviction = Conviction.HIGH
        elif recent_vol > avg_vol:
            part.conviction = Conviction.MEDIUM
        else:
            part.conviction = Conviction.LOW

        # Effort vs result
        price_move = abs(close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        if vol_ratio > 1.3 and price_move > 0.002:
            part.effort_result_match = True
        elif vol_ratio < 0.8 and price_move < 0.001:
            part.effort_result_match = True
        elif vol_ratio > 1.3 and price_move < 0.001:
            part.effort_result_match = False
        else:
            part.effort_result_match = True

        # Volume profile shape
        if len(volume) >= 20:
            upper_vol = np.mean(volume[-5:])
            lower_vol = np.mean(volume[-20:-10])
            if upper_vol > lower_vol * 1.3:
                part.volume_profile_shape = "SKEWED_HIGH"
            elif upper_vol < lower_vol * 0.7:
                part.volume_profile_shape = "SKEWED_LOW"

        # Institutional footprint (large volume spikes)
        if recent_vol > avg_vol * 2:
            part.institutional_footprint = True

        return part

    def _calculate_session_memory(self, bars_1m: pd.DataFrame,
                                   daily_bars: pd.DataFrame = None) -> SessionMemory:
        """Calculate session memory state"""
        memory = SessionMemory()

        if daily_bars is None or len(daily_bars) < 2:
            return memory

        # Prior day data
        memory.prior_day_close = float(daily_bars['close'].values[-2])
        memory.prior_day_high = float(daily_bars['high'].values[-2])
        memory.prior_day_low = float(daily_bars['low'].values[-2])

        # Gap calculation
        today_open = daily_bars['open'].values[-1]
        gap_pct = (today_open - memory.prior_day_close) / memory.prior_day_close

        if len(bars_1m) >= 14:
            tr = np.maximum(bars_1m['high'].values[1:] - bars_1m['low'].values[1:],
                           np.maximum(np.abs(bars_1m['high'].values[1:] - bars_1m['close'].values[:-1]),
                                     np.abs(bars_1m['low'].values[1:] - bars_1m['close'].values[:-1])))
            atr = np.mean(tr[-14:])
            memory.gap_size_atr = abs(gap_pct * memory.prior_day_close) / atr if atr > 0 else 0

        if abs(gap_pct) > 0.002:
            current = bars_1m['close'].values[-1] if len(bars_1m) > 0 else today_open

            if gap_pct > 0:
                if current < today_open:
                    memory.gap_behavior = GapBehavior.FILL
                elif current > today_open * 1.002:
                    memory.gap_behavior = GapBehavior.EXPAND
                else:
                    memory.gap_behavior = GapBehavior.HOLD
            else:
                if current > today_open:
                    memory.gap_behavior = GapBehavior.FILL
                elif current < today_open * 0.998:
                    memory.gap_behavior = GapBehavior.EXPAND
                else:
                    memory.gap_behavior = GapBehavior.HOLD

        # Memory persistence
        if len(daily_bars) >= 3:
            recent_dirs = []
            for i in range(-3, 0):
                if daily_bars['close'].values[i] > daily_bars['open'].values[i]:
                    recent_dirs.append(1)
                else:
                    recent_dirs.append(-1)

            if all(d == recent_dirs[0] for d in recent_dirs):
                memory.memory = MemoryType.PERSISTENT
            else:
                memory.memory = MemoryType.RESET

        return memory

    def _detect_failures(self, bars: pd.DataFrame) -> Failures:
        """Detect failure patterns"""
        failures = Failures()

        if len(bars) < 20:
            return failures

        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values

        recent_high = np.max(high[-10:])
        recent_low = np.min(low[-10:])
        prior_high = np.max(high[-20:-10]) if len(high) >= 20 else recent_high
        prior_low = np.min(low[-20:-10]) if len(low) >= 20 else recent_low

        # Failed breakouts
        if recent_high > prior_high and close[-1] < prior_high:
            failures.present = True
            failures.failure_types.append("FAILED_HIGH_BREAKOUT")
            failures.failed_breakout_level = float(prior_high)
            failures.most_recent_failure_bars_ago = int(np.argmax(high[-10:]))

        if recent_low < prior_low and close[-1] > prior_low:
            failures.present = True
            failures.failure_types.append("FAILED_LOW_BREAKOUT")
            failures.failed_breakout_level = float(prior_low)
            failures.most_recent_failure_bars_ago = int(np.argmin(low[-10:]))

        # Reversal patterns
        if len(bars) >= 3:
            if high[-2] > high[-3] and close[-1] < low[-2]:
                failures.present = True
                failures.failure_types.append("BEARISH_REVERSAL")
            if low[-2] < low[-3] and close[-1] > high[-2]:
                failures.present = True
                failures.failure_types.append("BULLISH_REVERSAL")

        return failures

    def _aggregate(self, truth: Phase1Truth) -> Phase1Aggregation:
        """Aggregate Phase 1 into summary"""
        agg = Phase1Aggregation()
        agg.resolved = True

        # Direction from range state
        if truth.range_state.state == RangeStateType.TREND:
            # Determine direction from price action
            if truth.core12.vwap_slope > 0:
                agg.direction = Direction.UP
            elif truth.core12.vwap_slope < 0:
                agg.direction = Direction.DOWN
            else:
                agg.direction = Direction.BALANCED
        else:
            agg.direction = Direction.BALANCED

        agg.dominant_timeframe = truth.mtf_continuity.dominant_tf

        # Truth score calculation
        score = 0

        if truth.acceptance.accepted:
            score += 20 if truth.acceptance.strength == AcceptanceStrength.STRONG else 15

        if truth.mtf_continuity.aligned:
            score += 25

        if truth.range_state.state == RangeStateType.TREND:
            score += 20
        elif truth.range_state.state == RangeStateType.BALANCE:
            score += 10

        if truth.participation.conviction == Conviction.HIGH:
            score += 15
        elif truth.participation.conviction == Conviction.MEDIUM:
            score += 10

        if truth.participation.effort_result_match:
            score += 10

        if truth.failures.present:
            score -= 30

        agg.truth_score = max(0, min(100, score))

        # Confidence tier
        if agg.truth_score >= 60:
            agg.confidence_tier = "STRUCTURAL_EDGE"
        elif agg.truth_score >= 30:
            agg.confidence_tier = "CONTEXT_ONLY"
        else:
            agg.confidence_tier = "NO_TRADE"

        return agg

    def _get_direction(self, bars: pd.DataFrame) -> Direction:
        """Get direction from price action"""
        if len(bars) < 20:
            return Direction.BALANCED

        close = bars['close'].values
        short_sma = np.mean(close[-20:])
        medium_sma = np.mean(close[-60:]) if len(close) >= 60 else short_sma
        current = close[-1]

        high = bars['high'].values
        low = bars['low'].values

        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        prior_high = np.max(high[-40:-20]) if len(bars) >= 40 else recent_high
        prior_low = np.min(low[-40:-20]) if len(bars) >= 40 else recent_low

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

    def _calc_volatility(self, bars: pd.DataFrame) -> float:
        """Calculate volatility"""
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
# PHASE 2 ENGINE
# ============================================================

class Phase2Engine:
    """
    Phase 2: SIGNAL_HEALTH - Data integrity and signal health assessment.
    """

    def __init__(self):
        self.weights = {
            'structural_integrity': 0.30,
            'time_persistence': 0.15,
            'volatility_alignment': 0.15,
            'participation_consistency': 0.20,
            'failure_risk': 0.20
        }

    def calculate(self, truth: Phase1Truth,
                  bars_1m: pd.DataFrame = None,
                  time_since_acceptance_minutes: int = 0,
                  stall_flags: bool = False,
                  vol_dislocation: bool = False) -> Phase2SignalHealth:
        """Calculate Phase 2 Signal Health."""
        health = Phase2SignalHealth()
        blockers = []

        # Data integrity checks
        health.integrity = self._check_integrity(bars_1m)
        if health.integrity.data_quality_score < 80:
            blockers.append(f"Data quality score: {health.integrity.data_quality_score}")

        # Signal conditions
        health.conditions = self._calculate_conditions(
            truth, time_since_acceptance_minutes, stall_flags, vol_dislocation, blockers
        )

        # Risk state
        health.risk_state = self._calculate_risk_state(truth, bars_1m)

        # Aggregate health score
        health.health_score = int(
            health.conditions.structural_integrity * self.weights['structural_integrity'] +
            health.conditions.time_persistence * self.weights['time_persistence'] +
            health.conditions.volatility_alignment * self.weights['volatility_alignment'] +
            health.conditions.participation_consistency * self.weights['participation_consistency'] +
            health.conditions.failure_risk * self.weights['failure_risk']
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

        health.blockers = blockers
        return health

    def _check_integrity(self, bars: pd.DataFrame) -> DataIntegrity:
        """Check data feed integrity"""
        integrity = DataIntegrity()

        if bars is None or len(bars) == 0:
            integrity.data_quality_score = 0
            integrity.time_alignment_ok = False
            return integrity

        # Check for missing bars (gaps in timestamp)
        if 'timestamp' in bars.columns:
            timestamps = pd.to_datetime(bars['timestamp'])
            diffs = timestamps.diff().dropna()
            expected_diff = pd.Timedelta(minutes=1)
            integrity.missing_bars = int(sum(diffs > expected_diff * 2))

        # Check for duplicates
        if 'timestamp' in bars.columns:
            integrity.duplicate_bars = int(bars.duplicated(subset=['timestamp']).sum())

        # Time alignment
        integrity.time_alignment_ok = integrity.missing_bars < 5 and integrity.duplicate_bars == 0

        # Overall score
        score = 100
        score -= integrity.missing_bars * 5
        score -= integrity.duplicate_bars * 10
        if not integrity.time_alignment_ok:
            score -= 20

        integrity.data_quality_score = max(0, score)
        return integrity

    def _calculate_conditions(self, truth: Phase1Truth,
                               time_since_acceptance: int,
                               stall_flags: bool,
                               vol_dislocation: bool,
                               blockers: List[str]) -> SignalConditions:
        """Calculate signal conditions"""
        cond = SignalConditions()

        # Structural integrity
        struct_score = 100
        if not truth.acceptance.accepted:
            struct_score -= 25
            blockers.append("No acceptance")
        if not truth.mtf_continuity.aligned:
            struct_score -= 15
            blockers.append("MTF conflict")
        if truth.failures.present:
            struct_score -= 30
            blockers.append("Failure signals present")
        if truth.range_state.state == RangeStateType.FAILED_EXPANSION:
            struct_score -= 20
            blockers.append("Failed expansion")
        cond.structural_integrity = max(0, struct_score)

        # Time persistence
        time_score = 100
        if stall_flags:
            time_score -= 20
            blockers.append("Stall detected")
        if time_since_acceptance > 60:
            time_score -= 10
            blockers.append("Long time since acceptance")
        cond.time_persistence = max(0, time_score)

        # Volatility alignment
        vol_score = 100
        if vol_dislocation:
            vol_score -= 25
            blockers.append("Volatility dislocation")
        cond.volatility_alignment = max(0, vol_score)

        # Participation consistency
        part_score = 100
        if truth.participation.conviction == Conviction.LOW:
            part_score -= 15
            blockers.append("Low conviction")
        if not truth.participation.effort_result_match:
            part_score -= 20
            blockers.append("Effort/result mismatch")
        cond.participation_consistency = max(0, part_score)

        # Failure risk
        fail_score = 100
        if truth.failures.present:
            fail_score -= 40
            blockers.append("Active failure patterns")
        cond.failure_risk = max(0, fail_score)

        return cond

    def _calculate_risk_state(self, truth: Phase1Truth, bars: pd.DataFrame) -> RiskState:
        """Calculate risk state"""
        risk = RiskState()

        # Volatility regime from ATR
        if truth.core12.atr_14 > 0:
            # Compare current ATR to historical
            if bars is not None and len(bars) >= 60:
                high = bars['high'].values
                low = bars['low'].values
                close = bars['close'].values
                tr = np.maximum(high[1:] - low[1:],
                               np.maximum(np.abs(high[1:] - close[:-1]),
                                         np.abs(low[1:] - close[:-1])))
                avg_atr = np.mean(tr[-60:])
                current_atr = truth.core12.atr_14

                if current_atr > avg_atr * 2:
                    risk.volatility_regime = "EXTREME"
                elif current_atr > avg_atr * 1.5:
                    risk.volatility_regime = "HIGH"
                elif current_atr < avg_atr * 0.5:
                    risk.volatility_regime = "LOW"
                else:
                    risk.volatility_regime = "NORMAL"

        # Determine risk state
        if risk.volatility_regime == "EXTREME":
            risk.state = RiskStateType.DEFENSIVE
        elif risk.volatility_regime == "HIGH":
            risk.state = RiskStateType.REDUCED
        else:
            risk.state = RiskStateType.NORMAL

        return risk


# ============================================================
# PHASE 3 ENGINE
# ============================================================

class Phase3Engine:
    """
    Phase 3: SIGNAL_DENSITY - Mode, cooldown, budget, material change.
    """

    def __init__(self):
        self.signal_history: List[Dict] = []
        self.daily_signal_count = 0
        self.last_signal_bar = None

    def calculate(self, truth: Phase1Truth,
                  health: Phase2SignalHealth,
                  current_bar: int = 0,
                  signals_last_10m: int = 0,
                  distinct_levels: int = 0) -> Phase3SignalDensity:
        """Calculate Phase 3 Signal Density."""
        density = Phase3SignalDensity()
        reasons = []

        # Density mode
        density.mode = self._calculate_mode(truth)

        # Cooldown
        density.cooldown = self._calculate_cooldown(current_bar, signals_last_10m)

        # Budget
        density.budget = self._calculate_budget(signals_last_10m)

        # Material change detection
        density.material_change = self._detect_material_change(truth, current_bar)

        # Density score calculation
        score = 100

        if signals_last_10m > 3 and distinct_levels <= 1:
            score -= 40
            reasons.append("Same-level spam")

        if density.cooldown.active:
            score -= 30
            reasons.append("In cooldown period")

        if density.budget.remaining <= 0:
            score -= 50
            reasons.append("Daily signal budget exhausted")

        if truth.range_state.state == RangeStateType.BALANCE and signals_last_10m > 4:
            score -= 30
            reasons.append("Noise in balance regime")

        density.density_score = max(0, score)
        density.reasons = reasons

        # Allow signal?
        density.allow_signal = density.density_score >= 40 and not density.cooldown.active

        return density

    def _calculate_mode(self, truth: Phase1Truth) -> DensityModeState:
        """Calculate density mode"""
        mode = DensityModeState()

        if truth.participation.institutional_footprint:
            if truth.aggregation.direction == Direction.UP:
                mode.mode = DensityMode.ACCUMULATION
            elif truth.aggregation.direction == Direction.DOWN:
                mode.mode = DensityMode.DISTRIBUTION
            mode.mode_confidence = 0.8
        elif truth.participation.conviction == Conviction.LOW:
            mode.mode = DensityMode.QUIET
            mode.mode_confidence = 0.6
        else:
            mode.mode = DensityMode.NORMAL
            mode.mode_confidence = 0.5

        return mode

    def _calculate_cooldown(self, current_bar: int, signals_last_10m: int) -> Cooldown:
        """Calculate cooldown state"""
        cooldown = Cooldown()

        cooldown.signals_in_window = signals_last_10m

        if signals_last_10m >= 3:
            cooldown.active = True
            cooldown.remaining_bars = max(0, 10 - (current_bar % 10))

        if self.last_signal_bar is not None:
            cooldown.last_signal_bars_ago = current_bar - self.last_signal_bar

        return cooldown

    def _calculate_budget(self, signals_used: int) -> SignalBudget:
        """Calculate signal budget"""
        budget = SignalBudget()

        budget.used_today = signals_used
        budget.remaining = max(0, budget.daily_budget - signals_used)

        if budget.remaining <= 0:
            budget.throttle = ThrottleLevel.BLOCKED
        elif budget.remaining <= 3:
            budget.throttle = ThrottleLevel.LIMITED
        else:
            budget.throttle = ThrottleLevel.OPEN

        return budget

    def _detect_material_change(self, truth: Phase1Truth, current_bar: int) -> MaterialChange:
        """Detect material changes"""
        change = MaterialChange()

        # Check for failure pattern (level break)
        if truth.failures.present:
            change.detected = True
            change.change_type = "LEVEL_BREAK"
            change.bars_since_change = truth.failures.most_recent_failure_bars_ago or 0

        # Check for regime shift
        if truth.range_state.state == RangeStateType.FAILED_EXPANSION:
            change.detected = True
            change.change_type = "REGIME_SHIFT"

        # Check for volume spike
        if truth.core12.relative_volume > 2.0:
            change.detected = True
            change.change_type = "VOLUME_SPIKE"
            change.magnitude = truth.core12.relative_volume

        return change

    def record_signal(self, bar: int):
        """Record a signal emission"""
        self.last_signal_bar = bar
        self.daily_signal_count += 1
        self.signal_history.append({'bar': bar, 'time': datetime.now()})

    def reset_daily(self):
        """Reset daily counters"""
        self.daily_signal_count = 0
        self.signal_history = []


# ============================================================
# PHASE 4 ENGINE
# ============================================================

class Phase4Engine:
    """
    Phase 4: EXECUTION_POSTURE - Bias, play type, confidence, invalidation.
    """

    def calculate(self, truth: Phase1Truth,
                  health: Phase2SignalHealth,
                  density: Phase3SignalDensity) -> Phase4ExecutionPosture:
        """Calculate Phase 4 Execution Posture."""
        posture = Phase4ExecutionPosture()

        # Check for stand-down conditions first
        posture.stand_down_logic = self._check_stand_down(truth, health, density)

        if posture.stand_down_logic.active:
            posture.allowed = False
            posture.bias.bias = Bias.NEUTRAL
            posture.play_type.play_type = PlayType.NO_TRADE
            return posture

        # Calculate bias
        posture.bias = self._calculate_bias(truth)

        # Calculate play type
        posture.play_type = self._calculate_play_type(truth, health)

        # Calculate confidence/risk
        posture.confidence_risk = self._calculate_confidence_risk(truth, health, density)

        # Calculate invalidation levels
        posture.invalidation = self._calculate_invalidation(truth)

        # Final allowed decision
        posture.allowed = (
            truth.aggregation.confidence_tier == "STRUCTURAL_EDGE" and
            not health.stand_down and
            density.allow_signal and
            posture.confidence_risk.confidence >= 50
        )

        return posture

    def _check_stand_down(self, truth: Phase1Truth,
                           health: Phase2SignalHealth,
                           density: Phase3SignalDensity) -> StandDownLogic:
        """Check for stand-down conditions"""
        stand_down = StandDownLogic()

        reasons = []

        if truth.aggregation.confidence_tier == "NO_TRADE":
            reasons.append("Confidence tier is NO_TRADE")

        if health.stand_down:
            reasons.append("Health stand_down active")

        if density.budget.throttle == ThrottleLevel.BLOCKED:
            reasons.append("Signal budget exhausted")

        if health.risk_state.volatility_regime == "EXTREME":
            reasons.append("Extreme volatility regime")

        if len(reasons) > 0:
            stand_down.active = True
            stand_down.reasons = reasons
            stand_down.duration_recommendation_bars = 10

        return stand_down

    def _calculate_bias(self, truth: Phase1Truth) -> ExecutionBias:
        """Calculate trading bias"""
        bias = ExecutionBias()

        if truth.aggregation.direction == Direction.UP:
            bias.bias = Bias.LONG
            bias.strength = 0.7 if truth.mtf_continuity.aligned else 0.5
            bias.reasons.append("Upward direction confirmed")
        elif truth.aggregation.direction == Direction.DOWN:
            bias.bias = Bias.SHORT
            bias.strength = 0.7 if truth.mtf_continuity.aligned else 0.5
            bias.reasons.append("Downward direction confirmed")
        else:
            bias.bias = Bias.NEUTRAL
            bias.strength = 0.3
            bias.reasons.append("No clear directional bias")

        if truth.mtf_continuity.aligned:
            bias.reasons.append("MTF aligned")

        return bias

    def _calculate_play_type(self, truth: Phase1Truth,
                              health: Phase2SignalHealth) -> PlayTypeSelection:
        """Calculate appropriate play type"""
        play = PlayTypeSelection()

        if truth.range_state.state == RangeStateType.TREND:
            if health.tier == HealthTier.HEALTHY:
                play.play_type = PlayType.TREND_CONTINUATION
                play.suitability_score = 85
            else:
                play.play_type = PlayType.SCALP
                play.suitability_score = 60
            play.alternatives = ["BREAKOUT"]
        elif truth.range_state.state == RangeStateType.BALANCE:
            if health.tier != HealthTier.UNSTABLE:
                play.play_type = PlayType.MEAN_REVERSION
                play.suitability_score = 75
                play.alternatives = ["SCALP"]
            else:
                play.play_type = PlayType.NO_TRADE
                play.suitability_score = 20
        else:
            play.play_type = PlayType.SCALP
            play.suitability_score = 50
            play.alternatives = ["NO_TRADE"]

        return play

    def _calculate_confidence_risk(self, truth: Phase1Truth,
                                    health: Phase2SignalHealth,
                                    density: Phase3SignalDensity) -> ConfidenceRisk:
        """Calculate confidence and risk parameters"""
        conf = ConfidenceRisk()

        # Base confidence from aggregated scores
        conf.confidence = int((truth.aggregation.truth_score + health.health_score + density.density_score) / 3)

        # Risk-reward from range
        if truth.core12.atr_14 > 0 and truth.range_state.range_width_atr > 0:
            conf.risk_reward_ratio = truth.range_state.range_width_atr / 2  # Rough estimate

        # Position size modifier
        if health.tier == HealthTier.HEALTHY and health.risk_state.state == RiskStateType.NORMAL:
            conf.position_size_modifier = 1.0
        elif health.tier == HealthTier.DEGRADED or health.risk_state.state == RiskStateType.REDUCED:
            conf.position_size_modifier = 0.5
        else:
            conf.position_size_modifier = 0.25

        # Max loss in ATR
        conf.max_loss_atr = 1.5 if conf.confidence >= 70 else 1.0

        return conf

    def _calculate_invalidation(self, truth: Phase1Truth) -> Invalidation:
        """Calculate invalidation levels"""
        inv = Invalidation()

        # Price invalidation levels
        if truth.range_state.range_high > 0:
            inv.price_levels.append(truth.range_state.range_high)
        if truth.range_state.range_low > 0:
            inv.price_levels.append(truth.range_state.range_low)

        # Add failed levels
        if truth.failures.failed_breakout_level:
            inv.price_levels.append(truth.failures.failed_breakout_level)

        # Time limit
        inv.time_limit_bars = 60  # 1 hour default

        # Conditions
        inv.conditions = [
            "Break of acceptance level",
            "MTF conflict emergence",
            "Failure pattern activation",
            "Volume conviction collapse"
        ]

        return inv


# ============================================================
# PHASE 5 ENGINE
# ============================================================

class Phase5Engine:
    """
    Phase 5: LEARNING_FORECASTING - Predictions, forecasts, entry/exit.
    INVARIANT: This is the ONLY phase allowed to use ML predictions.
    """

    def __init__(self, model_version: str = "v6_time_split"):
        self.model_version = model_version
        self.last_training_date = None

    def calculate(self, truth: Phase1Truth,
                  health: Phase2SignalHealth,
                  density: Phase3SignalDensity,
                  posture: Phase4ExecutionPosture,
                  bars_1m: pd.DataFrame = None,
                  v6_predictions: Dict = None) -> Phase5LearningForecasting:
        """Calculate Phase 5 Learning/Forecasting."""
        learning = Phase5LearningForecasting()
        learning.model_version = self.model_version
        learning.last_training_date = self.last_training_date

        # Create forecast horizons
        learning.horizons = self._create_horizons(truth, posture, v6_predictions)

        # Generate forecast
        learning.forecast = self._generate_forecast(truth, posture, v6_predictions)

        # Calculate entry/exit with triple barrier
        if posture.allowed:
            learning.entry_exit = self._calculate_entry_exit(truth, posture, bars_1m)

        return learning

    def _create_horizons(self, truth: Phase1Truth,
                          posture: Phase4ExecutionPosture,
                          v6_preds: Dict = None) -> List[ForecastHorizon]:
        """Create multi-horizon forecasts"""
        horizons = []

        # Define horizons: 15m, 1h, EOD
        horizon_configs = [
            ("15m", 15),
            ("1h", 60),
            ("EOD", 390)  # Roughly a full trading day
        ]

        for name, bars in horizon_configs:
            h = ForecastHorizon()
            h.horizon_name = name
            h.horizon_bars = bars

            # Base probabilities from Phase 4 posture
            if posture.bias.bias == Bias.LONG:
                h.direction_prob = {"UP": 0.55, "DOWN": 0.25, "FLAT": 0.20}
                h.expected_move_pct = 0.002 * (bars / 60)  # Scale with time
            elif posture.bias.bias == Bias.SHORT:
                h.direction_prob = {"UP": 0.25, "DOWN": 0.55, "FLAT": 0.20}
                h.expected_move_pct = -0.002 * (bars / 60)
            else:
                h.direction_prob = {"UP": 0.33, "DOWN": 0.33, "FLAT": 0.34}
                h.expected_move_pct = 0.0

            # Integrate V6 model predictions if available
            if v6_preds and name == "EOD":
                target_a_prob = v6_preds.get('target_a_prob', 0.5)
                target_b_prob = v6_preds.get('target_b_prob', 0.5)

                # Blend V6 predictions
                h.direction_prob["UP"] = (h.direction_prob["UP"] + target_a_prob) / 2
                h.direction_prob["DOWN"] = 1 - h.direction_prob["UP"] - 0.1
                h.direction_prob["FLAT"] = 0.1
                h.confidence = (target_a_prob + target_b_prob) / 2
            else:
                h.confidence = posture.confidence_risk.confidence / 100

            horizons.append(h)

        return horizons

    def _generate_forecast(self, truth: Phase1Truth,
                            posture: Phase4ExecutionPosture,
                            v6_preds: Dict = None) -> Forecast:
        """Generate regime and direction forecast"""
        forecast = Forecast()

        # Regime probabilities
        if truth.range_state.state == RangeStateType.TREND:
            forecast.regime_probs.trend = 0.6
            forecast.regime_probs.mean_reversion = 0.15
            forecast.regime_probs.chop = 0.15
            forecast.regime_probs.event = 0.1
            forecast.regime_probs.dominant = RegimeType.TREND
        elif truth.range_state.state == RangeStateType.BALANCE:
            forecast.regime_probs.trend = 0.15
            forecast.regime_probs.mean_reversion = 0.5
            forecast.regime_probs.chop = 0.25
            forecast.regime_probs.event = 0.1
            forecast.regime_probs.dominant = RegimeType.MEAN_REVERSION
        else:
            forecast.regime_probs.trend = 0.2
            forecast.regime_probs.mean_reversion = 0.2
            forecast.regime_probs.chop = 0.5
            forecast.regime_probs.event = 0.1
            forecast.regime_probs.dominant = RegimeType.CHOP

        # Next bar direction
        if posture.bias.bias == Bias.LONG:
            forecast.next_bar_direction = "UP"
            forecast.next_bar_confidence = posture.bias.strength
        elif posture.bias.bias == Bias.SHORT:
            forecast.next_bar_direction = "DOWN"
            forecast.next_bar_confidence = posture.bias.strength
        else:
            forecast.next_bar_direction = "FLAT"
            forecast.next_bar_confidence = 0.3

        # Trend strength
        if truth.mtf_continuity.aligned and truth.range_state.state == RangeStateType.TREND:
            forecast.trend_strength = 0.7
        elif truth.range_state.state == RangeStateType.TREND:
            forecast.trend_strength = 0.5
        else:
            forecast.trend_strength = 0.2

        return forecast

    def _calculate_entry_exit(self, truth: Phase1Truth,
                               posture: Phase4ExecutionPosture,
                               bars: pd.DataFrame = None) -> EntryExit:
        """Calculate entry/exit with triple barrier method"""
        entry_exit = EntryExit()

        if bars is None or len(bars) == 0:
            return entry_exit

        current_price = float(bars['close'].values[-1])
        atr = truth.core12.atr_14 if truth.core12.atr_14 > 0 else current_price * 0.01

        entry_exit.optimal_entry_level = current_price

        # Entry type based on regime
        if posture.play_type.play_type == PlayType.TREND_CONTINUATION:
            entry_exit.optimal_entry_type = "MARKET"
        else:
            entry_exit.optimal_entry_type = "LIMIT"

        # Triple barrier
        tb = TripleBarrier()
        tb.entry_price = current_price

        if posture.bias.bias == Bias.LONG:
            tb.profit_target = current_price + (atr * 2)
            tb.stop_loss = current_price - (atr * 1)
        elif posture.bias.bias == Bias.SHORT:
            tb.profit_target = current_price - (atr * 2)
            tb.stop_loss = current_price + (atr * 1)
        else:
            tb.profit_target = current_price + (atr * 1)
            tb.stop_loss = current_price - (atr * 1)

        tb.time_limit_bars = 60  # 1 hour

        # Outcome probabilities based on confidence
        conf = posture.confidence_risk.confidence / 100
        tb.outcome_probabilities = {
            "PROFIT_TARGET": 0.4 + (conf * 0.2),
            "STOP_LOSS": 0.3 - (conf * 0.1),
            "TIME_EXIT": 0.3 - (conf * 0.1)
        }

        if tb.outcome_probabilities["PROFIT_TARGET"] > tb.outcome_probabilities["STOP_LOSS"]:
            tb.expected_outcome = TripleBarrierOutcome.PROFIT_TARGET
        else:
            tb.expected_outcome = TripleBarrierOutcome.STOP_LOSS

        entry_exit.triple_barrier = tb
        entry_exit.expected_hold_bars = 30  # 30 minutes average

        return entry_exit


# ============================================================
# FULL RPE PIPELINE
# ============================================================

class RPEPipeline:
    """
    Complete Reality Proof Engine Pipeline.
    Orchestrates all 5 phases with strict layering invariants.
    """

    def __init__(self, model_version: str = "v6_time_split"):
        self.meta = RPEMeta()
        self.invariants = LayeringInvariants()

        self.phase1 = Phase1Engine()
        self.phase2 = Phase2Engine()
        self.phase3 = Phase3Engine()
        self.phase4 = Phase4Engine()
        self.phase5 = Phase5Engine(model_version)

    def run(self, symbol: str, bars_1m: pd.DataFrame,
            daily_bars: pd.DataFrame = None,
            signals_last_10m: int = 0,
            time_since_acceptance_minutes: int = 0,
            v6_predictions: Dict = None) -> Dict[str, Any]:
        """
        Run the complete 5-phase pipeline.

        Returns:
            Dict with meta, invariants, and phase1-5 outputs
        """
        # Phase 1: Truth (NO predictions, pure observation)
        truth = self.phase1.calculate(symbol, bars_1m, daily_bars)

        # Phase 2: Signal Health (NO predictions, data quality + conditions)
        health = self.phase2.calculate(
            truth, bars_1m,
            time_since_acceptance_minutes=time_since_acceptance_minutes
        )

        # Phase 3: Signal Density (NO predictions, spam control)
        density = self.phase3.calculate(
            truth, health,
            signals_last_10m=signals_last_10m
        )

        # Phase 4: Execution Posture (NO predictions, bias + play type)
        posture = self.phase4.calculate(truth, health, density)

        # Phase 5: Learning/Forecasting (ONLY phase with predictions)
        learning = self.phase5.calculate(
            truth, health, density, posture,
            bars_1m, v6_predictions
        )

        return {
            'meta': self.meta.to_dict(),
            'invariants': self.invariants.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'phase1_truth': truth.to_dict(),
            'phase2_signal_health': health.to_dict(),
            'phase3_signal_density': density.to_dict(),
            'phase4_execution_posture': posture.to_dict(),
            'phase5_learning_forecasting': learning.to_dict()
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def analyze_market_rpe(symbol: str, bars_1m: pd.DataFrame,
                       daily_bars: pd.DataFrame = None,
                       v6_predictions: Dict = None) -> Dict[str, Any]:
    """
    Convenience function to run the full RPE pipeline.
    """
    pipeline = RPEPipeline()
    return pipeline.run(symbol, bars_1m, daily_bars, v6_predictions=v6_predictions)
