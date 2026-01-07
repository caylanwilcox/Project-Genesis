"""
Inference and Daily Range Plan Generation
Produces actionable trading plans with calibrated probabilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import joblib
import os
import json

from config import (
    TARGET_LABELS, FIRST_TOUCH_CLASSES, ALL_FEATURES,
    TargetConfig, DataConfig
)
from features import FeatureEngineer, IntradayFeatureEngineer, calculate_unit
from labels import SessionLevels
from calibrate import ProbabilityCalibrator
from data_loader import SessionBuilder


@dataclass
class DailyRangePlan:
    """Complete daily range plan with probabilities and levels"""
    date: str
    symbol: str

    # Anchor and unit
    vwap: float
    unit: float
    unit_description: str

    # Long targets
    t1_long: float
    t2_long: float
    t3_long: float
    sl_long: float

    # Short targets
    t1_short: float
    t2_short: float
    t3_short: float
    sl_short: float

    # Calibrated probabilities
    prob_t1_long: float
    prob_t2_long: float
    prob_t3_long: float
    prob_sl_long: float
    prob_t1_short: float
    prob_t2_short: float
    prob_t3_short: float
    prob_sl_short: float

    # First touch probabilities
    first_touch_probs: Dict[str, float]

    # Recommendation
    bias: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float
    rationale: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'date': self.date,
            'symbol': self.symbol,
            'vwap': self.vwap,
            'unit': self.unit,
            'unit_description': self.unit_description,
            'levels': {
                'long': {
                    't1': self.t1_long,
                    't2': self.t2_long,
                    't3': self.t3_long,
                    'stop_loss': self.sl_long,
                },
                'short': {
                    't1': self.t1_short,
                    't2': self.t2_short,
                    't3': self.t3_short,
                    'stop_loss': self.sl_short,
                }
            },
            'probabilities': {
                'long': {
                    'touch_t1': self.prob_t1_long,
                    'touch_t2': self.prob_t2_long,
                    'touch_t3': self.prob_t3_long,
                    'touch_sl': self.prob_sl_long,
                },
                'short': {
                    'touch_t1': self.prob_t1_short,
                    'touch_t2': self.prob_t2_short,
                    'touch_t3': self.prob_t3_short,
                    'touch_sl': self.prob_sl_short,
                }
            },
            'first_touch_probs': self.first_touch_probs,
            'recommendation': {
                'bias': self.bias,
                'confidence': self.confidence,
                'rationale': self.rationale,
            }
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"  SPY DAILY RANGE PLAN - {self.date}",
            f"═══════════════════════════════════════════════════════════",
            f"",
            f"  ANCHOR: VWAP @ ${self.vwap:.2f}",
            f"  UNIT: ${self.unit:.2f} ({self.unit_description})",
            f"",
            f"  ─────────────────────────────────────────────────────────",
            f"  LONG SCENARIO (Buy at VWAP)",
            f"  ─────────────────────────────────────────────────────────",
            f"    T1 (0.5u): ${self.t1_long:.2f}  →  {self.prob_t1_long*100:.0f}% probability",
            f"    T2 (1.0u): ${self.t2_long:.2f}  →  {self.prob_t2_long*100:.0f}% probability",
            f"    T3 (1.5u): ${self.t3_long:.2f}  →  {self.prob_t3_long*100:.0f}% probability",
            f"    SL (1.25u): ${self.sl_long:.2f}  →  {self.prob_sl_long*100:.0f}% probability",
            f"",
            f"  ─────────────────────────────────────────────────────────",
            f"  SHORT SCENARIO (Sell at VWAP)",
            f"  ─────────────────────────────────────────────────────────",
            f"    T1 (0.5u): ${self.t1_short:.2f}  →  {self.prob_t1_short*100:.0f}% probability",
            f"    T2 (1.0u): ${self.t2_short:.2f}  →  {self.prob_t2_short*100:.0f}% probability",
            f"    T3 (1.5u): ${self.t3_short:.2f}  →  {self.prob_t3_short*100:.0f}% probability",
            f"    SL (1.25u): ${self.sl_short:.2f}  →  {self.prob_sl_short*100:.0f}% probability",
            f"",
            f"  ─────────────────────────────────────────────────────────",
            f"  FIRST TOUCH PROBABILITIES",
            f"  ─────────────────────────────────────────────────────────",
        ]

        # Sort first touch by probability
        sorted_touches = sorted(
            self.first_touch_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for touch, prob in sorted_touches[:5]:
            lines.append(f"    {touch}: {prob*100:.1f}%")

        lines.extend([
            f"",
            f"  ─────────────────────────────────────────────────────────",
            f"  RECOMMENDATION: {self.bias} (Confidence: {self.confidence*100:.0f}%)",
            f"  ─────────────────────────────────────────────────────────",
            f"  {self.rationale}",
            f"═══════════════════════════════════════════════════════════",
        ])

        return "\n".join(lines)


class PlanGenerator:
    """Generates daily range plans using trained models"""

    def __init__(
        self,
        model_dir: str = "models",
        data_config: Optional[DataConfig] = None,
        target_config: Optional[TargetConfig] = None
    ):
        self.model_dir = model_dir
        self.data_config = data_config or DataConfig()
        self.target_config = target_config or TargetConfig()

        # Load models
        self.binary_models = {}
        self.multiclass_model = None
        self.calibrator = ProbabilityCalibrator()

        self._load_models()

    def _load_models(self):
        """Load trained models and calibrators"""
        # Load binary models
        for target in TARGET_LABELS:
            model_path = os.path.join(self.model_dir, f"binary_{target}_final.joblib")
            if os.path.exists(model_path):
                self.binary_models[target] = joblib.load(model_path)

        # Load multiclass model
        multiclass_path = os.path.join(self.model_dir, "multiclass_first_touch_final.joblib")
        if os.path.exists(multiclass_path):
            self.multiclass_model = joblib.load(multiclass_path)

        # Load calibrators
        self.calibrator.load(self.model_dir)

    def calculate_levels(
        self,
        vwap: float,
        unit: float
    ) -> SessionLevels:
        """Calculate all price levels given VWAP and unit"""
        tc = self.target_config

        return SessionLevels(
            vwap=vwap,
            unit=unit,
            t1_long=vwap + tc.t1_units * unit,
            t2_long=vwap + tc.t2_units * unit,
            t3_long=vwap + tc.t3_units * unit,
            sl_long=vwap - tc.sl_units * unit,
            t1_short=vwap - tc.t1_units * unit,
            t2_short=vwap - tc.t2_units * unit,
            t3_short=vwap - tc.t3_units * unit,
            sl_short=vwap + tc.sl_units * unit,
        )

    def generate_plan(
        self,
        features: Dict[str, float],
        vwap: float,
        unit: float,
        date: str,
        symbol: str = "SPY"
    ) -> DailyRangePlan:
        """
        Generate a complete daily range plan

        Args:
            features: Feature dictionary
            vwap: Session VWAP (anchor)
            unit: ATR unit for level calculation
            date: Date string
            symbol: Ticker symbol

        Returns:
            DailyRangePlan object
        """
        # Calculate levels
        levels = self.calculate_levels(vwap, unit)

        # Create feature DataFrame
        feature_df = pd.DataFrame([features])

        # Get raw probabilities from models
        raw_probs = {}
        for target in TARGET_LABELS:
            if target in self.binary_models:
                prob = self.binary_models[target].predict_proba(
                    feature_df[ALL_FEATURES]
                )[0, 1]
                raw_probs[target] = prob

        # Get multiclass probabilities
        if self.multiclass_model is not None:
            multiclass_probs = self.multiclass_model.predict_proba(
                feature_df[ALL_FEATURES]
            )[0]
        else:
            multiclass_probs = np.ones(len(FIRST_TOUCH_CLASSES)) / len(FIRST_TOUCH_CLASSES)

        # Use raw model probabilities (already calibrated to base rates)
        calibrated_probs = {}
        for target in TARGET_LABELS:
            if target in raw_probs:
                # Clip to avoid extreme 0%/100%
                calibrated_probs[target] = np.clip(raw_probs[target], 0.01, 0.95)
            else:
                calibrated_probs[target] = 0.3

        # Calibrate multiclass
        if self.calibrator.multiclass_calibrators:
            multiclass_calibrated = self.calibrator.calibrate_multiclass(
                multiclass_probs.reshape(1, -1)
            )[0]
        else:
            multiclass_calibrated = multiclass_probs

        first_touch_probs = {
            name: prob
            for name, prob in zip(FIRST_TOUCH_CLASSES, multiclass_calibrated)
        }

        # Determine bias and confidence
        bias, confidence, rationale = self._determine_recommendation(
            calibrated_probs,
            first_touch_probs,
            features
        )

        # Determine unit description
        unit_description = self._describe_unit(unit, features)

        return DailyRangePlan(
            date=date,
            symbol=symbol,
            vwap=vwap,
            unit=unit,
            unit_description=unit_description,
            t1_long=levels.t1_long,
            t2_long=levels.t2_long,
            t3_long=levels.t3_long,
            sl_long=levels.sl_long,
            t1_short=levels.t1_short,
            t2_short=levels.t2_short,
            t3_short=levels.t3_short,
            sl_short=levels.sl_short,
            prob_t1_long=calibrated_probs.get('touch_t1_long', 0.5),
            prob_t2_long=calibrated_probs.get('touch_t2_long', 0.5),
            prob_t3_long=calibrated_probs.get('touch_t3_long', 0.5),
            prob_sl_long=calibrated_probs.get('touch_sl_long', 0.5),
            prob_t1_short=calibrated_probs.get('touch_t1_short', 0.5),
            prob_t2_short=calibrated_probs.get('touch_t2_short', 0.5),
            prob_t3_short=calibrated_probs.get('touch_t3_short', 0.5),
            prob_sl_short=calibrated_probs.get('touch_sl_short', 0.5),
            first_touch_probs=first_touch_probs,
            bias=bias,
            confidence=confidence,
            rationale=rationale,
        )

    def _determine_recommendation(
        self,
        probs: Dict[str, float],
        first_touch: Dict[str, float],
        features: Dict[str, float]
    ) -> Tuple[str, float, str]:
        """
        Determine trading bias and confidence

        Returns:
            Tuple of (bias, confidence, rationale)
        """
        # Calculate expected value for each direction
        # EV = P(hit target) * reward - P(hit stop) * risk

        # Long EV (simplified: T1 is 0.5u reward, SL is 1.25u risk)
        long_ev = (
            probs.get('touch_t1_long', 0) * 0.5 -
            probs.get('touch_sl_long', 0) * 1.25
        )

        # Short EV
        short_ev = (
            probs.get('touch_t1_short', 0) * 0.5 -
            probs.get('touch_sl_short', 0) * 1.25
        )

        # First touch advantage
        long_first = sum(
            first_touch.get(k, 0)
            for k in ['t1_long', 't2_long', 't3_long']
        )
        short_first = sum(
            first_touch.get(k, 0)
            for k in ['t1_short', 't2_short', 't3_short']
        )

        # Combine signals
        long_score = long_ev + 0.3 * long_first
        short_score = short_ev + 0.3 * short_first

        # Determine bias
        if long_score > 0.1 and long_score > short_score:
            bias = "LONG"
            confidence = min(long_score * 2, 1.0)
        elif short_score > 0.1 and short_score > long_score:
            bias = "SHORT"
            confidence = min(short_score * 2, 1.0)
        else:
            bias = "NEUTRAL"
            confidence = 0.5

        # Generate rationale
        rationale = self._generate_rationale(
            bias, probs, first_touch, features, long_ev, short_ev
        )

        return bias, confidence, rationale

    def _generate_rationale(
        self,
        bias: str,
        probs: Dict[str, float],
        first_touch: Dict[str, float],
        features: Dict[str, float],
        long_ev: float,
        short_ev: float
    ) -> str:
        """Generate explanation for recommendation"""
        parts = []

        if bias == "LONG":
            parts.append(f"Long EV ({long_ev:.2f}) exceeds short EV ({short_ev:.2f}).")
            parts.append(f"T1 long touch probability: {probs.get('touch_t1_long', 0)*100:.0f}%.")
            if probs.get('touch_sl_long', 0) < 0.3:
                parts.append("Low stop-loss risk.")

        elif bias == "SHORT":
            parts.append(f"Short EV ({short_ev:.2f}) exceeds long EV ({long_ev:.2f}).")
            parts.append(f"T1 short touch probability: {probs.get('touch_t1_short', 0)*100:.0f}%.")
            if probs.get('touch_sl_short', 0) < 0.3:
                parts.append("Low stop-loss risk.")

        else:
            parts.append("No clear directional edge detected.")
            parts.append("Consider waiting for better setup or reducing position size.")

        # Add context from features
        rsi = features.get('rsi_14', 50)
        if rsi > 70:
            parts.append(f"RSI overbought ({rsi:.0f}).")
        elif rsi < 30:
            parts.append(f"RSI oversold ({rsi:.0f}).")

        return " ".join(parts)

    def _describe_unit(self, unit: float, features: Dict[str, float]) -> str:
        """Describe how the unit was calculated"""
        atr_14 = features.get('atr_14', 0)
        or_atr_ratio = features.get('or_atr_ratio', 1)

        if or_atr_ratio > 1.2:
            return f"OR-driven (${unit:.2f}, {or_atr_ratio:.1f}x normal)"
        elif unit > atr_14 * 0.6:
            return f"Intraday ATR (${unit:.2f})"
        else:
            return f"Daily ATR based (${unit:.2f})"


def infer_today(
    model_dir: str = "models",
    api_key: Optional[str] = None
) -> DailyRangePlan:
    """
    Generate today's daily range plan

    Args:
        model_dir: Directory with trained models
        api_key: Polygon API key

    Returns:
        DailyRangePlan for today
    """
    from data_loader import PolygonDataLoader, SessionBuilder, DailyDataBuilder
    from features import FeatureEngineer

    # Load today's data
    loader = PolygonDataLoader(api_key)
    today = datetime.now().strftime('%Y-%m-%d')

    # Get recent daily data for indicators
    start_date = (datetime.now() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    daily_data = loader.fetch_intraday_bars(
        "SPY", start_date, today, "1", "day"
    )

    if daily_data.empty:
        raise ValueError("Could not fetch daily data")

    # Build daily indicators
    daily_builder = DailyDataBuilder()
    # Convert to proper format for daily builder
    daily_df = daily_data.set_index('timestamp').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    daily_df = daily_builder.add_daily_indicators(daily_df)

    # Get today's intraday data
    intraday_data = loader.fetch_intraday_bars(
        "SPY", today, today, "1", "minute"
    )

    if intraday_data.empty:
        raise ValueError("Could not fetch intraday data for today")

    # Build session
    session_builder = SessionBuilder()
    or_metrics = session_builder.get_opening_range(intraday_data)

    # Calculate VWAP
    vwap_series = session_builder.calculate_session_vwap(intraday_data)
    current_vwap = vwap_series.iloc[-1]

    # Calculate unit
    from features import calculate_intraday_atr
    or_range = or_metrics.get('or_range', 0) if or_metrics else 0
    intraday_atr = calculate_intraday_atr(intraday_data)
    daily_atr = daily_df['atr_14'].iloc[-1] if 'atr_14' in daily_df.columns else 1.0

    unit = calculate_unit(or_range, intraday_atr, daily_atr)

    # Calculate features
    engineer = FeatureEngineer()
    sessions = {today: intraday_data}
    features_df = engineer.build_feature_matrix(sessions, daily_df)

    if features_df.empty:
        raise ValueError("Could not calculate features")

    features = features_df.iloc[-1].to_dict()

    # Generate plan
    generator = PlanGenerator(model_dir)
    plan = generator.generate_plan(
        features=features,
        vwap=current_vwap,
        unit=unit,
        date=today,
        symbol="SPY"
    )

    return plan


if __name__ == "__main__":
    # Example with mock data
    print("Generating sample plan with mock data...")

    # Mock features
    features = {
        'open_to_vwap_pct': 0.1,
        'or_high_to_vwap_pct': 0.3,
        'or_low_to_vwap_pct': -0.2,
        'or_range_pct': 0.5,
        'prev_close_to_open_gap_pct': -0.1,
        'atr_14': 3.5,
        'atr_5': 4.0,
        'atr_ratio_5_14': 1.14,
        'or_atr_ratio': 0.8,
        'prev_day_range_pct': 1.2,
        'prev_day_body_pct': 0.6,
        'rsi_14': 55,
        'price_vs_sma_20': 0.5,
        'price_vs_sma_50': 2.0,
        'macd_hist': 0.3,
        'adx_14': 25,
        'volume_ratio_vs_20d_avg': 1.1,
        'or_volume_ratio': 1.2,
        'day_of_week': 2,
        'month': 1,
        'days_since_month_start': 15,
        'is_opex_week': 0,
        'is_fomc_day': 0,
        'vix_level': 18,
        'vix_change_1d': -0.5,
        'spy_20d_return': 2.5,
        'spy_5d_return': 0.8,
    }

    # Create generator (will use default probabilities if models not found)
    generator = PlanGenerator(model_dir="models")

    # Generate plan
    plan = generator.generate_plan(
        features=features,
        vwap=475.50,
        unit=3.20,
        date="2024-01-15",
        symbol="SPY"
    )

    print(plan.summary())
    print("\nJSON output:")
    print(plan.to_json())
