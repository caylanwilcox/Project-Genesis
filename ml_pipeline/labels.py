"""
Label Generation for SPY Daily Range Plan Prediction
Generates target labels based on price touches relative to VWAP
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from config import DataConfig, TargetConfig, TARGET_LABELS, FIRST_TOUCH_CLASSES
from data_loader import SessionBuilder
from features import calculate_intraday_atr, calculate_unit


@dataclass
class SessionLevels:
    """Price levels for a trading session"""
    vwap: float
    unit: float
    t1_long: float   # VWAP + 0.5u
    t2_long: float   # VWAP + 1.0u
    t3_long: float   # VWAP + 1.5u
    sl_long: float   # VWAP - 1.25u
    t1_short: float  # VWAP - 0.5u
    t2_short: float  # VWAP - 1.0u
    t3_short: float  # VWAP - 1.5u
    sl_short: float  # VWAP + 1.25u


@dataclass
class SessionLabels:
    """Labels for a trading session"""
    # Binary touch labels (did price touch this level?)
    touch_t1_long: int
    touch_t2_long: int
    touch_t3_long: int
    touch_sl_long: int
    touch_t1_short: int
    touch_t2_short: int
    touch_t3_short: int
    touch_sl_short: int

    # First touch (which level was touched first?)
    first_touch: str  # One of FIRST_TOUCH_CLASSES

    # MFE/MAE for analysis
    mfe_long: float   # Maximum Favorable Excursion (long)
    mae_long: float   # Maximum Adverse Excursion (long)
    mfe_short: float
    mae_short: float

    # Actual realized move
    close_vs_vwap: float


class LabelGenerator:
    """Generates training labels from session data"""

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        target_config: Optional[TargetConfig] = None
    ):
        self.data_config = data_config or DataConfig()
        self.target_config = target_config or TargetConfig()
        self.session_builder = SessionBuilder(self.data_config)

    def calculate_session_levels(
        self,
        session_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        date_str: str
    ) -> Optional[SessionLevels]:
        """
        Calculate all price levels for a session

        Args:
            session_df: Intraday data for the session
            daily_df: Historical daily data
            date_str: Date string YYYY-MM-DD

        Returns:
            SessionLevels object or None if can't calculate
        """
        date = pd.Timestamp(date_str)

        # Get previous day's ATR
        prev_dates = daily_df.index[daily_df.index < date]
        if len(prev_dates) == 0:
            return None

        prev_date = prev_dates[-1]
        daily_atr = daily_df.loc[prev_date, 'atr_14']

        if pd.isna(daily_atr) or daily_atr <= 0:
            return None

        # Calculate OR metrics
        or_metrics = self.session_builder.get_opening_range(session_df)
        or_range = or_metrics.get('or_range', 0) if or_metrics else 0

        # Calculate intraday ATR from first hour of trading
        first_hour = session_df.iloc[:60] if len(session_df) >= 60 else session_df
        intraday_atr = calculate_intraday_atr(first_hour)

        # Calculate unit
        unit = calculate_unit(or_range, intraday_atr, daily_atr)

        if unit <= 0:
            unit = daily_atr * 0.5  # Fallback

        # Calculate session VWAP (anchor point)
        # Use OR VWAP as the anchor for the day's plan
        if or_metrics and 'or_vwap' in or_metrics:
            vwap = or_metrics['or_vwap']
        else:
            # Calculate from session data
            vwap_series = self.session_builder.calculate_session_vwap(session_df)
            vwap = vwap_series.iloc[15] if len(vwap_series) > 15 else vwap_series.iloc[-1]

        # Calculate levels
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

    def generate_session_labels(
        self,
        session_df: pd.DataFrame,
        levels: SessionLevels
    ) -> SessionLabels:
        """
        Generate all labels for a session

        Args:
            session_df: Full session intraday data
            levels: Calculated price levels

        Returns:
            SessionLabels object
        """
        highs = session_df['high']
        lows = session_df['low']
        closes = session_df['close']

        # Binary touch labels
        touch_t1_long = int((highs >= levels.t1_long).any())
        touch_t2_long = int((highs >= levels.t2_long).any())
        touch_t3_long = int((highs >= levels.t3_long).any())
        touch_sl_long = int((lows <= levels.sl_long).any())

        touch_t1_short = int((lows <= levels.t1_short).any())
        touch_t2_short = int((lows <= levels.t2_short).any())
        touch_t3_short = int((lows <= levels.t3_short).any())
        touch_sl_short = int((highs >= levels.sl_short).any())

        # Find first touch
        first_touch = self._find_first_touch(session_df, levels)

        # MFE/MAE calculation
        session_high = highs.max()
        session_low = lows.min()

        # For long position (entered at VWAP)
        mfe_long = (session_high - levels.vwap) / levels.unit
        mae_long = (levels.vwap - session_low) / levels.unit

        # For short position (entered at VWAP)
        mfe_short = (levels.vwap - session_low) / levels.unit
        mae_short = (session_high - levels.vwap) / levels.unit

        # Close vs VWAP
        final_close = closes.iloc[-1]
        close_vs_vwap = (final_close - levels.vwap) / levels.unit

        return SessionLabels(
            touch_t1_long=touch_t1_long,
            touch_t2_long=touch_t2_long,
            touch_t3_long=touch_t3_long,
            touch_sl_long=touch_sl_long,
            touch_t1_short=touch_t1_short,
            touch_t2_short=touch_t2_short,
            touch_t3_short=touch_t3_short,
            touch_sl_short=touch_sl_short,
            first_touch=first_touch,
            mfe_long=mfe_long,
            mae_long=mae_long,
            mfe_short=mfe_short,
            mae_short=mae_short,
            close_vs_vwap=close_vs_vwap,
        )

    def _find_first_touch(
        self,
        session_df: pd.DataFrame,
        levels: SessionLevels
    ) -> str:
        """
        Determine which level was touched first

        Returns one of FIRST_TOUCH_CLASSES
        """
        level_map = {
            't1_long': levels.t1_long,
            't2_long': levels.t2_long,
            't3_long': levels.t3_long,
            'sl_long': levels.sl_long,
            't1_short': levels.t1_short,
            't2_short': levels.t2_short,
            't3_short': levels.t3_short,
            'sl_short': levels.sl_short,
        }

        first_touch_idx = {}

        for name, level in level_map.items():
            if 'long' in name and name != 'sl_long':
                # Long targets: check highs
                touches = session_df[session_df['high'] >= level]
            elif name == 'sl_long':
                # Long stop: check lows
                touches = session_df[session_df['low'] <= level]
            elif 'short' in name and name != 'sl_short':
                # Short targets: check lows
                touches = session_df[session_df['low'] <= level]
            else:  # sl_short
                # Short stop: check highs
                touches = session_df[session_df['high'] >= level]

            if not touches.empty:
                first_touch_idx[name] = touches.index[0]

        if not first_touch_idx:
            return 'none'

        # Find the earliest touch
        first = min(first_touch_idx.items(), key=lambda x: x[1])
        return first[0]

    def build_label_dataframe(
        self,
        sessions: Dict[str, pd.DataFrame],
        daily_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build label DataFrame for all sessions

        Args:
            sessions: Dict of date -> session DataFrame
            daily_df: Daily data with indicators

        Returns:
            Tuple of (labels DataFrame, levels DataFrame)
        """
        all_labels = []
        all_levels = []

        for date_str in sorted(sessions.keys()):
            session_df = sessions[date_str]

            # Calculate levels
            levels = self.calculate_session_levels(session_df, daily_df, date_str)

            if levels is None:
                continue

            # Generate labels
            labels = self.generate_session_labels(session_df, levels)

            # Convert to dict
            label_dict = {
                'date': date_str,
                'touch_t1_long': labels.touch_t1_long,
                'touch_t2_long': labels.touch_t2_long,
                'touch_t3_long': labels.touch_t3_long,
                'touch_sl_long': labels.touch_sl_long,
                'touch_t1_short': labels.touch_t1_short,
                'touch_t2_short': labels.touch_t2_short,
                'touch_t3_short': labels.touch_t3_short,
                'touch_sl_short': labels.touch_sl_short,
                'first_touch': labels.first_touch,
                'mfe_long': labels.mfe_long,
                'mae_long': labels.mae_long,
                'mfe_short': labels.mfe_short,
                'mae_short': labels.mae_short,
                'close_vs_vwap': labels.close_vs_vwap,
            }
            all_labels.append(label_dict)

            level_dict = {
                'date': date_str,
                'vwap': levels.vwap,
                'unit': levels.unit,
                't1_long': levels.t1_long,
                't2_long': levels.t2_long,
                't3_long': levels.t3_long,
                'sl_long': levels.sl_long,
                't1_short': levels.t1_short,
                't2_short': levels.t2_short,
                't3_short': levels.t3_short,
                'sl_short': levels.sl_short,
            }
            all_levels.append(level_dict)

        labels_df = pd.DataFrame(all_labels)
        labels_df['date'] = pd.to_datetime(labels_df['date'])
        labels_df = labels_df.set_index('date').sort_index()

        levels_df = pd.DataFrame(all_levels)
        levels_df['date'] = pd.to_datetime(levels_df['date'])
        levels_df = levels_df.set_index('date').sort_index()

        return labels_df, levels_df


def analyze_label_distribution(labels_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of labels

    Returns:
        Dictionary with label statistics
    """
    stats = {}

    # Binary touch rates
    binary_cols = [col for col in labels_df.columns if col.startswith('touch_')]
    for col in binary_cols:
        stats[f'{col}_rate'] = labels_df[col].mean()

    # First touch distribution
    first_touch_dist = labels_df['first_touch'].value_counts(normalize=True)
    stats['first_touch_distribution'] = first_touch_dist.to_dict()

    # MFE/MAE stats
    stats['mfe_long_mean'] = labels_df['mfe_long'].mean()
    stats['mfe_long_std'] = labels_df['mfe_long'].std()
    stats['mae_long_mean'] = labels_df['mae_long'].mean()
    stats['mae_long_std'] = labels_df['mae_long'].std()

    stats['mfe_short_mean'] = labels_df['mfe_short'].mean()
    stats['mfe_short_std'] = labels_df['mfe_short'].std()
    stats['mae_short_mean'] = labels_df['mae_short'].mean()
    stats['mae_short_std'] = labels_df['mae_short'].std()

    return stats


if __name__ == "__main__":
    # Test label generation
    from data_loader import load_and_prepare_data

    sessions, daily = load_and_prepare_data(
        start_year=2023,
        end_year=2024
    )

    generator = LabelGenerator()
    labels_df, levels_df = generator.build_label_dataframe(sessions, daily)

    print(f"\nLabels shape: {labels_df.shape}")
    print(f"\nLabel columns:\n{list(labels_df.columns)}")
    print(f"\nSample labels:\n{labels_df.tail()}")

    # Analyze distribution
    stats = analyze_label_distribution(labels_df)
    print(f"\nLabel statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.3f}")
        else:
            print(f"{key}: {value:.3f}")
