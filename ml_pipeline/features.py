"""
Feature Engineering for SPY Daily Range Plan Prediction
Generates features from session and daily data
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import calendar

from config import DataConfig, FEATURE_GROUPS, ALL_FEATURES
from data_loader import SessionBuilder, DailyDataBuilder


class FeatureEngineer:
    """Generates ML features for each trading session"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.session_builder = SessionBuilder(self.config)
        self.daily_builder = DailyDataBuilder(self.config)

    def _is_opex_week(self, date: datetime) -> bool:
        """Check if date is in options expiration week (3rd Friday)"""
        # Find third Friday
        c = calendar.Calendar(firstweekday=calendar.MONDAY)
        monthcal = c.monthdatescalendar(date.year, date.month)

        fridays = [
            day for week in monthcal for day in week
            if day.weekday() == calendar.FRIDAY and day.month == date.month
        ]

        if len(fridays) >= 3:
            third_friday = fridays[2]
            # Check if date is in the same week
            week_start = third_friday - timedelta(days=4)  # Monday
            week_end = third_friday + timedelta(days=2)    # Sunday
            return week_start <= date.date() <= week_end

        return False

    def _is_fomc_day(self, date: datetime) -> bool:
        """
        Check if date is an FOMC meeting day
        FOMC meets ~8 times per year, typically Wed of scheduled weeks
        This is a simplified check - in production, use a calendar API
        """
        # Simplified: FOMC typically meets in Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
        # Usually around 3rd week. This is approximate.
        fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}

        if date.month not in fomc_months:
            return False

        # Check if it's a Wednesday in the 3rd week
        if date.weekday() != 2:  # Wednesday
            return False

        day = date.day
        return 15 <= day <= 21

    def calculate_session_features(
        self,
        session_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        date_str: str,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate all features for a single trading session

        Args:
            session_df: Intraday data for the session
            daily_df: Historical daily data with indicators
            date_str: Date string YYYY-MM-DD
            vix_data: Optional VIX daily data

        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        date = pd.Timestamp(date_str)

        # Get previous day's data
        prev_dates = daily_df.index[daily_df.index < date]
        if len(prev_dates) == 0:
            return {}

        prev_date = prev_dates[-1]
        prev_day = daily_df.loc[prev_date]

        # Get current day open from session
        session_open = session_df['open'].iloc[0]

        # Calculate session VWAP
        session_vwap = self.session_builder.calculate_session_vwap(session_df)
        current_vwap = session_vwap.iloc[-1] if len(session_vwap) > 0 else session_open

        # Get Opening Range metrics
        or_metrics = self.session_builder.get_opening_range(session_df)

        # ============== Price Action Features ==============
        features['open_to_vwap_pct'] = (session_open - current_vwap) / current_vwap * 100

        if or_metrics:
            features['or_high_to_vwap_pct'] = (or_metrics['or_high'] - current_vwap) / current_vwap * 100
            features['or_low_to_vwap_pct'] = (or_metrics['or_low'] - current_vwap) / current_vwap * 100
            features['or_range_pct'] = or_metrics['or_range'] / session_open * 100
        else:
            features['or_high_to_vwap_pct'] = 0
            features['or_low_to_vwap_pct'] = 0
            features['or_range_pct'] = 0

        # Gap from previous close
        features['prev_close_to_open_gap_pct'] = (session_open - prev_day['close']) / prev_day['close'] * 100

        # ============== Volatility Features ==============
        features['atr_14'] = prev_day.get('atr_14', 0)
        features['atr_5'] = prev_day.get('atr_5', 0)
        features['atr_ratio_5_14'] = prev_day.get('atr_ratio_5_14', 1.0)

        # OR ATR ratio (how volatile was OR compared to expected)
        if or_metrics and features['atr_14'] > 0:
            features['or_atr_ratio'] = or_metrics['or_range'] / features['atr_14']
        else:
            features['or_atr_ratio'] = 1.0

        features['prev_day_range_pct'] = prev_day.get('day_range_pct', 0)
        features['prev_day_body_pct'] = prev_day.get('body_pct', 0)

        # ============== Momentum Features ==============
        features['rsi_14'] = prev_day.get('rsi_14', 50)
        features['price_vs_sma_20'] = prev_day.get('price_vs_sma_20', 0)
        features['price_vs_sma_50'] = prev_day.get('price_vs_sma_50', 0)
        features['macd_hist'] = prev_day.get('macd_hist', 0)
        features['adx_14'] = prev_day.get('adx_14', 25)

        # ============== Volume Features ==============
        # 20-day average volume
        vol_20d = daily_df.loc[:prev_date, 'volume'].tail(20).mean()
        features['volume_ratio_vs_20d_avg'] = prev_day['volume'] / vol_20d if vol_20d > 0 else 1.0

        # OR volume ratio
        if or_metrics:
            avg_or_vol = session_df['volume'].mean() * 15  # Estimate average 15-min volume
            features['or_volume_ratio'] = or_metrics['or_volume'] / avg_or_vol if avg_or_vol > 0 else 1.0
        else:
            features['or_volume_ratio'] = 1.0

        # ============== Time Features ==============
        features['day_of_week'] = date.dayofweek
        features['month'] = date.month
        features['days_since_month_start'] = date.day
        features['is_opex_week'] = 1.0 if self._is_opex_week(date) else 0.0
        features['is_fomc_day'] = 1.0 if self._is_fomc_day(date) else 0.0

        # ============== Market Regime Features ==============
        if vix_data is not None and date in vix_data.index:
            vix_row = vix_data.loc[date]
            features['vix_level'] = vix_row.get('close', 20)
            features['vix_change_1d'] = vix_row.get('return_1d', 0)
        else:
            features['vix_level'] = 20  # Default VIX
            features['vix_change_1d'] = 0

        features['spy_20d_return'] = prev_day.get('return_20d', 0)
        features['spy_5d_return'] = prev_day.get('return_5d', 0)

        return features

    def build_feature_matrix(
        self,
        sessions: Dict[str, pd.DataFrame],
        daily_df: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build feature matrix for all sessions

        Args:
            sessions: Dict of date -> session DataFrame
            daily_df: Daily data with indicators
            vix_data: Optional VIX data

        Returns:
            DataFrame with date index and feature columns
        """
        all_features = []

        for date_str in sorted(sessions.keys()):
            session_df = sessions[date_str]

            features = self.calculate_session_features(
                session_df,
                daily_df,
                date_str,
                vix_data
            )

            if features:
                features['date'] = date_str
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        feature_df = pd.DataFrame(all_features)
        feature_df['date'] = pd.to_datetime(feature_df['date'])
        feature_df = feature_df.set_index('date').sort_index()

        return feature_df

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names"""
        return ALL_FEATURES

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups dictionary"""
        return FEATURE_GROUPS


class IntradayFeatureEngineer:
    """
    Generates real-time features during a trading session
    Used for live inference
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.session_builder = SessionBuilder(self.config)

    def calculate_live_features(
        self,
        current_bars: pd.DataFrame,
        daily_history: pd.DataFrame,
        vix_current: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate features from live intraday data

        Args:
            current_bars: Current session's intraday bars so far
            daily_history: Historical daily data with indicators
            vix_current: Current VIX level

        Returns:
            Feature dictionary
        """
        if current_bars.empty or daily_history.empty:
            return {}

        # Use the FeatureEngineer but with current partial session
        engineer = FeatureEngineer(self.config)

        # Get today's date
        today = current_bars['timestamp'].iloc[0].strftime('%Y-%m-%d')

        # Create mock VIX data if we have current VIX
        vix_data = None
        if vix_current is not None:
            vix_data = pd.DataFrame({
                'close': [vix_current],
                'return_1d': [0]
            }, index=[pd.Timestamp(today)])

        features = engineer.calculate_session_features(
            current_bars,
            daily_history,
            today,
            vix_data
        )

        return features


def calculate_intraday_atr(session_df: pd.DataFrame) -> float:
    """
    Calculate intraday ATR from session bars

    Uses 14-period ATR on intraday bars to estimate current volatility
    """
    if len(session_df) < 15:
        return 0

    high = session_df['high']
    low = session_df['low']
    close = session_df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    return atr.iloc[-1] if pd.notna(atr.iloc[-1]) else 0


def calculate_unit(
    or_range: float,
    intraday_atr: float,
    daily_atr: float
) -> float:
    """
    Calculate the ATR unit for target/stop calculation

    Formula: unit = max(OR_15, intraday_ATR, 0.5 * daily_ATR)
    This ensures we have a meaningful unit even on quiet days
    """
    return max(or_range, intraday_atr, 0.5 * daily_atr)


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_and_prepare_data

    sessions, daily = load_and_prepare_data(
        start_year=2023,
        end_year=2024
    )

    engineer = FeatureEngineer()
    features_df = engineer.build_feature_matrix(sessions, daily)

    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"\nFeature columns:\n{list(features_df.columns)}")
    print(f"\nSample features:\n{features_df.tail()}")
    print(f"\nFeature statistics:\n{features_df.describe()}")
