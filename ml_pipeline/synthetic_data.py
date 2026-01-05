"""
Synthetic Data Generator for Testing and Development
Generates realistic SPY-like intraday and daily data
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pytz

ET = pytz.timezone('America/New_York')


class SyntheticDataGenerator:
    """
    Generates synthetic market data for testing the ML pipeline

    Creates realistic price patterns including:
    - Daily OHLCV with proper relationships
    - Intraday 1-minute bars with volume profiles
    - Trending and mean-reverting regimes
    - Realistic volatility clustering
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        # SPY-like parameters
        self.base_price = 450.0
        self.daily_volatility = 0.012  # ~1.2% daily vol
        self.intraday_volatility = 0.0003  # Per-minute vol

    def generate_daily_data(
        self,
        start_date: str,
        end_date: str,
        start_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate daily OHLCV data with technical indicators

        Args:
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            start_price: Starting price (default: 450)

        Returns:
            DataFrame with daily OHLCV and indicators
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Generate trading days (exclude weekends)
        dates = pd.bdate_range(start, end)

        if start_price is None:
            start_price = self.base_price

        # Generate returns with volatility clustering (GARCH-like)
        n_days = len(dates)
        returns = np.zeros(n_days)
        vol = np.ones(n_days) * self.daily_volatility

        # Add regime changes
        regime = 0  # 0: normal, 1: high vol, -1: low vol
        for i in range(n_days):
            # Regime switching
            if np.random.random() < 0.02:  # 2% chance of regime change
                regime = np.random.choice([-1, 0, 1])

            # Volatility based on regime
            if regime == 1:
                vol[i] = self.daily_volatility * 1.8
            elif regime == -1:
                vol[i] = self.daily_volatility * 0.6
            else:
                vol[i] = self.daily_volatility

            # Generate return with slight mean reversion
            drift = -0.0001 if i > 20 and np.sum(returns[i-20:i]) > 0.05 else 0.0003
            returns[i] = np.random.normal(drift, vol[i])

        # Calculate prices
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLC
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            prev_close = prices[i-1] if i > 0 else start_price
            daily_vol = vol[i]

            # Open with gap
            gap = np.random.normal(0, daily_vol * 0.3)
            open_price = prev_close * (1 + gap)

            # High/Low around close
            high_ext = abs(np.random.normal(0, daily_vol))
            low_ext = abs(np.random.normal(0, daily_vol))

            high = max(open_price, close) * (1 + high_ext)
            low = min(open_price, close) * (1 - low_ext)

            # Volume with some randomness
            base_volume = 80_000_000  # SPY-like volume
            volume = int(base_volume * (0.7 + 0.6 * np.random.random()))

            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df = df.set_index('date')

        # Add technical indicators
        df = self._add_indicators(df)

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to daily data"""
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        df['atr_14'] = tr.rolling(14).mean()
        df['atr_5'] = tr.rolling(5).mean()
        df['atr_ratio_5_14'] = df['atr_5'] / df['atr_14']

        # Range metrics
        df['day_range'] = df['high'] - df['low']
        df['day_range_pct'] = df['day_range'] / df['close'] * 100
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close'] * 100

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ADX (simplified)
        df['adx_14'] = 25 + np.random.normal(0, 5, len(df))  # Simplified
        df['adx_14'] = df['adx_14'].clip(10, 60)

        # Returns
        df['return_1d'] = df['close'].pct_change() * 100
        df['return_5d'] = df['close'].pct_change(5) * 100
        df['return_20d'] = df['close'].pct_change(20) * 100

        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1) * 100

        return df

    def generate_intraday_session(
        self,
        date: str,
        open_price: float,
        daily_atr: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate 1-minute intraday data for a single session

        Args:
            date: Date string YYYY-MM-DD
            open_price: Opening price
            daily_atr: Expected daily ATR (for realistic range)

        Returns:
            DataFrame with 1-minute OHLCV
        """
        if daily_atr is None:
            daily_atr = open_price * self.daily_volatility

        # Market hours: 9:30 AM - 4:00 PM ET
        date_obj = pd.Timestamp(date, tz=ET)
        market_open = date_obj.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = date_obj.replace(hour=16, minute=0, second=0, microsecond=0)

        # Generate minute timestamps
        timestamps = pd.date_range(
            market_open,
            market_close - timedelta(minutes=1),
            freq='1min'
        )
        n_bars = len(timestamps)

        # Generate price path with intraday patterns
        minute_vol = daily_atr / np.sqrt(n_bars) * 0.3

        # Create intraday volatility profile (U-shaped)
        vol_profile = self._create_vol_profile(n_bars)

        # Generate returns
        returns = np.zeros(n_bars)
        trend = np.random.choice([-1, 0, 1]) * 0.0001  # Daily trend

        for i in range(n_bars):
            # Add trend + mean reversion + noise
            mean_rev = -0.00002 * (np.sum(returns[:i]) if i > 0 else 0)
            returns[i] = trend + mean_rev + np.random.normal(0, minute_vol * vol_profile[i])

        # Convert to prices
        prices = open_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            prev = prices[i-1] if i > 0 else open_price

            # Intrabar range
            bar_vol = minute_vol * vol_profile[i]
            bar_range = abs(np.random.normal(0, bar_vol * 2)) * open_price

            if close >= prev:
                open_p = prev + np.random.uniform(0, close - prev)
                high = close + np.random.uniform(0, bar_range/2)
                low = open_p - np.random.uniform(0, bar_range/2)
            else:
                open_p = prev - np.random.uniform(0, prev - close)
                high = open_p + np.random.uniform(0, bar_range/2)
                low = close - np.random.uniform(0, bar_range/2)

            # Ensure OHLC consistency
            high = max(high, open_p, close)
            low = min(low, open_p, close)

            # Volume profile (also U-shaped with opening spike)
            base_vol = 50000
            if i < 30:  # Opening 30 minutes
                vol_mult = 3.0 - (i / 30) * 1.5
            elif i > n_bars - 30:  # Closing 30 minutes
                vol_mult = 1.5 + ((i - (n_bars - 30)) / 30) * 1.5
            else:
                vol_mult = 1.0 + 0.3 * np.random.random()

            volume = int(base_vol * vol_mult * (0.5 + np.random.random()))

            data.append({
                'timestamp': ts,
                'open': open_p,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)

        # Calculate VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return df

    def _create_vol_profile(self, n_bars: int) -> np.ndarray:
        """Create U-shaped intraday volatility profile"""
        x = np.linspace(0, 1, n_bars)
        # U-shape: high at open, low midday, high at close
        profile = 1.5 - 0.8 * np.sin(x * np.pi)
        # Normalize
        profile = profile / profile.mean()
        return profile

    def generate_sessions(
        self,
        daily_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate intraday sessions for all days in daily data

        Args:
            daily_df: Daily data with OHLC

        Returns:
            Dict of date -> session DataFrame
        """
        sessions = {}

        for date in daily_df.index:
            # Skip if not a weekday
            if date.dayofweek >= 5:
                continue

            open_price = daily_df.loc[date, 'open']
            daily_atr = daily_df.loc[date, 'atr_14']

            if pd.isna(daily_atr):
                daily_atr = daily_df['atr_14'].mean()

            date_str = date.strftime('%Y-%m-%d')
            sessions[date_str] = self.generate_intraday_session(
                date_str,
                open_price,
                daily_atr if not pd.isna(daily_atr) else None
            )

        return sessions

    def generate_complete_dataset(
        self,
        start_year: int = 2020,
        end_year: int = 2024
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Generate complete dataset for pipeline testing

        Returns:
            Tuple of (sessions dict, daily DataFrame)
        """
        daily = self.generate_daily_data(
            f"{start_year}-01-01",
            f"{end_year}-12-31",
            450.0
        )

        # Generate sessions for a subset (to keep test fast)
        # Only generate sessions for last 2 years
        recent_daily = daily[daily.index.year >= end_year - 1]
        sessions = self.generate_sessions(recent_daily)

        return sessions, daily


def generate_vix_data(
    daily_df: pd.DataFrame,
    base_vix: float = 20.0
) -> pd.DataFrame:
    """
    Generate synthetic VIX data correlated with market volatility

    Args:
        daily_df: Daily market data
        base_vix: Base VIX level

    Returns:
        DataFrame with VIX OHLCV
    """
    # VIX tends to spike on down moves and high volatility
    returns = daily_df['close'].pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100

    # VIX is correlated with realized vol but with mean reversion
    vix_level = base_vix + (realized_vol - realized_vol.mean()) * 0.5
    vix_level = vix_level.clip(10, 80)

    # Add noise and spikes on down days
    down_days = returns < -0.01
    vix_level[down_days] = vix_level[down_days] * 1.2

    # Create OHLCV
    vix_data = pd.DataFrame(index=daily_df.index)
    vix_data['close'] = vix_level
    vix_data['open'] = vix_level.shift(1).fillna(base_vix)
    vix_data['high'] = vix_level * (1 + np.random.uniform(0, 0.05, len(vix_level)))
    vix_data['low'] = vix_level * (1 - np.random.uniform(0, 0.05, len(vix_level)))
    vix_data['volume'] = 0  # VIX doesn't have traditional volume
    vix_data['return_1d'] = vix_data['close'].pct_change() * 100

    return vix_data


if __name__ == "__main__":
    # Test synthetic data generation
    generator = SyntheticDataGenerator(seed=42)

    # Generate daily data
    daily = generator.generate_daily_data(
        "2023-01-01",
        "2024-01-31",
        450.0
    )
    print(f"Generated {len(daily)} daily bars")
    print(f"Columns: {list(daily.columns)}")
    print(f"\nDaily sample:\n{daily.tail()}")

    # Generate sessions
    sessions = generator.generate_sessions(daily.tail(30))
    print(f"\nGenerated {len(sessions)} intraday sessions")

    # Show one session
    first_session = list(sessions.values())[0]
    print(f"\nSession sample ({len(first_session)} bars):\n{first_session.head(10)}")

    # Generate VIX
    vix = generate_vix_data(daily)
    print(f"\nVIX sample:\n{vix.tail()}")
