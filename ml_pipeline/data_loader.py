"""
Data Loader and Sessionizer for SPY Intraday Data
Fetches data from Polygon.io and organizes into trading sessions
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from polygon import RESTClient
from tqdm import tqdm
import pytz
import time

from config import DataConfig

ET = pytz.timezone('America/New_York')


class PolygonDataLoader:
    """Loads intraday SPY data from Polygon.io API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        self.client = RESTClient(self.api_key)
        self.config = DataConfig()

    def fetch_intraday_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1",
        timespan: str = "minute"
    ) -> pd.DataFrame:
        """
        Fetch intraday bars from Polygon

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            timeframe: Bar multiplier (1, 5, etc.)
            timespan: Bar timespan (minute, hour, day)

        Returns:
            DataFrame with OHLCV data
        """
        all_bars = []

        try:
            bars = self.client.get_aggs(
                ticker=symbol,
                multiplier=int(timeframe),
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            )

            for bar in bars:
                all_bars.append({
                    'timestamp': pd.Timestamp(bar.timestamp, unit='ms', tz='UTC'),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': getattr(bar, 'vwap', None),
                    'transactions': getattr(bar, 'transactions', None)
                })
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df['timestamp'] = df['timestamp'].dt.tz_convert(ET)
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def fetch_year_data(self, symbol: str, year: int) -> pd.DataFrame:
        """Fetch full year of 1-minute intraday data"""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print(f"Fetching {symbol} data for {year}...")

        # Polygon has limits, fetch in monthly chunks
        all_data = []
        current_date = datetime(year, 1, 1)
        end = datetime(year, 12, 31)

        while current_date <= end:
            month_end = min(
                datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
                if current_date.month < 12
                else datetime(current_date.year, 12, 31),
                end
            )

            chunk = self.fetch_intraday_bars(
                symbol,
                current_date.strftime('%Y-%m-%d'),
                month_end.strftime('%Y-%m-%d')
            )

            if not chunk.empty:
                all_data.append(chunk)

            current_date = month_end + timedelta(days=1)
            time.sleep(0.25)  # Rate limiting

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def fetch_multi_year_data(
        self,
        symbol: str,
        start_year: int,
        end_year: int,
        cache_dir: str = "data"
    ) -> pd.DataFrame:
        """
        Fetch multiple years of data with caching

        Args:
            symbol: Ticker symbol
            start_year: First year to fetch
            end_year: Last year to fetch (inclusive)
            cache_dir: Directory to cache parquet files

        Returns:
            Combined DataFrame of all years
        """
        os.makedirs(cache_dir, exist_ok=True)
        all_data = []

        for year in tqdm(range(start_year, end_year + 1), desc="Loading years"):
            cache_file = os.path.join(cache_dir, f"{symbol}_{year}_1min.parquet")

            if os.path.exists(cache_file):
                print(f"Loading cached data for {year}")
                df = pd.read_parquet(cache_file)
            else:
                df = self.fetch_year_data(symbol, year)
                if not df.empty:
                    df.to_parquet(cache_file, index=False)
                    print(f"Cached {year} data to {cache_file}")

            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


class SessionBuilder:
    """Builds trading sessions from intraday data"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def _parse_time(self, time_str: str) -> Tuple[int, int]:
        """Parse HH:MM to hour, minute tuple"""
        parts = time_str.split(':')
        return int(parts[0]), int(parts[1])

    def _is_market_hours(self, ts: pd.Timestamp) -> bool:
        """Check if timestamp is during regular market hours"""
        open_h, open_m = self._parse_time(self.config.market_open)
        close_h, close_m = self._parse_time(self.config.market_close)

        market_open = ts.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        market_close = ts.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

        return market_open <= ts < market_close

    def sessionize(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split intraday data into daily trading sessions

        Args:
            df: DataFrame with 'timestamp' column in Eastern time

        Returns:
            Dictionary mapping date strings to session DataFrames
        """
        if df.empty:
            return {}

        # Filter to market hours only
        df = df[df['timestamp'].apply(self._is_market_hours)].copy()

        if df.empty:
            return {}

        # Extract date for grouping
        df['date'] = df['timestamp'].dt.date

        sessions = {}
        for date, group in df.groupby('date'):
            if len(group) >= self.config.min_session_bars:
                date_str = date.strftime('%Y-%m-%d')
                sessions[date_str] = group.reset_index(drop=True)

        return sessions

    def get_opening_range(self, session_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Opening Range (first 15 minutes) metrics

        Args:
            session_df: Single session DataFrame

        Returns:
            Dict with OR high, low, range, VWAP
        """
        if session_df.empty:
            return {}

        or_h, or_m = self._parse_time(self.config.or_end)
        first_ts = session_df['timestamp'].iloc[0]
        or_end = first_ts.replace(hour=or_h, minute=or_m, second=0, microsecond=0)

        or_data = session_df[session_df['timestamp'] <= or_end]

        if or_data.empty:
            return {}

        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_range = or_high - or_low

        # Calculate OR VWAP
        if 'vwap' in or_data.columns and or_data['vwap'].notna().any():
            or_vwap = (or_data['vwap'] * or_data['volume']).sum() / or_data['volume'].sum()
        else:
            or_vwap = (or_data['close'] * or_data['volume']).sum() / or_data['volume'].sum()

        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_range': or_range,
            'or_vwap': or_vwap,
            'or_volume': or_data['volume'].sum()
        }

    def calculate_session_vwap(self, session_df: pd.DataFrame) -> pd.Series:
        """Calculate cumulative VWAP for session"""
        if 'vwap' in session_df.columns and session_df['vwap'].notna().all():
            # Use provided VWAP
            return session_df['vwap']

        # Calculate from OHLC
        typical_price = (session_df['high'] + session_df['low'] + session_df['close']) / 3
        cum_tp_vol = (typical_price * session_df['volume']).cumsum()
        cum_vol = session_df['volume'].cumsum()

        return cum_tp_vol / cum_vol


class DailyDataBuilder:
    """Builds daily summary data with indicators"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def build_daily_bars(self, sessions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate sessions to daily OHLCV bars

        Args:
            sessions: Dict of date -> session DataFrame

        Returns:
            Daily OHLCV DataFrame
        """
        daily_data = []

        for date_str, session in sorted(sessions.items()):
            daily_data.append({
                'date': pd.Timestamp(date_str),
                'open': session['open'].iloc[0],
                'high': session['high'].max(),
                'low': session['low'].min(),
                'close': session['close'].iloc[-1],
                'volume': session['volume'].sum()
            })

        df = pd.DataFrame(daily_data)
        df = df.set_index('date').sort_index()

        return df

    def calculate_atr(self, daily_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = daily_df['high']
        low = daily_df['low']
        close = daily_df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_adx(self, daily_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = daily_df['high']
        low = daily_df['low']
        close = daily_df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def add_daily_indicators(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to daily data"""
        df = daily_df.copy()

        # ATR
        df['atr_14'] = self.calculate_atr(df, 14)
        df['atr_5'] = self.calculate_atr(df, 5)
        df['atr_ratio_5_14'] = df['atr_5'] / df['atr_14']

        # Price ranges
        df['day_range'] = df['high'] - df['low']
        df['day_range_pct'] = df['day_range'] / df['close'] * 100
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close'] * 100

        # Momentum
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['sma_20'] = self.calculate_sma(df['close'], 20)
        df['sma_50'] = self.calculate_sma(df['close'], 50)
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100

        # MACD
        macd, signal, hist = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # ADX
        df['adx_14'] = self.calculate_adx(df, 14)

        # Returns
        df['return_1d'] = df['close'].pct_change() * 100
        df['return_5d'] = df['close'].pct_change(5) * 100
        df['return_20d'] = df['close'].pct_change(20) * 100

        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1) * 100

        return df


def load_and_prepare_data(
    api_key: Optional[str] = None,
    start_year: int = 2004,
    end_year: int = 2024,
    cache_dir: str = "data"
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Main function to load and prepare all data

    Returns:
        Tuple of (sessions dict, daily DataFrame with indicators)
    """
    loader = PolygonDataLoader(api_key)
    session_builder = SessionBuilder()
    daily_builder = DailyDataBuilder()

    # Fetch raw data
    raw_data = loader.fetch_multi_year_data("SPY", start_year, end_year, cache_dir)

    if raw_data.empty:
        raise ValueError("No data fetched")

    # Build sessions
    sessions = session_builder.sessionize(raw_data)
    print(f"Built {len(sessions)} trading sessions")

    # Build daily data with indicators
    daily_df = daily_builder.build_daily_bars(sessions)
    daily_df = daily_builder.add_daily_indicators(daily_df)
    print(f"Built daily data with {len(daily_df)} bars")

    return sessions, daily_df


if __name__ == "__main__":
    # Test data loading
    sessions, daily = load_and_prepare_data(
        start_year=2023,
        end_year=2024
    )
    print(f"\nSessions: {len(sessions)}")
    print(f"Daily bars: {len(daily)}")
    print(f"\nDaily columns: {list(daily.columns)}")
    print(f"\nSample daily data:\n{daily.tail()}")
