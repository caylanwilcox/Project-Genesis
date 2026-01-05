"""
Data Loader V2 - More robust data fetching using requests
Handles Polygon API limitations and large data volumes
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests
import time
from tqdm import tqdm
import pytz

from config import DataConfig

ET = pytz.timezone('America/New_York')


class RobustPolygonLoader:
    """More robust data loader using direct requests"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found")
        self.base_url = "https://api.polygon.io"
        self.config = DataConfig()

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch daily bars - more reliable than minute data"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'limit': 50000,
            'sort': 'asc'
        }

        all_bars = []
        while url:
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if 'results' in data:
                    for bar in data['results']:
                        all_bars.append({
                            'timestamp': pd.Timestamp(bar['t'], unit='ms', tz='UTC'),
                            'open': bar['o'],
                            'high': bar['h'],
                            'low': bar['l'],
                            'close': bar['c'],
                            'volume': bar['v'],
                            'vwap': bar.get('vw'),
                        })

                # Handle pagination
                url = data.get('next_url')
                if url:
                    params = {'apiKey': self.api_key}
                time.sleep(0.15)  # Rate limit

            except Exception as e:
                print(f"Error fetching {start_date} to {end_date}: {e}")
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df['timestamp'] = df['timestamp'].dt.tz_convert(ET)
        return df.sort_values('timestamp').reset_index(drop=True)

    def fetch_minute_bars_day(
        self,
        symbol: str,
        date: str
    ) -> pd.DataFrame:
        """Fetch minute bars for a single day"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {
            'apiKey': self.api_key,
            'limit': 50000,
            'sort': 'asc'
        }

        all_bars = []
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if 'results' in data:
                for bar in data['results']:
                    all_bars.append({
                        'timestamp': pd.Timestamp(bar['t'], unit='ms', tz='UTC'),
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v'],
                        'vwap': bar.get('vw'),
                    })
        except Exception as e:
            print(f"Error fetching {date}: {e}")

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df['timestamp'] = df['timestamp'].dt.tz_convert(ET)
        return df.sort_values('timestamp').reset_index(drop=True)


class SessionBuilder:
    """Builds trading sessions from intraday data"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def _parse_time(self, time_str: str):
        parts = time_str.split(':')
        return int(parts[0]), int(parts[1])

    def _is_market_hours(self, ts: pd.Timestamp) -> bool:
        open_h, open_m = self._parse_time(self.config.market_open)
        close_h, close_m = self._parse_time(self.config.market_close)
        market_open = ts.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        market_close = ts.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
        return market_open <= ts < market_close

    def get_opening_range(self, session_df: pd.DataFrame) -> Dict[str, float]:
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

        if 'vwap' in or_data.columns and or_data['vwap'].notna().any():
            or_vwap = (or_data['vwap'] * or_data['volume']).sum() / or_data['volume'].sum()
        else:
            or_vwap = (or_data['close'] * or_data['volume']).sum() / or_data['volume'].sum()

        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_range': or_high - or_low,
            'or_vwap': or_vwap,
            'or_volume': or_data['volume'].sum()
        }

    def calculate_session_vwap(self, session_df: pd.DataFrame) -> pd.Series:
        if 'vwap' in session_df.columns and session_df['vwap'].notna().all():
            return session_df['vwap']
        typical_price = (session_df['high'] + session_df['low'] + session_df['close']) / 3
        cum_tp_vol = (typical_price * session_df['volume']).cumsum()
        cum_vol = session_df['volume'].cumsum()
        return cum_tp_vol / cum_vol


class DailyDataBuilder:
    """Builds daily summary data with indicators"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def add_daily_indicators(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['atr_14'] = tr.rolling(14).mean()
        df['atr_5'] = tr.rolling(5).mean()
        df['atr_ratio_5_14'] = df['atr_5'] / df['atr_14']

        # Price ranges
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
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        df['adx_14'] = dx.rolling(14).mean()

        # Returns
        df['return_1d'] = df['close'].pct_change() * 100
        df['return_5d'] = df['close'].pct_change(5) * 100
        df['return_20d'] = df['close'].pct_change(20) * 100

        # Gap
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1) * 100

        return df


def build_dataset_from_daily(
    api_key: Optional[str] = None,
    start_year: int = 2000,
    end_year: int = 2025,
    cache_dir: str = "data"
) -> pd.DataFrame:
    """
    Build dataset using daily data (more reliable than minute data)
    For intraday features, we'll estimate based on daily patterns
    """
    os.makedirs(cache_dir, exist_ok=True)

    loader = RobustPolygonLoader(api_key)
    builder = DailyDataBuilder()

    # Check cache
    cache_file = os.path.join(cache_dir, f"SPY_daily_{start_year}_{end_year}.parquet")
    if os.path.exists(cache_file):
        print(f"Loading cached daily data from {cache_file}")
        daily_df = pd.read_parquet(cache_file)
    else:
        print(f"Fetching SPY daily data {start_year}-{end_year}...")
        daily_df = loader.fetch_daily_bars(
            "SPY",
            f"{start_year}-01-01",
            f"{end_year}-12-31"
        )

        if daily_df.empty:
            raise ValueError("No data fetched")

        # Set date as index
        daily_df['date'] = daily_df['timestamp'].dt.date
        daily_df = daily_df.set_index('date')
        daily_df.index = pd.to_datetime(daily_df.index)

        # Save cache
        daily_df.to_parquet(cache_file)
        print(f"Cached to {cache_file}")

    # Add indicators
    daily_df = builder.add_daily_indicators(daily_df)

    return daily_df


if __name__ == "__main__":
    import sys

    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("Set POLYGON_API_KEY environment variable")
        sys.exit(1)

    daily = build_dataset_from_daily(
        api_key=api_key,
        start_year=2000,
        end_year=2025
    )

    print(f"\nDaily data shape: {daily.shape}")
    print(f"Date range: {daily.index.min()} to {daily.index.max()}")
    print(f"\nSample:\n{daily.tail()}")
