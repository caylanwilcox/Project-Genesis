"""
Polygon.io API Integration

All data fetching from Polygon API.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from ..config import POLYGON_API_KEY


def fetch_polygon_data(ticker: str, days: int = 100, ticker_type: str = 'stock') -> pd.DataFrame:
    """Fetch historical daily data from Polygon.io

    Args:
        ticker: Stock symbol or index name
        days: Number of days of history
        ticker_type: 'stock' for regular stocks, 'index' for indices like VIX

    Returns:
        DataFrame with OHLCV data
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # For indices like VIX, use I: prefix
    api_ticker = f"I:{ticker}" if ticker_type == 'index' else ticker

    url = f"https://api.polygon.io/v2/aggs/ticker/{api_ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') == 'ERROR':
        raise ValueError(f"Polygon API error for {ticker}: {data.get('error', 'Unknown error')}")

    if 'results' not in data or len(data['results']) == 0:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume'
    })
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    return df


def fetch_daily_bars(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch daily bars from Polygon

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of bar dicts with o, h, l, c, v, t keys
    """
    if not POLYGON_API_KEY:
        return []

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        print(f"Error fetching daily bars for {ticker}: {e}")
        return []


def fetch_hourly_bars(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch hourly bars from Polygon

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of bar dicts with o, h, l, c, v, t keys
    """
    if not POLYGON_API_KEY:
        return []

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        print(f"Error fetching hourly bars for {ticker}: {e}")
        return []


def fetch_weekly_bars(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch weekly bars from Polygon

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of bar dicts with o, h, l, c, v, t keys
    """
    if not POLYGON_API_KEY:
        return []

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/week/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        print(f"Error fetching weekly bars for {ticker}: {e}")
        return []


def fetch_minute_bars(ticker: str, date: str) -> list:
    """Fetch minute bars from Polygon

    Args:
        ticker: Stock symbol
        date: Date (YYYY-MM-DD)

    Returns:
        List of bar dicts with o, h, l, c, v, t keys
    """
    if not POLYGON_API_KEY:
        return []

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get('results', [])
    except Exception as e:
        print(f"Error fetching minute bars for {ticker}: {e}")
        return []


def fetch_intraday_snapshot(ticker: str) -> dict:
    """Fetch today's intraday snapshot: current price, today's high, today's low

    Uses Polygon's snapshot endpoint for real-time data.

    Returns:
        Dict with current_price, today_open, today_high, today_low, today_volume
    """
    if not POLYGON_API_KEY:
        return None

    try:
        # Use snapshot endpoint for current price
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') == 'OK' and data.get('ticker'):
            ticker_data = data['ticker']
            day_data = ticker_data.get('day', {})
            return {
                'current_price': ticker_data.get('lastTrade', {}).get('p') or day_data.get('c', 0),
                'today_open': day_data.get('o', 0),
                'today_high': day_data.get('h', 0),
                'today_low': day_data.get('l', 0),
                'today_volume': day_data.get('v', 0),
            }
    except Exception as e:
        print(f"Error fetching snapshot for {ticker}: {e}")

    # Fallback: use today's daily bar
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        bars = fetch_daily_bars(ticker, today, today)
        if bars:
            bar = bars[-1]
            return {
                'current_price': bar['c'],
                'today_open': bar['o'],
                'today_high': bar['h'],
                'today_low': bar['l'],
                'today_volume': bar['v'],
            }
    except Exception as e:
        print(f"Error fetching daily bar for {ticker}: {e}")

    return None
