"""
Economic Calendar Module

Uses hardcoded event dates (publicly known schedules) for:
- FOMC (Federal Reserve decisions) - 8x/year
- CPI (Consumer Price Index) - monthly
- NFP (Non-Farm Payrolls) - first Friday each month

Price data fetched from Polygon.io (existing integration).
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz

# Data directory for caching
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, 'economic_events.json')
HISTORICAL_CACHE_FILE = os.path.join(DATA_DIR, 'historical_events.json')

# Event type definitions with typical announcement times (Eastern Time)
EVENT_TYPES = {
    # High Impact Events
    'FOMC': {
        'impact': 'high',
        'typical_time': '14:00',
        'keywords': ['FOMC', 'Federal Reserve', 'Fed Interest Rate', 'Fed Funds'],
        'description': 'Federal Reserve interest rate decision'
    },
    'CPI': {
        'impact': 'high',
        'typical_time': '08:30',
        'keywords': ['CPI', 'Consumer Price Index', 'Inflation Rate'],
        'description': 'Consumer inflation data'
    },
    'NFP': {
        'impact': 'high',
        'typical_time': '08:30',
        'keywords': ['Non-Farm Payroll', 'Nonfarm Payroll', 'Employment Change', 'Unemployment Rate'],
        'description': 'Jobs report - first Friday of month'
    },
    'GDP': {
        'impact': 'high',
        'typical_time': '08:30',
        'keywords': ['GDP', 'Gross Domestic Product'],
        'description': 'Economic growth data'
    },

    # Medium Impact Events
    'Retail Sales': {
        'impact': 'medium',
        'typical_time': '08:30',
        'keywords': ['Retail Sales'],
        'description': 'Consumer spending indicator'
    },
    'PPI': {
        'impact': 'medium',
        'typical_time': '08:30',
        'keywords': ['PPI', 'Producer Price Index'],
        'description': 'Producer inflation data'
    },
    'Fed Speech': {
        'impact': 'medium',
        'typical_time': 'varies',
        'keywords': ['Fed Chair', 'Powell', 'FOMC Member'],
        'description': 'Federal Reserve official speech'
    },
    'ISM Manufacturing': {
        'impact': 'medium',
        'typical_time': '10:00',
        'keywords': ['ISM Manufacturing', 'ISM PMI'],
        'description': 'Manufacturing activity index'
    },
    'ISM Services': {
        'impact': 'medium',
        'typical_time': '10:00',
        'keywords': ['ISM Services', 'ISM Non-Manufacturing'],
        'description': 'Services sector activity'
    },

    # Lower Impact Events
    'Initial Claims': {
        'impact': 'low',
        'typical_time': '08:30',
        'keywords': ['Initial Jobless Claims', 'Unemployment Claims'],
        'description': 'Weekly jobless claims'
    },
    'Durable Goods': {
        'impact': 'low',
        'typical_time': '08:30',
        'keywords': ['Durable Goods'],
        'description': 'Manufacturing orders'
    },
}


def fetch_finnhub_calendar(from_date: str, to_date: str) -> List[Dict]:
    """
    Fetch economic calendar from Finnhub API.

    Args:
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)

    Returns:
        List of economic events
    """
    if not FINNHUB_API_KEY:
        print("Warning: FINNHUB_API_KEY not set, using cached data only")
        return []

    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        'from': from_date,
        'to': to_date,
        'token': FINNHUB_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Filter for US events only
        events = data.get('economicCalendar', [])
        us_events = [e for e in events if e.get('country') == 'US']

        return us_events
    except Exception as e:
        print(f"Error fetching Finnhub calendar: {e}")
        return []


def classify_event(event: Dict) -> Optional[str]:
    """
    Classify an event into our event types based on keywords.

    Args:
        event: Finnhub event dict

    Returns:
        Event type string or None if not tracked
    """
    event_name = event.get('event', '').lower()

    for event_type, config in EVENT_TYPES.items():
        for keyword in config['keywords']:
            if keyword.lower() in event_name:
                return event_type

    return None


def parse_event(event: Dict) -> Optional[Dict]:
    """
    Parse a Finnhub event into our standard format.

    Args:
        event: Raw Finnhub event

    Returns:
        Standardized event dict or None
    """
    event_type = classify_event(event)
    if not event_type:
        return None

    config = EVENT_TYPES[event_type]

    # Parse time - Finnhub provides time in various formats
    time_str = event.get('time', config['typical_time'])
    if time_str == 'varies' or not time_str:
        time_str = config['typical_time']

    return {
        'date': event.get('date'),
        'time': time_str,
        'event_type': event_type,
        'event_name': event.get('event'),
        'impact': config['impact'],
        'actual': event.get('actual'),
        'estimate': event.get('estimate'),
        'previous': event.get('prev'),
        'unit': event.get('unit'),
        'description': config['description'],
    }


def fetch_and_cache_calendar(days_ahead: int = 30, days_back: int = 7) -> List[Dict]:
    """
    Fetch calendar and update cache.

    Args:
        days_ahead: Number of days to fetch ahead
        days_back: Number of days to fetch back (for recent actuals)

    Returns:
        List of parsed events
    """
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

    raw_events = fetch_finnhub_calendar(from_date, to_date)

    # Parse and filter events
    parsed_events = []
    for event in raw_events:
        parsed = parse_event(event)
        if parsed:
            parsed_events.append(parsed)

    # Sort by date and time
    parsed_events.sort(key=lambda x: (x['date'], x['time']))

    # Cache the results
    cache_data = {
        'fetched_at': datetime.now().isoformat(),
        'from_date': from_date,
        'to_date': to_date,
        'events': parsed_events
    }

    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached {len(parsed_events)} events to {CACHE_FILE}")
    except Exception as e:
        print(f"Error caching calendar: {e}")

    return parsed_events


def load_cached_calendar() -> List[Dict]:
    """Load calendar from cache file."""
    if not os.path.exists(CACHE_FILE):
        return []

    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        return data.get('events', [])
    except Exception as e:
        print(f"Error loading cached calendar: {e}")
        return []


def get_todays_events() -> List[Dict]:
    """
    Get all events for today.

    Returns:
        List of today's events sorted by time
    """
    today = datetime.now().strftime('%Y-%m-%d')
    events = load_cached_calendar()

    todays = [e for e in events if e['date'] == today]
    todays.sort(key=lambda x: x['time'])

    return todays


def get_upcoming_events(days: int = 7) -> List[Dict]:
    """
    Get upcoming events for the next N days.

    Args:
        days: Number of days to look ahead

    Returns:
        List of upcoming events
    """
    today = datetime.now()
    end_date = today + timedelta(days=days)

    today_str = today.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    events = load_cached_calendar()

    upcoming = [
        e for e in events
        if today_str <= e['date'] <= end_str
    ]
    upcoming.sort(key=lambda x: (x['date'], x['time']))

    return upcoming


def is_high_impact_day(date: Optional[str] = None) -> bool:
    """
    Check if a date has any high-impact events.

    Args:
        date: Date string (YYYY-MM-DD) or None for today

    Returns:
        True if high-impact event scheduled
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    events = load_cached_calendar()

    for event in events:
        if event['date'] == date and event['impact'] == 'high':
            return True

    return False


def get_event_context(date: Optional[str] = None) -> Dict:
    """
    Get full event context for a date, including timing info.

    Args:
        date: Date string or None for today

    Returns:
        Event context dict for model selection
    """
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    if date is None:
        date = now_et.strftime('%Y-%m-%d')

    events = load_cached_calendar()
    todays_events = [e for e in events if e['date'] == date]

    if not todays_events:
        return {
            'is_event_day': False,
            'events': [],
            'high_impact': False,
            'phase': None,
            'next_event': None,
            'hours_to_event': None,
            'event_type': None,
            'announcement_time': None,
        }

    # Find high-impact events
    high_impact_events = [e for e in todays_events if e['impact'] == 'high']

    # Determine phase (pre/post announcement)
    current_time = now_et.strftime('%H:%M')
    phase = None
    next_event = None
    hours_to_event = None

    for event in sorted(todays_events, key=lambda x: x['time']):
        event_time = event['time']
        if event_time == 'varies':
            continue

        if current_time < event_time:
            # We're before this event
            phase = 'pre'
            next_event = event

            # Calculate hours to event
            try:
                event_dt = datetime.strptime(f"{date} {event_time}", '%Y-%m-%d %H:%M')
                event_dt = et_tz.localize(event_dt)
                delta = event_dt - now_et
                hours_to_event = delta.total_seconds() / 3600
            except:
                hours_to_event = None
            break
    else:
        # All events have passed
        phase = 'post'
        if todays_events:
            next_event = todays_events[-1]  # Most recent event

    return {
        'is_event_day': True,
        'events': todays_events,
        'high_impact': len(high_impact_events) > 0,
        'phase': phase,
        'next_event': next_event,
        'hours_to_event': hours_to_event,
        'event_type': high_impact_events[0]['event_type'] if high_impact_events else todays_events[0]['event_type'],
        'announcement_time': next_event['time'] if next_event else None,
    }


# ============================================================
# Historical Data Functions (for model training)
# ============================================================

# Hardcoded FOMC dates for training (more reliable than API for historical)
FOMC_DATES = [
    # 2024
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    # 2023
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
    '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
    # 2022
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
    '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    # 2021
    '2021-01-27', '2021-03-17', '2021-04-28', '2021-06-16',
    '2021-07-28', '2021-09-22', '2021-11-03', '2021-12-15',
    # 2020
    '2020-01-29', '2020-03-03', '2020-03-15', '2020-04-29',
    '2020-06-10', '2020-07-29', '2020-09-16', '2020-11-05', '2020-12-16',
    # 2019
    '2019-01-30', '2019-03-20', '2019-05-01', '2019-06-19',
    '2019-07-31', '2019-09-18', '2019-10-30', '2019-12-11',
    # 2018
    '2018-01-31', '2018-03-21', '2018-05-02', '2018-06-13',
    '2018-08-01', '2018-09-26', '2018-11-08', '2018-12-19',
    # 2017
    '2017-02-01', '2017-03-15', '2017-05-03', '2017-06-14',
    '2017-07-26', '2017-09-20', '2017-11-01', '2017-12-13',
    # 2016
    '2016-01-27', '2016-03-16', '2016-04-27', '2016-06-15',
    '2016-07-27', '2016-09-21', '2016-11-02', '2016-12-14',
    # 2015
    '2015-01-28', '2015-03-18', '2015-04-29', '2015-06-17',
    '2015-07-29', '2015-09-17', '2015-10-28', '2015-12-16',
    # 2014
    '2014-01-29', '2014-03-19', '2014-04-30', '2014-06-18',
    '2014-07-30', '2014-09-17', '2014-10-29', '2014-12-17',
    # 2013
    '2013-01-30', '2013-03-20', '2013-05-01', '2013-06-19',
    '2013-07-31', '2013-09-18', '2013-10-30', '2013-12-18',
    # 2012
    '2012-01-25', '2012-03-13', '2012-04-25', '2012-06-20',
    '2012-08-01', '2012-09-13', '2012-10-24', '2012-12-12',
    # 2011
    '2011-01-26', '2011-03-15', '2011-04-27', '2011-06-22',
    '2011-08-09', '2011-09-21', '2011-11-02', '2011-12-13',
    # 2010
    '2010-01-27', '2010-03-16', '2010-04-28', '2010-06-23',
    '2010-08-10', '2010-09-21', '2010-11-03', '2010-12-14',
    # 2009
    '2009-01-28', '2009-03-18', '2009-04-29', '2009-06-24',
    '2009-08-12', '2009-09-23', '2009-11-04', '2009-12-16',
    # 2008
    '2008-01-22', '2008-01-30', '2008-03-18', '2008-04-30',
    '2008-06-25', '2008-08-05', '2008-09-16', '2008-10-08',
    '2008-10-29', '2008-12-16',
    # 2007
    '2007-01-31', '2007-03-21', '2007-05-09', '2007-06-28',
    '2007-08-07', '2007-09-18', '2007-10-31', '2007-12-11',
    # 2006
    '2006-01-31', '2006-03-28', '2006-05-10', '2006-06-29',
    '2006-08-08', '2006-09-20', '2006-10-25', '2006-12-12',
    # 2005
    '2005-02-02', '2005-03-22', '2005-05-03', '2005-06-30',
    '2005-08-09', '2005-09-20', '2005-11-01', '2005-12-13',
    # 2004
    '2004-01-28', '2004-03-16', '2004-05-04', '2004-06-30',
    '2004-08-10', '2004-09-21', '2004-11-10', '2004-12-14',
    # 2003
    '2003-01-29', '2003-03-18', '2003-05-06', '2003-06-25',
    '2003-08-12', '2003-09-16', '2003-10-28', '2003-12-09',
]

# 2025 FOMC dates
FOMC_DATES_2025 = [
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
]

# Add 2025 dates
FOMC_DATES.extend(FOMC_DATES_2025)


def get_historical_fomc_dates() -> List[str]:
    """Get all historical FOMC meeting dates."""
    return sorted(FOMC_DATES)


def is_fomc_day(date: str) -> bool:
    """Check if a date is an FOMC decision day."""
    return date in FOMC_DATES


def generate_nfp_dates(start_year: int = 2003, end_year: int = 2025) -> List[str]:
    """
    Generate NFP dates (first Friday of each month).

    Returns:
        List of NFP dates
    """
    nfp_dates = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Find first Friday
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            nfp_dates.append(first_friday.strftime('%Y-%m-%d'))

    return nfp_dates


def get_all_high_impact_dates() -> Dict[str, List[str]]:
    """
    Get all high-impact event dates for training.

    Returns:
        Dict mapping event type to list of dates
    """
    return {
        'FOMC': get_historical_fomc_dates(),
        'NFP': generate_nfp_dates(),
        # CPI/GDP dates would need to be fetched or hardcoded
    }


# ============================================================
# Initialization
# ============================================================

def initialize_calendar():
    """Initialize calendar on module load."""
    # Try to refresh cache if API key available
    if FINNHUB_API_KEY:
        print("Refreshing economic calendar from Finnhub...")
        fetch_and_cache_calendar(days_ahead=30, days_back=7)
    else:
        print("FINNHUB_API_KEY not set - using cached/hardcoded data")

        # Create a basic cache from hardcoded FOMC dates
        today = datetime.now().strftime('%Y-%m-%d')
        events = []

        # Include today and future FOMC dates
        for date in FOMC_DATES:
            if date >= today or date == today:
                events.append({
                    'date': date,
                    'time': '14:00',
                    'event_type': 'FOMC',
                    'event_name': 'FOMC Interest Rate Decision',
                    'impact': 'high',
                    'actual': None,
                    'estimate': None,
                    'previous': None,
                    'unit': None,
                    'description': 'Federal Reserve interest rate decision'
                })

        # Also add NFP dates
        nfp_dates = generate_nfp_dates(2024, 2025)
        for date in nfp_dates:
            if date >= today:
                events.append({
                    'date': date,
                    'time': '08:30',
                    'event_type': 'NFP',
                    'event_name': 'Non-Farm Payrolls',
                    'impact': 'high',
                    'actual': None,
                    'estimate': None,
                    'previous': None,
                    'unit': 'K',
                    'description': 'Jobs report - first Friday of month'
                })

        # Sort and dedupe
        events.sort(key=lambda x: (x['date'], x['time']))

        cache_data = {
            'fetched_at': datetime.now().isoformat(),
            'events': events[:20]  # Next 20 events
        }

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Created cache with {len(events[:20])} events")


# Run initialization when module loads
if __name__ != '__main__':
    initialize_calendar()


# ============================================================
# CLI for testing
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Economic Calendar Module Test")
    print("=" * 60)

    # Initialize
    initialize_calendar()

    # Test today's events
    print("\n--- Today's Events ---")
    todays = get_todays_events()
    if todays:
        for event in todays:
            print(f"  {event['time']} - {event['event_type']}: {event['event_name']}")
            print(f"           Impact: {event['impact']}")
    else:
        print("  No events today")

    # Test event context
    print("\n--- Event Context ---")
    context = get_event_context()
    print(f"  Is event day: {context['is_event_day']}")
    print(f"  High impact: {context['high_impact']}")
    print(f"  Phase: {context['phase']}")
    if context['hours_to_event']:
        print(f"  Hours to event: {context['hours_to_event']:.1f}")

    # Test upcoming events
    print("\n--- Upcoming Events (7 days) ---")
    upcoming = get_upcoming_events(7)
    for event in upcoming[:10]:
        print(f"  {event['date']} {event['time']} - [{event['impact'].upper()}] {event['event_type']}")

    # Test FOMC dates
    print("\n--- FOMC Dates ---")
    fomc_dates = get_historical_fomc_dates()
    print(f"  Total FOMC dates: {len(fomc_dates)}")
    print(f"  Date range: {fomc_dates[0]} to {fomc_dates[-1]}")

    # Check if today is FOMC
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"  Is today ({today}) FOMC? {is_fomc_day(today)}")
