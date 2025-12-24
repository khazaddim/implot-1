"""
Time series utilities for market-hour filtering, gap detection, and block analysis.

Extracted from Transform_Prototype.ipynb for reuse across the project.
"""
from __future__ import annotations

from functools import lru_cache
from datetime import datetime
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import requests
from pathlib import Path
from urllib.parse import urlencode

__all__ = [
    "get_timezone",
    "is_weekday_mask",
    "is_market_hour_mask",
    "is_market_hour_minute_mask",
    "median_delta_time",
    "find_large_gaps",
    "chunks",
    "find_gaps_and_chunks",
    "find_blocks",
    "align_chunks",
    "is_trading_day_mask",       # added
    "is_market_open_mask_pmc",   # added
]


@lru_cache(maxsize=16)
def get_timezone(tz_str: str):
    """
    Cached timezone getter.

    Parameters
    - tz_str: IANA timezone string (e.g., "US/Eastern").
    """
    return pytz.timezone(tz_str)


def is_weekday_mask(unix_times: np.ndarray, tz_str: str = "US/Eastern") -> np.ndarray:
    """
    Boolean mask for timestamps that fall on weekdays (Mon-Fri) in the provided timezone.

    Parameters
    - unix_times: Array of UNIX timestamps (seconds since epoch).
    - tz_str: Timezone string (default "US/Eastern").

    Supported Time Step Durations
    - Hourly
    - Minutely
    - Daily

    Returns
    - np.ndarray[bool]: True for Monday-Friday, False for Saturday/Sunday.
    """
    # Vectorized conversion via pandas is much faster than looping with datetime + pytz
    idx = pd.to_datetime(unix_times, unit="s", utc=True).tz_convert(tz_str)
    weekdays = idx.weekday.to_numpy()
    return (weekdays >= 0) & (weekdays <= 4)


def is_market_hour_mask(
    unix_times: np.ndarray,
    market_open: int = 9,
    market_close: int = 16,
    tz_str: str = "US/Eastern",
) -> np.ndarray:
    """
    Boolean mask for regular market hours by hour (default: 09:00-16:00 in the given timezone).

    Notes
    - Does not filter weekends or holidays; combine with `is_weekday_mask` and a holiday calendar if needed.

    Supported Time Step Durations
    - Hourly
    - Daily
    """
    # Vectorize to avoid slow per-element timezone conversions
    idx = pd.to_datetime(unix_times, unit="s", utc=True).tz_convert(tz_str)
    hours = idx.hour.to_numpy()
    return (hours >= market_open) & (hours < market_close)


def is_market_hour_minute_mask(
    unix_times: np.ndarray,
    market_open_hour: int = 9,
    market_open_minute: int = 30,
    market_close_hour: int = 16,
    market_close_minute: int = 0,
    tz_str: str = "US/Eastern",
) -> np.ndarray:
    """
    Boolean mask for market hours with minute precision.

    True only for timestamps where
      open <= time < close
    where open = market_open_hour:market_open_minute and close = market_close_hour:market_close_minute.

    Notes
    - Does not filter weekends or holidays; combine with `is_weekday_mask` and a holiday calendar if needed.

    Supported Time Step Durations
    - Minutely
    - Hourly
    - Daily
    """
    tz = get_timezone(tz_str)
    times = np.array([datetime.fromtimestamp(int(ts), tz) for ts in unix_times])
    opens = np.array([
        (dt.hour > market_open_hour)
        or (dt.hour == market_open_hour and dt.minute >= market_open_minute)
        for dt in times
    ])
    closes = np.array([
        (dt.hour < market_close_hour)
        or (dt.hour == market_close_hour and dt.minute < market_close_minute)
        for dt in times
    ])
    return opens & closes


# --- Gap utilities ---

def median_delta_time(series: pd.Series) -> int | float:
    """
    Median delta (in same units as the series) between successive values.

    Expects a pandas Series of integer-like UNIX timestamps.
    Returns the median of first differences, ignoring NaNs.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    values = series.astype(int)
    deltas = values.diff().dropna()
    return deltas.median()


def find_large_gaps(series: pd.Series) -> List[Dict[str, np.int64]]:
    """
    Find gaps where the delta exceeds the median delta.

    Returns a list of dicts: {'Type': 'gap', 'start': int, 'stop': int, 'duration': int}
    """
    median_delta = median_delta_time(series)
    if median_delta is None:
        return []
    deltas = series.diff().dropna()
    large_gaps = deltas[deltas > median_delta]
    gaps: List[Dict[str, np.int64]] = []
    for idx in large_gaps.index:
        start = series.loc[idx - 1]
        stop = series.loc[idx]
        duration = stop - start
        gaps.append({"Type": "gap", "start": start, "stop": stop, "duration": duration})
    return gaps


def chunks(gaps: List[Dict[str, np.int64]], start: int, end: int) -> List[Dict[str, int]]:
    """
    Build contiguous chunks between gaps, inclusive of the outer ranges.

    Returns a list of dicts: {'Type': 'chunk', 'start': int, 'stop': int, 'duration': int}
    """
    chunk_list: List[Dict[str, int]] = []
    if not gaps:
        chunk_list.append({"Type": "chunk", "start": int(start), "stop": int(end), "duration": int(end) - int(start)})
        return chunk_list
    for i in range(len(gaps) + 1):
        if i == 0:
            chunk_list.append({
                "Type": "chunk",
                "start": int(start),
                "stop": int(gaps[i]["start"]),
                "duration": int(gaps[i]["start"]) - int(start),
            })
        elif i == len(gaps):
            chunk_list.append({
                "Type": "chunk",
                "start": int(gaps[i - 1]["stop"]),
                "stop": int(end),
                "duration": int(end) - int(gaps[i - 1]["stop"]),
            })
        else:
            chunk_list.append({
                "Type": "chunk",
                "start": int(gaps[i - 1]["stop"]),
                "stop": int(gaps[i]["start"]),
                "duration": int(gaps[i]["start"]) - int(gaps[i - 1]["stop"]),
            })
    return chunk_list


def find_gaps_and_chunks(series: pd.Series | np.ndarray | Iterable[int]) -> List[Dict[str, int]]:
    """
    Find large gaps (based on median delta) and return a single list containing
    both gaps and chunks sorted by start time.
    Each element has keys: 'Type' (gap|chunk), 'start', 'stop', 'duration'.
    """
    # Accept numpy arrays or other iterable types by converting them to a pandas Series.
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        # Cover lists, tuples, generators, etc.
        series = pd.Series(list(series))

    # Normalize: drop NaNs, ensure integer-like unix timestamps, sort and reset index
    series = series.dropna()
    if series.empty:
        return []
    series = series.astype(int).sort_values().reset_index(drop=True)

    gaps = find_large_gaps(series)
    start = int(series.min())
    end = int(series.max())
    chunk_list = chunks(gaps, start, end)
    combined = gaps + chunk_list
    combined.sort(key=lambda x: x["start"])
    return combined


def find_blocks(mask: np.ndarray) -> List[Dict[str, int]]:
    """
    Find contiguous blocks of True or False in a boolean mask.

    Returns list of dicts: {'value': bool, 'start': int, 'stop': int, 'length': int}
    where [start, stop) is a half-open interval of indices into the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    changes = np.where(np.diff(mask) != 0)[0] + 1
    indices = np.concatenate(([0], changes, [len(mask)])).astype(int)
    blocks: List[Dict[str, int]] = []
    for i in range(len(indices) - 1):
        val = bool(mask[indices[i]])
        blocks.append({
            "value": val,
            "start": int(indices[i]),
            "stop": int(indices[i + 1]),
            "length": int(indices[i + 1] - indices[i]),
        })
    return blocks


def align_chunks(chunks_list: List[Dict[str, int]], data: pd.DataFrame) -> None:
    """
    For each chunk, set 'Time_Shift' in `data` between the row of chunk['start'] and chunk['stop']
    to the chunk's duration. Expects `data` to contain a 'Date' column of UNIX seconds.
    """
    if not chunks_list:
        return
    if "Date" not in data.columns or "Time_Shift" not in data.columns:
        raise KeyError("DataFrame must contain 'Date' and 'Time_Shift' columns")
    for chunk in chunks_list:
        if chunk["start"] == int(data["Date"].min()):
            continue
        # locate indices matching the start/stop times
        start_rows = data.index[data["Date"] == chunk["start"]].to_list()
        stop_rows = data.index[data["Date"] == chunk["stop"]].to_list()
        if not start_rows or not stop_rows:
            continue
        start_idx, stop_idx = start_rows[0], stop_rows[0]
        data.loc[start_idx:stop_idx, "Time_Shift"] = int(chunk["duration"])


def is_trading_day_mask(
    unix_times: np.ndarray,
    tz_str: str = "US/Eastern",
    calendar: str = "XNYS",
) -> np.ndarray:
    """
    True where the date is an exchange trading day (not a weekend/holiday).
    Uses pandas-market-calendars (e.g., calendar='XNYS', 'XNAS', etc.).

    Parameters
    - unix_times: UNIX timestamps (seconds since epoch)
    - tz_str: IANA timezone for interpreting dates (local exchange tz)
    - calendar: exchange calendar code for pandas-market-calendars

    Returns
    - np.ndarray[bool]: True for trading days per exchange calendar
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError as e:
        raise ImportError("Install pandas-market-calendars: py -m pip install pandas-market-calendars") from e

    idx_local = pd.to_datetime(unix_times, unit="s", utc=True).tz_convert(tz_str)
    start_date = idx_local.min().date()
    end_date = idx_local.max().date()

    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)  # market_open/close (UTC tz-aware)
    trading_days = pd.DatetimeIndex(schedule.index)  # tz-naive dates

    # Normalize to local date (drop time/tz), compare against schedule index
    days_local = idx_local.normalize().tz_localize(None)
    return days_local.isin(trading_days).to_numpy()


def is_market_open_mask_pmc(
    unix_times: np.ndarray,
    tz_str: str = "US/Eastern",
    calendar: str = "XNYS",
) -> np.ndarray:
    """
    True where timestamp is inside official open-close for the exchange day,
    holiday-aware and respecting early closes, via pandas-market-calendars.

    Parameters
    - unix_times: UNIX timestamps (seconds since epoch)
    - tz_str: IANA timezone for interpreting local exchange days
    - calendar: exchange calendar code (e.g., 'XNYS')

    Returns
    - np.ndarray[bool]: True when ts is within [market_open, market_close) per schedule
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError as e:
        raise ImportError("Install pandas-market-calendars: py -m pip install pandas-market-calendars") from e

    ts_utc = pd.to_datetime(unix_times, unit="s", utc=True)         # tz-aware UTC
    ts_local = ts_utc.tz_convert(tz_str)                            # local tz for day grouping
    start_date = ts_local.min().date()
    end_date = ts_local.max().date()

    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)  # UTC tz-aware times
    # Join each timestamp to its local trading day (tz-naive date) and compare against schedule in UTC
    days_local = ts_local.normalize().tz_localize(None)

    df = pd.DataFrame({"ts_utc": ts_utc, "day": days_local})
    df = df.join(schedule[["market_open", "market_close"]], on="day")
    mask = df["market_open"].notna() & (df["ts_utc"] >= df["market_open"]) & (df["ts_utc"] < df["market_close"])
    return mask.to_numpy()


DEFAULT_FMP_CONFIG = Path(r"C:\Users\khazy\OneDrive\Documents\keys\FMP\fmp_config.json")

class FMPClient:
    """
    Lightweight client for FinancialModelingPrep endpoints.
    Stores api key as a private attribute; safe to instantiate per-use.
    """
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[Path] = None):
        self._config_path = Path(config_path) if config_path else DEFAULT_FMP_CONFIG
        self._api_key = api_key or self._read_key_from_file(self._config_path)

    @staticmethod
    def _read_key_from_file(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(path)
        data = path.read_text(encoding="utf-8")
        obj = __import__("json").loads(data)
        for k in ("apikey", "api_key", "FMP_API_KEY", "key"):
            v = obj.get(k)
            if v:
                return str(v).strip()
        raise KeyError(f"No API key found in {path}")

    @property
    def api_key(self) -> str:
        # do not reveal full key when inspected
        return self._api_key

    @property
    def api_key_masked(self) -> str:
        if not self._api_key:
            return "<missing>"
        k = self._api_key
        return k[:4] + "..." + k[-4:]

    def build_holidays_url(self, exchange: str = "NASDAQ") -> str:
        
        base = "https://financialmodelingprep.com/stable/holidays-by-exchange"
        qs = urlencode({"exchange": exchange, "apikey": self._api_key})
        return f"{base}?{qs}"

    def fetch_holidays(self, exchange: str = "NASDAQ", timeout: int = 15) -> pd.DataFrame:
        url = self.build_holidays_url(exchange=exchange)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def clear_key(self):
        """Remove key from memory when no longer needed."""
        self._api_key = None


    def get_5min_chart(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        nonadjusted: Optional[bool] = None,
        timeout: int = 3,
    ) -> tuple[list[dict], str]:
         """
         Fetch 5-minute historical chart for `symbol` from FMP.
 
         Query parameters supported:
           - symbol* (string) e.g. "AAPL"
           - from (date)     e.g. "2024-01-01"
           - to (date)       e.g. "2024-03-01"
           - nonadjusted (bool) e.g. False
 
         Returns
         - (data, url) where data is the parsed JSON (usually list[dict]) and url
           is the full request URL that was used (useful for debugging/testing).
 
         Notes:
           - `from_date` and `to_date` must be 'YYYY-MM-DD' if provided.
           - `nonadjusted` will be sent as 'true'/'false' if provided.
         """
         if not self._api_key:
             raise ValueError("API key missing")
 
         # validate simple YYYY-MM-DD date strings
         def _validate_ymd(s: str, name: str) -> str:
             try:
                 datetime.strptime(s, "%Y-%m-%d")
             except Exception:
                 raise ValueError(f"{name} must be 'YYYY-MM-DD', got: {s!r}")
             return s
 
         base = "https://financialmodelingprep.com/stable/historical-chart/5min"
         params = {"symbol": symbol, "apikey": self._api_key}
         if from_date:
             params["from"] = _validate_ymd(from_date, "from_date")
         if to_date:
             params["to"] = _validate_ymd(to_date, "to_date")
         if nonadjusted is not None:
             # API expects lowercase 'true'/'false' strings
             params["nonadjusted"] = "true" if nonadjusted else "false"
 
         # build full URL for debugging/testing; use it for the request so returned URL matches
         full_url = f"{base}?{urlencode(params)}"
 
         resp = requests.get(full_url, timeout=timeout)
         resp.raise_for_status()
         data = resp.json()
         return data, full_url


    # ugh, they changed this to premium plan only
    def get_1min_chart(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        nonadjusted: Optional[bool] = None,
        timeout: int = 10,
    ) -> tuple[list[dict], str]:
        """
        Fetch 1-minute historical chart for `symbol` from FMP.

        Query parameters supported:
        - symbol* (string) e.g. "AAPL"
        - from (date)     e.g. "2024-01-01"
        - to (date)       e.g. "2024-03-01"
        - nonadjusted (bool) e.g. False

        Returns
        - (data, url) where data is the parsed JSON (usually list[dict]) and url
        is the full request URL that was used (useful for debugging/testing).
        """
        if not self._api_key:
            raise ValueError("API key missing")

        def _validate_ymd(s: str, name: str) -> str:
            try:
                datetime.strptime(s, "%Y-%m-%d")
            except Exception:
                raise ValueError(f"{name} must be 'YYYY-MM-DD', got: {s!r}")
            return s

        base = "https://financialmodelingprep.com/stable/historical-chart/1min"
        params = {"symbol": symbol, "apikey": self._api_key}
        if from_date:
            params["from"] = _validate_ymd(from_date, "from_date")
        if to_date:
            params["to"] = _validate_ymd(to_date, "to_date")
        if nonadjusted is not None:
            params["nonadjusted"] = "true" if nonadjusted else "false"

        full_url = f"{base}?{urlencode(params)}"
        resp = requests.get(full_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data, full_url
