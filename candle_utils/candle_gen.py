import numpy as np
from datetime import datetime, timezone

def generate_fake_candlestick_data(
    dates=None,
    base_price=150.0,
    volatility=0.02,
    seed=42,
    remove_weekends=True,
    length=30,
    start_date="2024-08-05",
    random=False,
    interval="daily"  # New kwarg: "daily", "hourly", or "minute"
):
    """
    Generate fake OHLC (Open, High, Low, Close) stock data for candlestick charts,
    optionally removing entries that fall on weekends.

    Parameters
    ----------
    dates : np.ndarray or None
        Array of UNIX timestamps. If None, generates sequential intervals starting from start_date.
    base_price : float
        Starting price for the stock.
    volatility : float
        Standard deviation of returns.
    seed : int
        Random seed for reproducibility.
    remove_weekends : bool
        If True, remove entries that fall on Saturday or Sunday.
    length : int
        Number of intervals to generate if dates is None.
    start_date : str
        Start date in 'YYYY-MM-DD' format (used if dates is None).
    interval : str
        "daily", "hourly", or "minute" candles.

    Returns
    -------
    dates : np.ndarray
    opens : np.ndarray
    highs : np.ndarray
    lows : np.ndarray
    closes : np.ndarray
    index : np.ndarray
        Continuous index for the data points.
    volume : np.ndarray
        Simulated trading volume data.
    """


    # Determine interval in seconds
    if interval == "daily":
        step = 86400
    elif interval == "hourly":
        step = 3600
    elif interval == "minute":
        step = 60
    else:
        raise ValueError("interval must be 'daily', 'hourly', or 'minute'")

    if dates is None:
        # Parse the start_date string to a datetime object
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        start = int(dt.replace(tzinfo=timezone.utc).timestamp())
        dates = np.array([start + step * i for i in range(length)])

    if not random:
        np.random.seed(seed)
    changes = np.random.normal(0.001, volatility, len(dates))
    changes += 0.002  # Upward trend

    closes = np.zeros(len(dates))
    closes[0] = base_price
    for i in range(1, len(dates)):
        closes[i] = closes[i-1] * (1 + changes[i])

    opens = np.zeros(len(dates))
    highs = np.zeros(len(dates))
    lows = np.zeros(len(dates))

    for i in range(len(dates)):
        if i == 0:
            opens[i] = base_price * (1 - volatility/2)
        else:
            opens[i] = closes[i-1] * (1 + np.random.normal(0, volatility/2))
        highs[i] = max(opens[i], closes[i]) * (1 + abs(np.random.normal(0, volatility)))
        lows[i] = min(opens[i], closes[i]) * (1 - abs(np.random.normal(0, volatility)))

    # set the seed for volume generation
    if not random:
        np.random.seed(seed + 1)  # Different seed for volume

    # Generate fake volume data
    volume = np.random.uniform(1, 20, len(dates))
    volume = np.clip(volume, 0, None)  # Ensure no negative volumes

    if remove_weekends:
        # Filter out weekends for any interval
        #old logic that would just remove weekends for daily data
        
        weekdays = np.array([
            datetime.fromtimestamp(ts, tz=timezone.utc).weekday() for ts in dates
        ])
        '''
        mask = (weekdays != 5) & (weekdays != 6)  # 5=Saturday, 6=Sunday
        dates = dates[mask]
        opens = opens[mask]
        highs = highs[mask]
        lows = lows[mask]
        closes = closes[mask]
        '''
        # new logic that shift all timestamps after the start of a weekend forward by 2 days
        # find the indices where weekends start
        # these will be rising edges in the weekday array
        weekend_starts = np.where((weekdays[:-1] < 5) & (weekdays[1:] >= 5))[0] + 1
        # for each weekend start, shift all subsequent timestamps into the future by 2 days
        for start_idx in weekend_starts:
            dates[start_idx:] += 2 * 86400  # shift by 2 days in seconds

    # Add a continuous index column
    index = np.arange(len(dates))

    return dates, opens, highs, lows, closes, index, volume
