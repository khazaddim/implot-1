"""A small, self-contained Python port of ImPlot's time-axis tick locator.

This is based on the logic in `implot.cpp` (function `Locator_Time`) and the
associated helpers in the "Time Ticks and Utils" section.

Goal
----
Given:
  - `t_min` and `t_max` as Unix timestamps in seconds (float or int)
  - `pixels` as the axis pixel length (float or int)

Produce:
  - tick positions (float seconds)
  - label strings (or hidden labels)
  - two label levels (0 = minor/top, 1 = major/bottom)

Notes / Differences vs ImPlot
-----------------------------
- ImPlot measures label widths in pixels using the current ImGui font.
  Python doesn't have that here, so we approximate width as:

      width_px ~= len(label) * char_px

  You can tune `char_px` if your UI font is wider/narrower.
- Month/year stepping matches ImPlot's approach: stepping by whole days
  (86400s * days-in-month / days-in-year) rather than doing timezone-aware
  calendar arithmetic.

This file is intended as a learning aid and a starting point for a Cython
implementation if you need speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import calendar
import math
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Enums / constants
# -----------------------------

# Match ImPlotTimeUnit_ ordering.
TIME_US = 0
TIME_MS = 1
TIME_S = 2
TIME_MIN = 3
TIME_HR = 4
TIME_DAY = 5
TIME_MO = 6
TIME_YR = 7
TIME_COUNT = 8

# Seconds per unit (months/years are treated specially; this is only used for
# density estimates and for pix_per_major_div calculations).
TIME_UNIT_SPANS: Tuple[float, ...] = (
    0.000001,
    0.001,
    1.0,
    60.0,
    3600.0,
    86400.0,
    2629800.0,
    31557600.0,
)

IMPLOT_MIN_TIME = 0
IMPLOT_MAX_TIME = 32503680000  # ~ 01/01/3000 UTC

MONTH_ABRVS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# -----------------------------
# Data structures
# -----------------------------

# These dataclasses mirror a small subset of ImPlot's internal time types.
# They are intentionally tiny and immutable (frozen=True) so they are safe to
# pass around and compare/sort without worrying about accidental mutation.

@dataclass(frozen=True, order=True)
class ImPlotTime:
    """Two-part timestamp used throughout the locator.

    Shape
    -----
    - `S`: whole seconds since Unix epoch (int)
    - `Us`: microseconds offset within the second (int)

    Invariants
    ----------
    `Us` is normalized in `__post_init__` such that `0 <= Us < 1_000_000`.

    Notes
    -----
    - `frozen=True` makes instances effectively immutable; `__post_init__`
        uses `object.__setattr__` to normalize fields during construction.
    - `order=True` enables sorting/comparisons by (S, Us).
    """

    S: int          # Whole seconds since epoch.
    Us: int = 0     # Microseconds within the current second (normalized).

    def __post_init__(self) -> None:
        # normalize so 0 <= Us < 1_000_000
        s = int(self.S)
        us = int(self.Us)
        s = s + us // 1_000_000
        us = us % 1_000_000
        object.__setattr__(self, "S", s)
        object.__setattr__(self, "Us", us)

    def roll_over(self) -> ImPlotTime:
        return ImPlotTime(self.S, self.Us)

    def to_double(self) -> float:
        return float(self.S) + float(self.Us) / 1_000_000.0

    @staticmethod
    def from_double(t: float) -> ImPlotTime:
        s = int(t)
        us = int(t * 1_000_000 - math.floor(t) * 1_000_000)
        return ImPlotTime(s, us)

    def __add__(self, other: ImPlotTime) -> ImPlotTime:
        return ImPlotTime(self.S + other.S, self.Us + other.Us)

    def __sub__(self, other: ImPlotTime) -> ImPlotTime:
        return ImPlotTime(self.S - other.S, self.Us - other.Us)


@dataclass(frozen=True)
class DateTimeSpec:
    """Formatting preset for date/time labels (ImPlotDateTimeSpec equivalent).

    Shape
    -----
    - `date_fmt`: one of the DATE_* constants (controls the date portion)
    - `time_fmt`: one of the TIMEFMT_* constants (controls the time portion)
    - `use_24_hour` / `use_iso8601`: optional style overrides

    These are used as presets in `TIME_FORMAT_LEVEL0/1` and then combined with
    the runtime `use_24_hour` / `use_iso8601` arguments when formatting.
    """

    date_fmt: int
    time_fmt: int
    use_24_hour: bool = False
    use_iso8601: bool = False


@dataclass(frozen=True)
class Tick:
    """A single tick produced by `locator_time`.

    Shape
    -----
    - `pos`: tick position as Unix timestamp in seconds (float)
    - `level`: label row (0 = minor/top, 1 = major/bottom)
    - `major`: whether this tick is a major division boundary
    - `show_label`: whether a label should be drawn
    - `label`: the formatted label text, or None if hidden
    """
    pos: float
    level: int           # 0 or 1
    major: bool
    show_label: bool
    label: Optional[str]


# -----------------------------
# Format enums
# -----------------------------

DATE_NONE = 0
DATE_DAY_MO = 1
DATE_DAY_MO_YR = 2
DATE_MO_YR = 3
DATE_MO = 4
DATE_YR = 5

TIMEFMT_NONE = 0
TIMEFMT_US = 1
TIMEFMT_SUS = 2
TIMEFMT_SMS = 3
TIMEFMT_S = 4
TIMEFMT_MIN_SMS = 5
TIMEFMT_HR_MIN_SMS = 6
TIMEFMT_HR_MIN_S = 7
TIMEFMT_HR_MIN = 8
TIMEFMT_HR = 9


# Default format presets (from implot.cpp)
TIME_FORMAT_LEVEL0: Tuple[DateTimeSpec, ...] = (
    DateTimeSpec(DATE_NONE, TIMEFMT_US),
    DateTimeSpec(DATE_NONE, TIMEFMT_SMS),
    DateTimeSpec(DATE_NONE, TIMEFMT_S),
    DateTimeSpec(DATE_NONE, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_NONE, TIMEFMT_HR),
    DateTimeSpec(DATE_DAY_MO, TIMEFMT_NONE),
    DateTimeSpec(DATE_MO, TIMEFMT_NONE),
    DateTimeSpec(DATE_YR, TIMEFMT_NONE),
)

TIME_FORMAT_LEVEL1: Tuple[DateTimeSpec, ...] = (
    DateTimeSpec(DATE_NONE, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_NONE, TIMEFMT_HR_MIN_S),
    DateTimeSpec(DATE_NONE, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_NONE, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_YR, TIMEFMT_NONE),
)

TIME_FORMAT_LEVEL1_FIRST: Tuple[DateTimeSpec, ...] = (
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_HR_MIN_S),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_HR_MIN_S),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_HR_MIN),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_DAY_MO_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_YR, TIMEFMT_NONE),
    DateTimeSpec(DATE_YR, TIMEFMT_NONE),
)


# -----------------------------
# Helper functions
# -----------------------------

def constrain_time(t: float) -> float:
    return max(float(IMPLOT_MIN_TIME), min(float(IMPLOT_MAX_TIME), float(t)))


def get_unit_for_range(span_seconds: float) -> int:
    """Match ImPlot's GetUnitForRange."""
    cutoffs = (0.001, 1.0, 60.0, 3600.0, 86400.0, 2629800.0, 31557600.0, IMPLOT_MAX_TIME)
    for i in range(TIME_COUNT):
        if span_seconds <= cutoffs[i]:
            return i
    return TIME_YR


def lower_bound_step(max_divs: int, divs: Sequence[int], step: Sequence[int]) -> int:
    if max_divs < divs[0]:
        return 0
    for i in range(1, len(divs)):
        if max_divs < divs[i]:
            return step[i - 1]
    return step[-1]


def get_time_step(max_divs: int, unit: int) -> int:
    """Match ImPlot's GetTimeStep."""
    if unit in (TIME_MS, TIME_US):
        step = (500, 250, 200, 100, 50, 25, 20, 10, 5, 2, 1)
        divs = (2, 4, 5, 10, 20, 40, 50, 100, 200, 500, 1000)
        return lower_bound_step(max_divs, divs, step)
    if unit in (TIME_S, TIME_MIN):
        step = (30, 15, 10, 5, 1)
        divs = (2, 4, 6, 12, 60)
        return lower_bound_step(max_divs, divs, step)
    if unit == TIME_HR:
        step = (12, 6, 3, 2, 1)
        divs = (2, 4, 8, 12, 24)
        return lower_bound_step(max_divs, divs, step)
    if unit == TIME_DAY:
        step = (14, 7, 2, 1)
        divs = (2, 4, 14, 28)
        return lower_bound_step(max_divs, divs, step)
    if unit == TIME_MO:
        step = (6, 3, 2, 1)
        divs = (2, 4, 6, 12)
        return lower_bound_step(max_divs, divs, step)
    return 0


def is_leap_year(year: int) -> bool:
    return calendar.isleap(year)


def get_days_in_month(year: int, month_0_11: int) -> int:
    # month_0_11: 0=Jan .. 11=Dec
    return calendar.monthrange(year, month_0_11 + 1)[1]


def get_time_fields(t: ImPlotTime, use_local_time: bool) -> time.struct_time:
    return time.localtime(t.S) if use_local_time else time.gmtime(t.S)


def mk_time_from_fields(fields: time.struct_time, use_local_time: bool) -> ImPlotTime:
    # Build seconds from a time.struct_time (microseconds handled elsewhere).
    if use_local_time:
        s = int(time.mktime(fields))
    else:
        s = int(calendar.timegm(fields))
    if s < 0:
        s = 0
    return ImPlotTime(s, 0)


def make_time(
    year: int,
    month_0_11: int = 0,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    us: int = 0,
    *,
    use_local_time: bool,
) -> ImPlotTime:
    """Match ImPlot::MakeTime (normalizes us into seconds, uses local/UTC based on style)."""
    # Normalize microseconds into seconds.
    second = int(second + us // 1_000_000)
    us = int(us % 1_000_000)

    # tm_year is years since 1900; but we pass real year to struct_time builder.
    # Construct a struct_time-like tuple: (Y, M, D, h, m, s, wday, yday, isdst)
    # wday/yday/isdst are ignored by mktime/timegm.
    tup = (year, month_0_11 + 1, day, hour, minute, second, 0, 0, -1)
    if use_local_time:
        s = int(time.mktime(tup))
    else:
        s = int(calendar.timegm(tup))
    if s < 0:
        s = 0
    return ImPlotTime(s, us)


def add_time(t: ImPlotTime, unit: int, count: int, *, use_local_time: bool) -> ImPlotTime:
    """Match ImPlot::AddTime semantics."""
    t_out = ImPlotTime(t.S, t.Us)
    if unit == TIME_US:
        return ImPlotTime(t_out.S, t_out.Us + count)
    if unit == TIME_MS:
        return ImPlotTime(t_out.S, t_out.Us + count * 1000)
    if unit == TIME_S:
        return ImPlotTime(t_out.S + count, t_out.Us)
    if unit == TIME_MIN:
        return ImPlotTime(t_out.S + count * 60, t_out.Us)
    if unit == TIME_HR:
        return ImPlotTime(t_out.S + count * 3600, t_out.Us)
    if unit == TIME_DAY:
        return ImPlotTime(t_out.S + count * 86400, t_out.Us)

    # Calendar-ish units: month/year. ImPlot implements this by repeatedly
    # converting to tm and adding 86400*days.
    if unit == TIME_MO:
        for _ in range(abs(count)):
            tm = get_time_fields(t_out, use_local_time)
            year = tm.tm_year
            mon = tm.tm_mon - 1  # 0-11
            if count > 0:
                t_out = ImPlotTime(t_out.S + 86400 * get_days_in_month(year, mon), t_out.Us)
            else:
                # Previous month
                prev_year = year - (1 if mon == 0 else 0)
                prev_mon = 11 if mon == 0 else mon - 1
                t_out = ImPlotTime(t_out.S - 86400 * get_days_in_month(prev_year, prev_mon), t_out.Us)
        return t_out

    if unit == TIME_YR:
        for _ in range(abs(count)):
            year = get_year(t_out, use_local_time=use_local_time)
            if count > 0:
                days = 365 + (1 if is_leap_year(year) else 0)
                t_out = ImPlotTime(t_out.S + 86400 * days, t_out.Us)
            else:
                days = 365 + (1 if is_leap_year(year - 1) else 0)
                t_out = ImPlotTime(t_out.S - 86400 * days, t_out.Us)
        return t_out

    return t_out


def floor_time(t: ImPlotTime, unit: int, *, use_local_time: bool) -> ImPlotTime:
    """Match ImPlot::FloorTime."""
    if unit == TIME_S:
        return ImPlotTime(t.S, 0)
    if unit == TIME_MS:
        return ImPlotTime(t.S, (t.Us // 1000) * 1000)
    if unit == TIME_US:
        return t

    tm = get_time_fields(t, use_local_time)
    year = tm.tm_year
    mon = tm.tm_mon
    mday = tm.tm_mday
    hour = tm.tm_hour
    minute = tm.tm_min
    sec = tm.tm_sec

    # fall-through behavior from ImPlot:
    if unit == TIME_YR:
        mon = 1
    if unit in (TIME_YR, TIME_MO):
        mday = 1
    if unit in (TIME_YR, TIME_MO, TIME_DAY):
        hour = 0
    if unit in (TIME_YR, TIME_MO, TIME_DAY, TIME_HR):
        minute = 0
    if unit in (TIME_YR, TIME_MO, TIME_DAY, TIME_HR, TIME_MIN):
        sec = 0

    tup = (year, mon, mday, hour, minute, sec, 0, 0, -1)
    if use_local_time:
        s = int(time.mktime(tup))
    else:
        s = int(calendar.timegm(tup))
    return ImPlotTime(s, 0)


def ceil_time(t: ImPlotTime, unit: int, *, use_local_time: bool) -> ImPlotTime:
    return add_time(floor_time(t, unit, use_local_time=use_local_time), unit, 1, use_local_time=use_local_time)


def get_year(t: ImPlotTime, *, use_local_time: bool) -> int:
    return get_time_fields(t, use_local_time).tm_year


def nice_num(x: float, round_: bool) -> float:
    """Port of ImPlot's NiceNum (used for 'nice' year intervals)."""
    if x == 0:
        return 0
    expv = math.floor(math.log10(abs(x)))
    f = abs(x) / (10 ** expv)
    if round_:
        if f < 1.5:
            nf = 1
        elif f < 3:
            nf = 2
        elif f < 7:
            nf = 5
        else:
            nf = 10
    else:
        if f <= 1:
            nf = 1
        elif f <= 2:
            nf = 2
        elif f <= 5:
            nf = 5
        else:
            nf = 10
    return math.copysign(nf * (10 ** expv), x)


def format_time_of_day(t: ImPlotTime, fmt: int, *, use_local_time: bool, use_24_hour: bool) -> str:
    tm = get_time_fields(t, use_local_time)
    us = t.Us % 1000
    ms = t.Us // 1000
    sec = tm.tm_sec
    minute = tm.tm_min

    if use_24_hour:
        hr = tm.tm_hour
        if fmt == TIMEFMT_US:
            return f".{ms:03d} {us:03d}"
        if fmt == TIMEFMT_SUS:
            return f":{sec:02d}.{ms:03d} {us:03d}"
        if fmt == TIMEFMT_SMS:
            return f":{sec:02d}.{ms:03d}"
        if fmt == TIMEFMT_S:
            return f":{sec:02d}"
        if fmt == TIMEFMT_MIN_SMS:
            return f":{minute:02d}:{sec:02d}.{ms:03d}"
        if fmt == TIMEFMT_HR_MIN_SMS:
            return f"{hr:02d}:{minute:02d}:{sec:02d}.{ms:03d}"
        if fmt == TIMEFMT_HR_MIN_S:
            return f"{hr:02d}:{minute:02d}:{sec:02d}"
        if fmt == TIMEFMT_HR_MIN:
            return f"{hr:02d}:{minute:02d}"
        if fmt == TIMEFMT_HR:
            return f"{hr:02d}:00"
        return ""

    # 12-hour clock
    ap = "am" if tm.tm_hour < 12 else "pm"
    hr = 12 if (tm.tm_hour == 0 or tm.tm_hour == 12) else (tm.tm_hour % 12)

    if fmt == TIMEFMT_US:
        return f".{ms:03d} {us:03d}"
    if fmt == TIMEFMT_SUS:
        return f":{sec:02d}.{ms:03d} {us:03d}"
    if fmt == TIMEFMT_SMS:
        return f":{sec:02d}.{ms:03d}"
    if fmt == TIMEFMT_S:
        return f":{sec:02d}"
    if fmt == TIMEFMT_MIN_SMS:
        return f":{minute:02d}:{sec:02d}.{ms:03d}"
    if fmt == TIMEFMT_HR_MIN_SMS:
        return f"{hr}:{minute:02d}:{sec:02d}.{ms:03d}{ap}"
    if fmt == TIMEFMT_HR_MIN_S:
        return f"{hr}:{minute:02d}:{sec:02d}{ap}"
    if fmt == TIMEFMT_HR_MIN:
        return f"{hr}:{minute:02d}{ap}"
    if fmt == TIMEFMT_HR:
        return f"{hr}{ap}"
    return ""


def format_date_part(t: ImPlotTime, fmt: int, *, use_local_time: bool, use_iso8601: bool) -> str:
    tm = get_time_fields(t, use_local_time)
    day = tm.tm_mday
    mon = tm.tm_mon
    year = tm.tm_year
    yr2 = year % 100

    if use_iso8601:
        if fmt == DATE_DAY_MO:
            return f"--{mon:02d}-{day:02d}"
        if fmt == DATE_DAY_MO_YR:
            return f"{year:d}-{mon:02d}-{day:02d}"
        if fmt == DATE_MO_YR:
            return f"{year:d}-{mon:02d}"
        if fmt == DATE_MO:
            return f"--{mon:02d}"
        if fmt == DATE_YR:
            return f"{year:d}"
        return ""

    # Default (non-ISO)
    if fmt == DATE_DAY_MO:
        return f"{mon:d}/{day:d}"
    if fmt == DATE_DAY_MO_YR:
        return f"{mon:d}/{day:d}/{yr2:02d}"
    if fmt == DATE_MO_YR:
        return f"{MONTH_ABRVS[mon-1]} {year:d}"
    if fmt == DATE_MO:
        return f"{MONTH_ABRVS[mon-1]}"
    if fmt == DATE_YR:
        return f"{year:d}"
    return ""


def format_datetime(t: ImPlotTime, spec: DateTimeSpec, *, use_local_time: bool, use_24_hour: bool, use_iso8601: bool) -> str:
    # Spec values are presets; style overrides are applied via arguments.
    date_str = "" if spec.date_fmt == DATE_NONE else format_date_part(t, spec.date_fmt, use_local_time=use_local_time, use_iso8601=use_iso8601)
    time_str = "" if spec.time_fmt == TIMEFMT_NONE else format_time_of_day(t, spec.time_fmt, use_local_time=use_local_time, use_24_hour=use_24_hour)

    if date_str and time_str:
        return f"{date_str} {time_str}"
    return date_str or time_str


def time_label_same_suffix(l1: str, l2: str) -> bool:
    n = min(len(l1), len(l2))
    return l1[-n:] == l2[-n:]


def estimate_label_width_px(spec: DateTimeSpec, *, char_px: float, use_local_time: bool, use_24_hour: bool, use_iso8601: bool) -> float:
    # Mimic ImPlot's "t_max_width" trick: pick a timestamp likely to yield wide strings.
    # ImPlot uses: MakeTime(2888,12,22,12,58,58,888888)
    t = make_time(2888, 11, 22, 12, 58, 58, 888888, use_local_time=use_local_time)
    s = format_datetime(t, spec, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
    return max(1.0, len(s) * char_px)


# -----------------------------
# Public API: locator_time
# -----------------------------

def locator_time(
    t_min: float,
    t_max: float,
    pixels: float,
    *,
    use_local_time: bool = False,
    use_24_hour: bool = False,
    use_iso8601: bool = False,
    max_density: float = 0.5,
    char_px: float = 7.0,
) -> List[Tick]:
    """Python port of ImPlot::Locator_Time.

    Parameters
    ----------
    t_min, t_max:
        Unix timestamps (seconds). Can be floats.
    pixels:
        Pixel length of the axis (e.g., plot width).
    use_local_time:
        If True, interpret timestamps in local time for formatting/flooring.
        If False, treat as UTC.
    use_24_hour, use_iso8601:
        Formatting toggles.
    max_density:
        Same meaning as in ImPlot: limits label density.
    char_px:
        Approximate pixels per character for width estimation.

    Returns
    -------
    A flat list of Tick objects. Each tick has:
      - pos (float seconds)
      - level (0 minor/top, 1 major/bottom)
      - major (bool)
      - show_label (bool)
      - label (Optional[str])

    You can split by `tick.level` if desired.
    """

    t_min = constrain_time(min(t_min, t_max))
    t_max = constrain_time(max(t_min, t_max))

    span = float(t_max - t_min)
    if pixels <= 0 or span <= 0:
        return []

    # STEP 0: unit selection (same heuristic as ImPlot)
    unit0 = get_unit_for_range(span / (pixels / 100.0))
    unit1 = min(unit0 + 1, TIME_COUNT - 1)

    # STEP 1: formatting specs (style overrides applied later)
    fmt0 = TIME_FORMAT_LEVEL0[unit0]
    fmt1 = TIME_FORMAT_LEVEL1[unit1]
    fmtf = TIME_FORMAT_LEVEL1_FIRST[unit1]

    tmin_tp = ImPlotTime.from_double(t_min)
    tmax_tp = ImPlotTime.from_double(t_max)

    ticks: List[Tick] = []
    last_major_label: Optional[str] = None

    if unit0 != TIME_YR:
        # pixels per major division (unit1)
        pix_per_major_div = float(pixels) / (span / TIME_UNIT_SPANS[unit1])

        # estimate label widths
        fmt0_width = estimate_label_width_px(fmt0, char_px=char_px, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
        fmt1_width = estimate_label_width_px(fmt1, char_px=char_px, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
        fmtf_width = estimate_label_width_px(fmtf, char_px=char_px, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)

        minor_per_major = int(max_density * pix_per_major_div / fmt0_width)
        step = get_time_step(minor_per_major, unit0)

        # start at first major boundary
        t1 = floor_time(ImPlotTime.from_double(t_min), unit1, use_local_time=use_local_time)

        while t1 < tmax_tp:
            t2 = add_time(t1, unit1, 1, use_local_time=use_local_time)

            # Emit ticks at major boundary if inside visible range
            if t1 >= tmin_tp and t1 <= tmax_tp:
                # Level 0 tick (always labeled)
                label0 = format_datetime(t1, fmt0, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
                ticks.append(Tick(pos=t1.to_double(), level=0, major=True, show_label=True, label=label0))

                # Level 1 tick
                spec = fmtf if last_major_label is None else fmt1
                label1 = format_datetime(t1, spec, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
                show1 = True
                if last_major_label is not None and time_label_same_suffix(last_major_label, label1):
                    show1 = False
                ticks.append(Tick(pos=t1.to_double(), level=1, major=True, show_label=show1, label=(label1 if show1 else None)))
                last_major_label = label1

            # Emit minor ticks in (t1, t2)
            if minor_per_major > 1 and (tmin_tp <= t2 and t1 <= tmax_tp) and step > 0:
                t12 = add_time(t1, unit0, step, use_local_time=use_local_time)
                while t12 < t2:
                    # remaining pixels to the next major boundary
                    px_to_t2 = float((t2 - t12).to_double() / span) * float(pixels)
                    if t12 >= tmin_tp and t12 <= tmax_tp:
                        show0 = px_to_t2 >= fmt0_width
                        label0 = None
                        if show0:
                            label0 = format_datetime(t12, fmt0, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
                        ticks.append(Tick(pos=t12.to_double(), level=0, major=False, show_label=show0, label=label0))

                        # If we have not placed any major label yet, we may place the first
                        # major label at a minor tick if there's room.
                        if last_major_label is None and px_to_t2 >= fmt0_width and px_to_t2 >= (fmt1_width + fmtf_width) / 2.0:
                            label1 = format_datetime(t12, fmtf, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
                            ticks.append(Tick(pos=t12.to_double(), level=1, major=True, show_label=True, label=label1))
                            last_major_label = label1

                    t12 = add_time(t12, unit0, step, use_local_time=use_local_time)

            t1 = t2

    else:
        # Year-scale special case: choose a "nice" year interval.
        fmty = TIME_FORMAT_LEVEL0[TIME_YR]
        label_width = estimate_label_width_px(fmty, char_px=char_px, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
        max_labels = int(max_density * float(pixels) / label_width)
        max_labels = max(2, max_labels)  # avoid division by zero

        year_min = get_year(tmin_tp, use_local_time=use_local_time)
        year_max = get_year(ceil_time(tmax_tp, TIME_YR, use_local_time=use_local_time), use_local_time=use_local_time)

        nice_range = nice_num((year_max - year_min) * 0.99, round_=False)
        interval = nice_num(nice_range / (max_labels - 1), round_=True)
        if interval <= 0:
            interval = 1

        graphmin = int(math.floor(year_min / interval) * interval)
        graphmax = int(math.ceil(year_max / interval) * interval)
        step = int(interval) if int(interval) > 0 else 1

        y = graphmin
        while y < graphmax:
            t = make_time(y, use_local_time=use_local_time)
            if t >= tmin_tp and t <= tmax_tp:
                label = format_datetime(t, fmty, use_local_time=use_local_time, use_24_hour=use_24_hour, use_iso8601=use_iso8601)
                ticks.append(Tick(pos=t.to_double(), level=0, major=True, show_label=True, label=label))
            y += step

    return ticks


# -----------------------------
# Quick demo (optional)
# -----------------------------

if __name__ == "__main__":
    # Example: one hour window, 800px wide.
    now = time.time()
    ticks = locator_time(now, now + 3600, 800, use_local_time=True)
    # Print a compact view
    for tk in ticks[:25]:
        if tk.show_label:
            print(f"L{tk.level} {'M' if tk.major else 'm'} {tk.pos:.3f}  {tk.label}")
        else:
            print(f"L{tk.level} {'M' if tk.major else 'm'} {tk.pos:.3f}")
