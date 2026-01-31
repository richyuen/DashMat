"""Constants used across DashMat calculations."""

# Window mapping for calendar day-based rolling calculations (daily data)
WINDOW_MAP_DAYS = {
    "3m": "91D",
    "6m": "183D",
    "1y": "365D",
    "3y": "1096D",
    "5y": "1826D",
    "10y": "3652D",
}

# Window mapping to integer days for minimum period checks
WINDOW_DAYS_MAP = {
    "3m": 91,
    "6m": 183,
    "1y": 365,
    "3y": 1096,
    "5y": 1826,
    "10y": 3652,
}

# Window mapping to fractional years for annualization
WINDOW_YEARS_MAP = {
    "3m": 0.25,
    "6m": 0.5,
    "1y": 1.0,
    "3y": 3.0,
    "5y": 5.0,
    "10y": 10.0,
}
