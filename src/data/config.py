"""
Configuration settings for data acquisition module.
"""
from datetime import datetime, timedelta
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Time settings
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)  # 1 year of data
INTERVAL = "5min"

# NIFTY settings
NIFTY_SPOT_SYMBOL = "^NSEI"  # Yahoo Finance symbol for NIFTY 50
NIFTY_FUTURES_SYMBOL = "NIFTY"
NIFTY_LOT_SIZE = 25

# Options chain settings
ATM_RANGE = 2  # ATM ± 2 strikes
STRIKE_INTERVAL = 50  # NIFTY strike interval

# NSE settings
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Output file names
SPOT_OUTPUT_FILE = "nifty_spot_5min.csv"
FUTURES_OUTPUT_FILE = "nifty_futures_5min.csv"
OPTIONS_OUTPUT_FILE = "nifty_options_5min.csv"

# Trading hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Expiry settings (NIFTY monthly expiry is last Thursday of month)
EXPIRY_DAY = 3  # Thursday (0=Monday, 3=Thursday)
