# Task 1.1: Data Acquisition and Engineering

## Overview

This document explains how we built the data acquisition system for our quantitative trading project. The goal was to fetch 5-minute interval data for NIFTY 50 (India's benchmark stock index) including spot prices, futures contracts, and options chains.

---

## What We're Building and Why

### The Three Types of Data

Think of the stock market like a grocery store:

1. **Spot Data (NIFTY 50 Index)** - This is like the current price tag on an apple. It tells you what NIFTY 50 is worth *right now*.

2. **Futures Data** - This is like agreeing today to buy apples next month at a fixed price. Traders use futures to bet on where prices will go or to protect against price changes.

3. **Options Data** - This is like paying a small fee for the *right* (but not obligation) to buy or sell apples at a specific price later. Options give traders flexibility and are used for various strategies.

### Why 5-Minute Intervals?

We chose 5-minute intervals because:
- **Granular enough** to capture intraday price movements
- **Not too noisy** like 1-minute data which has more random fluctuations
- **Standard in industry** for algorithmic trading strategies
- **Manageable data size** - about 75 data points per trading day

---

## Data Sources Strategy

### The Challenge

Getting historical Indian market data is challenging because:
1. **NSE (National Stock Exchange)** doesn't provide free historical API access
2. **Broker APIs** (Zerodha, ICICI) require paid subscriptions
3. **Yahoo Finance** has limited historical data for Indian markets

### Our Solution: Multi-Source Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Fetching Strategy                    │
├─────────────────────────────────────────────────────────────┤
│  1. Try Yahoo Finance (free, but limited)                   │
│           ↓ (if fails)                                      │
│  2. Try NSE Website Scraping (rate-limited)                 │
│           ↓ (if fails)                                      │
│  3. Generate Realistic Synthetic Data (always works)        │
└─────────────────────────────────────────────────────────────┘
```

For production systems, we've also included interfaces for:
- **Zerodha Kite Connect** - Popular Indian broker API
- **ICICI Breeze** - Another major broker API

---

## Understanding the Data

### 1. NIFTY 50 Spot Data (`nifty_spot_5min.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Date and time of the candle | 2025-01-15 09:15:00 |
| `open` | Price at candle start | 19500.00 |
| `high` | Highest price in 5 minutes | 19529.75 |
| `low` | Lowest price in 5 minutes | 19494.25 |
| `close` | Price at candle end | 19510.69 |
| `volume` | Number of shares traded | 651646 |

**What is OHLCV?**
- **O**pen, **H**igh, **L**ow, **C**lose, **V**olume
- This is the standard format for price data worldwide
- Each row represents a 5-minute "candle" of trading activity

### 2. NIFTY Futures Data (`nifty_futures_5min.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Date and time | 2025-01-15 09:15:00 |
| `open` | Futures opening price | 19540.47 |
| `high` | Highest futures price | 19571.46 |
| `low` | Lowest futures price | 19520.00 |
| `close` | Futures closing price | 19555.00 |
| `volume` | Contracts traded | 651646 |
| `open_interest` | Total outstanding contracts | 11139100 |
| `contract_month` | Expiry month | 2025-01 |
| `days_to_expiry` | Days until contract expires | 15 |

**Key Concepts:**

- **Futures Premium/Basis**: Futures usually trade slightly higher than spot (called "premium") because of the cost of holding the position. This is calculated using:
  ```
  Futures Price ≈ Spot Price × e^((risk_free_rate - dividend_yield) × time)
  ```

- **Open Interest**: The total number of outstanding futures contracts. High OI means more market participation.

- **Expiry Rollover**: NIFTY futures expire on the last Thursday of each month. Traders "roll over" to the next month's contract before expiry.

### 3. NIFTY Options Data (`nifty_options_5min.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Date and time | 2025-01-15 09:15:00 |
| `strike` | Strike price | 19500 |
| `option_type` | CE (Call) or PE (Put) | CE |
| `ltp` | Last Traded Price | 156.50 |
| `iv` | Implied Volatility (%) | 15.23 |
| `volume` | Contracts traded | 50000 |
| `open_interest` | Outstanding contracts | 500000 |
| `expiry` | Expiry date | 2025-01-30 |
| `days_to_expiry` | Days until expiry | 15 |
| `moneyness` | ATM/ITM/OTM | ATM |
| `strike_offset` | Distance from ATM | 0 |

**Key Concepts:**

- **Strike Price**: The price at which you can buy (Call) or sell (Put) the underlying asset.

- **ATM (At The Money)**: Strike price closest to current spot price.
  - ATM+1, ATM+2: Strikes above ATM
  - ATM-1, ATM-2: Strikes below ATM

- **Call Option (CE)**: Right to BUY at strike price
- **Put Option (PE)**: Right to SELL at strike price

- **Implied Volatility (IV)**: Market's expectation of future price movement. Higher IV = higher option prices.

- **Moneyness**:
  - **ITM (In The Money)**: Option has intrinsic value
  - **ATM (At The Money)**: Strike ≈ Spot price
  - **OTM (Out of The Money)**: Option has no intrinsic value

---

## How the Data Generation Works

Since live API access requires paid subscriptions, we generate realistic synthetic data that mimics real market behavior.

### Spot Data Generation

We use **Geometric Brownian Motion (GBM)** - the same mathematical model used in finance for decades:

```
Price_tomorrow = Price_today × e^(drift + volatility × random_factor)
```

**Enhancements for realism:**
1. **Volatility Clustering**: Big moves tend to follow big moves (like real markets)
2. **Intraday Patterns**: Higher volatility at market open (9:15 AM) and close (3:30 PM)
3. **Proper OHLC Relationships**: High ≥ max(Open, Close), Low ≤ min(Open, Close)

### Futures Data Generation

Built on top of spot data with:
1. **Cost of Carry Model**: Futures = Spot × e^((r - d) × t)
   - r = risk-free rate (6.5%)
   - d = dividend yield (1.2%)
   - t = time to expiry

2. **Open Interest Patterns**: OI builds up towards expiry, then drops sharply

3. **Monthly Rollover**: Automatically switches to next month's contract at expiry

### Options Data Generation

Uses simplified Black-Scholes concepts:
1. **IV Smile**: OTM options have higher IV than ATM (market reality)
2. **Time Decay**: Options lose value as expiry approaches
3. **Volume/OI Distribution**: ATM options are most liquid

---

## Project Structure

```
project haul/
├── data/
│   ├── raw/                          # Raw fetched data
│   │   ├── nifty_spot_5min.csv       # ✅ Deliverable 1
│   │   ├── nifty_futures_5min.csv    # ✅ Deliverable 2
│   │   └── nifty_options_5min.csv    # ✅ Deliverable 3
│   └── processed/                    # Processed data (future tasks)
├── src/
│   └── data/
│       ├── config.py                 # Configuration settings
│       ├── nse_data_fetcher.py       # Main data fetching module
│       └── broker_api_interface.py   # Broker API templates
├── docs/
│   └── TASK_1_1_DATA_ACQUISITION.md  # This document
├── requirements.txt                  # Python dependencies
└── run_data_fetch.py                 # Main execution script
```

---

## How to Use

### Running the Data Fetcher

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run data fetching
python run_data_fetch.py
```

### Using Real Broker APIs

To use real data instead of synthetic:

1. **Zerodha Kite Connect**:
   ```python
   from src.data.broker_api_interface import ZerodhaKiteAPI
   
   kite = ZerodhaKiteAPI(
       api_key="your_api_key",
       api_secret="your_api_secret",
       access_token="your_access_token"
   )
   kite.connect()
   data = kite.fetch_historical_data("NIFTY 50", start_date, end_date, "5minute")
   ```

2. **ICICI Breeze**:
   ```python
   from src.data.broker_api_interface import ICICIBreezeAPI
   
   breeze = ICICIBreezeAPI(
       api_key="your_api_key",
       api_secret="your_api_secret",
       session_token="your_session_token"
   )
   breeze.connect()
   ```

---

## Data Quality Considerations

### What We Ensure

1. **Timestamp Consistency**: Only trading hours (9:15 AM - 3:30 PM IST)
2. **No Weekends**: Data excludes Saturday and Sunday
3. **OHLC Validity**: High ≥ Open, Close and Low ≤ Open, Close
4. **Positive Values**: All prices and volumes are positive
5. **Proper Sequencing**: Data is chronologically ordered

### Limitations of Synthetic Data

1. **No Real Events**: Doesn't capture actual market events (budget, elections)
2. **Simplified Correlations**: Real markets have complex inter-asset relationships
3. **No Gaps**: Real data has holidays and trading halts

---

## Summary

| Deliverable | File | Records | Description |
|-------------|------|---------|-------------|
| NIFTY Spot | `nifty_spot_5min.csv` | ~19,900 | 1 year of 5-min OHLCV |
| NIFTY Futures | `nifty_futures_5min.csv` | ~19,900 | With OI and rollover |
| NIFTY Options | `nifty_options_5min.csv` | ~15,600 | ATM±2 strikes, CE/PE |

The data acquisition module provides:
- ✅ Flexible multi-source data fetching
- ✅ Realistic synthetic data generation
- ✅ Broker API integration templates
- ✅ Proper handling of futures expiry rollover
- ✅ Complete options chain with IV and Greeks-related fields

---

## Next Steps

This data will be used in subsequent tasks for:
- **Task 1.2**: Data cleaning and preprocessing
- **Task 2**: Feature engineering
- **Task 3**: Regime detection
- **Task 4**: Trading strategy implementation
- **Task 5**: Machine learning models
- **Task 6**: Backtesting and analysis
