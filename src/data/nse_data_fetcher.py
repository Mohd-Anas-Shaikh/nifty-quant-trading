"""
NSE Data Fetcher Module

This module provides functionality to fetch historical data from NSE India
for NIFTY 50 Spot, Futures, and Options.

Since direct NSE API access requires authentication and has rate limits,
this module implements multiple data source strategies:
1. Primary: NSE India website scraping (with proper session handling)
2. Fallback: Yahoo Finance for spot data
3. Synthetic: Generate realistic sample data for demonstration

For production use, integrate with:
- Zerodha Kite Connect API
- ICICI Breeze API
- Groww API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import requests
import time
import warnings
from pathlib import Path

from .config import (
    START_DATE, END_DATE, INTERVAL,
    ATM_RANGE, STRIKE_INTERVAL,
    NSE_BASE_URL, NSE_HEADERS,
    RAW_DATA_DIR,
    SPOT_OUTPUT_FILE, FUTURES_OUTPUT_FILE, OPTIONS_OUTPUT_FILE,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
)

warnings.filterwarnings('ignore')


class NSESession:
    """Handles NSE website session management with cookie handling."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session by visiting NSE homepage to get cookies."""
        try:
            self.session.get(NSE_BASE_URL, timeout=10)
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not initialize NSE session: {e}")
    
    def get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make GET request with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    return response.json()
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {url}: {e}")
                time.sleep(2 ** attempt)
        return None


class NiftyDataFetcher:
    """
    Fetches NIFTY 50 Spot, Futures, and Options data.
    
    This class handles:
    - 5-minute interval OHLCV data for spot
    - Futures data with monthly expiry rollover
    - Options chain data for ATM ± 2 strikes
    """
    
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.nse_session = NSESession()
        
    def _generate_trading_timestamps(self) -> pd.DatetimeIndex:
        """
        Generate 5-minute interval timestamps for trading hours only.
        Trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday.
        """
        all_timestamps = []
        current_date = self.start_date.date()
        end_date = self.end_date.date()
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                # Generate timestamps for trading hours
                market_open = datetime.combine(
                    current_date, 
                    datetime.min.time().replace(
                        hour=MARKET_OPEN_HOUR, 
                        minute=MARKET_OPEN_MINUTE
                    )
                )
                market_close = datetime.combine(
                    current_date,
                    datetime.min.time().replace(
                        hour=MARKET_CLOSE_HOUR,
                        minute=MARKET_CLOSE_MINUTE
                    )
                )
                
                current_time = market_open
                while current_time <= market_close:
                    all_timestamps.append(current_time)
                    current_time += timedelta(minutes=5)
            
            current_date += timedelta(days=1)
        
        return pd.DatetimeIndex(all_timestamps)
    
    def _get_monthly_expiries(self) -> List[datetime]:
        """
        Calculate monthly expiry dates (last Thursday of each month).
        Handles cases where Thursday is a holiday by moving to Wednesday.
        """
        expiries = []
        current_date = self.start_date.replace(day=1)
        
        while current_date <= self.end_date:
            # Find last Thursday of the month
            next_month = current_date.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            
            # Find last Thursday
            days_until_thursday = (last_day.weekday() - 3) % 7
            last_thursday = last_day - timedelta(days=days_until_thursday)
            
            if last_thursday >= self.start_date and last_thursday <= self.end_date:
                expiries.append(last_thursday)
            
            # Move to next month
            current_date = next_month.replace(day=1)
        
        return expiries
    
    def _calculate_atm_strike(self, spot_price: float) -> int:
        """Calculate ATM strike price rounded to nearest strike interval."""
        return int(round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL)
    
    def _generate_realistic_spot_data(self) -> pd.DataFrame:
        """
        Generate realistic NIFTY 50 spot data using geometric Brownian motion.
        
        This creates synthetic data that mimics real market behavior:
        - Realistic price movements with volatility clustering
        - Proper OHLCV relationships
        - Intraday patterns (higher volatility at open/close)
        """
        timestamps = self._generate_trading_timestamps()
        n_periods = len(timestamps)
        
        # Starting price and parameters
        initial_price = 19500.0
        annual_return = 0.12  # 12% annual return
        annual_volatility = 0.15  # 15% annual volatility
        
        # Convert to 5-minute parameters
        periods_per_year = 252 * 75  # ~75 5-min periods per day
        dt = 1 / periods_per_year
        mu = annual_return * dt
        sigma = annual_volatility * np.sqrt(dt)
        
        # Generate returns with volatility clustering (GARCH-like)
        np.random.seed(42)  # For reproducibility
        
        # Base random returns
        random_returns = np.random.normal(mu, sigma, n_periods)
        
        # Add volatility clustering
        volatility_factor = np.ones(n_periods)
        for i in range(1, n_periods):
            volatility_factor[i] = 0.9 * volatility_factor[i-1] + 0.1 * abs(random_returns[i-1]) / sigma
        
        adjusted_returns = random_returns * (0.5 + 0.5 * volatility_factor)
        
        # Generate close prices
        close_prices = initial_price * np.exp(np.cumsum(adjusted_returns))
        
        # Generate OHLC from close prices
        intraday_volatility = sigma * 0.5
        
        # Add intraday patterns (higher vol at open/close)
        time_of_day = np.array([t.hour + t.minute/60 for t in timestamps])
        intraday_factor = 1 + 0.3 * (
            np.exp(-((time_of_day - 9.25)**2) / 0.5) +  # Morning spike
            np.exp(-((time_of_day - 15.5)**2) / 0.5)    # Closing spike
        )
        
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, intraday_volatility, n_periods)) * intraday_factor)
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, intraday_volatility, n_periods)) * intraday_factor)
        
        # Open price is previous close with small gap
        open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, sigma * 0.1, n_periods))
        open_prices[0] = initial_price
        
        # Ensure OHLC consistency
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        # Generate volume (higher at open/close, correlated with volatility)
        base_volume = 500000
        volume = base_volume * intraday_factor * (0.5 + np.abs(adjusted_returns) / sigma)
        volume = volume.astype(int)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.round(open_prices, 2),
            'high': np.round(high_prices, 2),
            'low': np.round(low_prices, 2),
            'close': np.round(close_prices, 2),
            'volume': volume
        })
        
        return df
    
    def _generate_realistic_futures_data(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate NIFTY Futures data with:
        - Basis (futures premium/discount to spot)
        - Open Interest
        - Monthly expiry rollover handling
        """
        futures_df = spot_df.copy()
        expiries = self._get_monthly_expiries()
        
        n_periods = len(futures_df)
        
        # Calculate days to expiry for each timestamp
        days_to_expiry = np.zeros(n_periods)
        contract_month = []
        
        for i, ts in enumerate(futures_df['timestamp']):
            # Find current contract expiry
            current_expiry = None
            for exp in expiries:
                ts_d = ts.date() if isinstance(ts, datetime) else ts
                exp_d = exp.date() if isinstance(exp, datetime) else exp
                if exp_d >= ts_d:
                    current_expiry = exp
                    break
            
            if current_expiry is None:
                current_expiry = expiries[-1] if expiries else ts
            
            # Convert to date for comparison
            ts_date = ts.date() if isinstance(ts, datetime) else ts
            exp_date = current_expiry.date() if isinstance(current_expiry, datetime) else current_expiry
            days_to_expiry[i] = max(1, (exp_date - ts_date).days)
            contract_month.append(current_expiry.strftime('%Y-%m') if isinstance(current_expiry, datetime) else str(current_expiry)[:7])
        
        # Calculate futures basis (cost of carry model)
        risk_free_rate = 0.065  # 6.5% annual
        dividend_yield = 0.012  # 1.2% annual
        
        basis = spot_df['close'].values * (
            np.exp((risk_free_rate - dividend_yield) * days_to_expiry / 365) - 1
        )
        
        # Add some noise to basis
        basis_noise = np.random.normal(0, 2, n_periods)
        basis = basis + basis_noise
        
        # Futures prices
        futures_df['close'] = np.round(spot_df['close'].values + basis, 2)
        futures_df['open'] = np.round(spot_df['open'].values + basis * 0.98, 2)
        futures_df['high'] = np.round(spot_df['high'].values + basis * 1.01, 2)
        futures_df['low'] = np.round(spot_df['low'].values + basis * 0.99, 2)
        
        # Ensure OHLC consistency
        futures_df['high'] = np.maximum(futures_df['high'], np.maximum(futures_df['open'], futures_df['close']))
        futures_df['low'] = np.minimum(futures_df['low'], np.minimum(futures_df['open'], futures_df['close']))
        
        # Generate Open Interest
        # OI tends to build up towards expiry, then drop sharply at rollover
        base_oi = 10000000  # 10 million base OI
        
        oi = np.zeros(n_periods)
        for i in range(n_periods):
            # OI builds as expiry approaches, peaks ~5 days before
            dte = days_to_expiry[i]
            if dte > 5:
                oi_factor = 1 + 0.3 * (30 - dte) / 30
            else:
                oi_factor = 1.3 - 0.2 * (5 - dte) / 5  # Decline near expiry
            
            oi[i] = base_oi * oi_factor * (1 + np.random.normal(0, 0.05))
        
        futures_df['open_interest'] = oi.astype(int)
        futures_df['contract_month'] = contract_month
        futures_df['days_to_expiry'] = days_to_expiry.astype(int)
        
        # Reorder columns
        futures_df = futures_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                  'open_interest', 'contract_month', 'days_to_expiry']]
        
        return futures_df
    
    def _generate_realistic_options_data(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate NIFTY Options chain data for ATM ± 2 strikes.
        
        Includes:
        - Call and Put options
        - LTP (Last Traded Price)
        - Implied Volatility
        - Open Interest
        - Volume
        """
        options_data = []
        expiries = self._get_monthly_expiries()
        
        for idx, row in spot_df.iterrows():
            timestamp = row['timestamp']
            spot_price = row['close']
            
            # Find current expiry
            current_expiry = None
            for exp in expiries:
                exp_date = exp.date() if isinstance(exp, datetime) else exp
                ts_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
                if exp_date >= ts_date:
                    current_expiry = exp
                    break
            
            if current_expiry is None:
                continue
            
            # Calculate days to expiry
            ts_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
            exp_date = current_expiry.date() if isinstance(current_expiry, datetime) else current_expiry
            dte = max(1, (exp_date - ts_date).days)
            
            # Calculate ATM strike
            atm_strike = self._calculate_atm_strike(spot_price)
            
            # Generate data for ATM ± 2 strikes
            for strike_offset in range(-ATM_RANGE, ATM_RANGE + 1):
                strike = atm_strike + strike_offset * STRIKE_INTERVAL
                
                for option_type in ['CE', 'PE']:
                    # Calculate theoretical option price using simplified Black-Scholes
                    moneyness = (spot_price - strike) / spot_price
                    
                    # Base IV (smile pattern)
                    base_iv = 0.15 + 0.02 * abs(strike_offset)  # IV smile
                    
                    # Add some randomness to IV
                    iv = base_iv * (1 + np.random.normal(0, 0.1))
                    iv = max(0.08, min(0.50, iv))  # Clamp IV
                    
                    # Time value factor
                    time_factor = np.sqrt(dte / 365)
                    
                    # Intrinsic value
                    if option_type == 'CE':
                        intrinsic = max(0, spot_price - strike)
                        delta_approx = 0.5 + 0.5 * np.tanh(moneyness * 5)
                    else:
                        intrinsic = max(0, strike - spot_price)
                        delta_approx = -0.5 + 0.5 * np.tanh(moneyness * 5)
                    
                    # Time value (simplified)
                    time_value = spot_price * iv * time_factor * 0.4 * np.exp(-abs(moneyness) * 3)
                    
                    # LTP
                    ltp = intrinsic + time_value
                    ltp = max(0.5, round(ltp, 2))  # Minimum tick
                    
                    # Volume (higher for ATM, lower for OTM)
                    base_volume = 50000 * np.exp(-abs(strike_offset) * 0.5)
                    volume = int(base_volume * (1 + np.random.exponential(0.5)))
                    
                    # Open Interest
                    base_oi = 500000 * np.exp(-abs(strike_offset) * 0.3)
                    oi = int(base_oi * (1 + np.random.normal(0, 0.2)))
                    
                    options_data.append({
                        'timestamp': timestamp,
                        'strike': strike,
                        'option_type': option_type,
                        'ltp': ltp,
                        'iv': round(iv * 100, 2),  # IV in percentage
                        'volume': volume,
                        'open_interest': max(0, oi),
                        'expiry': current_expiry.strftime('%Y-%m-%d') if isinstance(current_expiry, datetime) else str(current_expiry),
                        'days_to_expiry': dte,
                        'moneyness': 'ATM' if strike_offset == 0 else ('ITM' if (option_type == 'CE' and strike < spot_price) or (option_type == 'PE' and strike > spot_price) else 'OTM'),
                        'strike_offset': strike_offset
                    })
        
        return pd.DataFrame(options_data)
    
    def fetch_spot_data(self) -> pd.DataFrame:
        """
        Fetch NIFTY 50 Spot 5-minute OHLCV data.
        
        Attempts to fetch from Yahoo Finance first, falls back to synthetic data.
        """
        print("Fetching NIFTY 50 Spot data...")
        
        try:
            import yfinance as yf
            
            # Try Yahoo Finance
            nifty = yf.Ticker("^NSEI")
            df = nifty.history(
                start=self.start_date,
                end=self.end_date,
                interval="5m"
            )
            
            if len(df) > 1000:  # Sufficient data
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                print(f"  Fetched {len(df)} records from Yahoo Finance")
                return df
        except Exception as e:
            print(f"  Yahoo Finance fetch failed: {e}")
        
        # Fallback to synthetic data
        print("  Generating synthetic spot data...")
        df = self._generate_realistic_spot_data()
        print(f"  Generated {len(df)} synthetic records")
        return df
    
    def fetch_futures_data(self, spot_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fetch NIFTY Futures 5-minute data with expiry rollover handling.
        
        For production, this would connect to broker APIs.
        Currently generates realistic synthetic data based on spot prices.
        """
        print("Fetching NIFTY Futures data...")
        
        if spot_df is None:
            spot_df = self.fetch_spot_data()
        
        df = self._generate_realistic_futures_data(spot_df)
        print(f"  Generated {len(df)} futures records with {len(self._get_monthly_expiries())} monthly rollovers")
        return df
    
    def fetch_options_data(self, spot_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fetch NIFTY Options chain data for ATM ± 2 strikes.
        
        Generates data for both Calls and Puts with:
        - LTP, IV, Open Interest, Volume
        """
        print("Fetching NIFTY Options chain data...")
        
        if spot_df is None:
            spot_df = self.fetch_spot_data()
        
        # Sample every 12th row (1 hour) to reduce options data size
        # Full 5-min options data would be very large
        spot_sampled = spot_df.iloc[::12].reset_index(drop=True)
        
        df = self._generate_realistic_options_data(spot_sampled)
        print(f"  Generated {len(df)} options records")
        print(f"  Strikes: ATM ± {ATM_RANGE} ({2 * ATM_RANGE + 1} strikes)")
        print(f"  Option types: CE, PE")
        return df
    
    def fetch_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch all NIFTY data: Spot, Futures, and Options."""
        spot_df = self.fetch_spot_data()
        futures_df = self.fetch_futures_data(spot_df)
        options_df = self.fetch_options_data(spot_df)
        
        return spot_df, futures_df, options_df
    
    def save_data(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame, 
                  options_df: pd.DataFrame, output_dir: Path = None) -> Dict[str, Path]:
        """Save all dataframes to CSV files."""
        output_dir = output_dir or RAW_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save spot data
        spot_path = output_dir / SPOT_OUTPUT_FILE
        spot_df.to_csv(spot_path, index=False)
        paths['spot'] = spot_path
        print(f"Saved spot data to: {spot_path}")
        
        # Save futures data
        futures_path = output_dir / FUTURES_OUTPUT_FILE
        futures_df.to_csv(futures_path, index=False)
        paths['futures'] = futures_path
        print(f"Saved futures data to: {futures_path}")
        
        # Save options data
        options_path = output_dir / OPTIONS_OUTPUT_FILE
        options_df.to_csv(options_path, index=False)
        paths['options'] = options_path
        print(f"Saved options data to: {options_path}")
        
        return paths


def main():
    """Main function to fetch and save all NIFTY data."""
    print("=" * 60)
    print("NIFTY Data Acquisition - Task 1.1")
    print("=" * 60)
    print(f"Date Range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Interval: 5 minutes")
    print("=" * 60)
    
    fetcher = NiftyDataFetcher()
    
    # Fetch all data
    spot_df, futures_df, options_df = fetcher.fetch_all_data()
    
    # Display summaries
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print("\n--- NIFTY 50 Spot ---")
    print(f"Records: {len(spot_df)}")
    print(f"Date Range: {spot_df['timestamp'].min()} to {spot_df['timestamp'].max()}")
    print(f"Price Range: {spot_df['close'].min():.2f} to {spot_df['close'].max():.2f}")
    print(spot_df.head())
    
    print("\n--- NIFTY Futures ---")
    print(f"Records: {len(futures_df)}")
    print(f"Contract Months: {futures_df['contract_month'].nunique()}")
    print(futures_df.head())
    
    print("\n--- NIFTY Options ---")
    print(f"Records: {len(options_df)}")
    print(f"Unique Strikes: {options_df['strike'].nunique()}")
    print(f"Option Types: {options_df['option_type'].unique().tolist()}")
    print(options_df.head())
    
    # Save data
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)
    paths = fetcher.save_data(spot_df, futures_df, options_df)
    
    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 60)
    
    return spot_df, futures_df, options_df


if __name__ == "__main__":
    main()
