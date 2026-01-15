"""
Options Greeks Calculator Module - Task 2.2

This module calculates Options Greeks using the Black-Scholes model:
- Delta: Rate of change of option price with respect to underlying price
- Gamma: Rate of change of Delta with respect to underlying price
- Theta: Rate of change of option price with respect to time (time decay)
- Vega: Rate of change of option price with respect to volatility
- Rho: Rate of change of option price with respect to interest rate

Uses py_vollib library for Black-Scholes calculations.
Risk-free rate: 6.5% (as specified)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try py_vollib first, fall back to mibian, then manual implementation
try:
    from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
    from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
    from py_vollib.black_scholes.greeks.analytical import theta as bs_theta
    from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
    from py_vollib.black_scholes.greeks.analytical import rho as bs_rho
    from py_vollib.black_scholes import black_scholes as bs_price
    GREEKS_LIB = 'py_vollib'
except ImportError:
    try:
        import mibian
        GREEKS_LIB = 'mibian'
    except ImportError:
        GREEKS_LIB = 'manual'

from scipy.stats import norm
from scipy import optimize


class BlackScholesGreeks:
    """
    Black-Scholes Options Greeks Calculator.
    
    The Black-Scholes model assumes:
    - European-style options (can only exercise at expiry)
    - No dividends during option life
    - Constant volatility and interest rate
    - Log-normal distribution of returns
    
    Parameters:
        risk_free_rate: Annual risk-free interest rate (default: 6.5%)
    """
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Greeks calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 6.5% = 0.065)
        """
        self.risk_free_rate = risk_free_rate
        self.library = GREEKS_LIB
        print(f"Greeks calculator initialized using: {self.library}")
        print(f"Risk-free rate: {self.risk_free_rate * 100}%")
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def calculate_delta(self, S: float, K: float, T: float, sigma: float, 
                        option_type: str = 'c') -> float:
        """
        Calculate Delta - sensitivity to underlying price.
        
        Delta measures how much the option price changes for a $1 change
        in the underlying asset price.
        
        Call Delta: 0 to 1 (ATM ≈ 0.5)
        Put Delta: -1 to 0 (ATM ≈ -0.5)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility (as decimal, e.g., 0.15 for 15%)
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Delta value
        """
        if T <= 0:
            # At expiry
            if option_type.lower() == 'c':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        if self.library == 'py_vollib':
            try:
                return bs_delta(option_type.lower(), S, K, T, self.risk_free_rate, sigma)
            except:
                pass
        
        # Manual calculation
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        if option_type.lower() == 'c':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def calculate_gamma(self, S: float, K: float, T: float, sigma: float,
                        option_type: str = 'c') -> float:
        """
        Calculate Gamma - rate of change of Delta.
        
        Gamma measures how much Delta changes for a $1 change in underlying.
        Gamma is the same for calls and puts at the same strike.
        Highest for ATM options, decreases for ITM/OTM.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        if self.library == 'py_vollib':
            try:
                return bs_gamma(option_type.lower(), S, K, T, self.risk_free_rate, sigma)
            except:
                pass
        
        # Manual calculation
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def calculate_theta(self, S: float, K: float, T: float, sigma: float,
                        option_type: str = 'c') -> float:
        """
        Calculate Theta - time decay.
        
        Theta measures how much the option price decreases per day.
        Usually negative (options lose value over time).
        Expressed as daily decay (divided by 365).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Theta value (daily)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        if self.library == 'py_vollib':
            try:
                # py_vollib returns annual theta, convert to daily
                return bs_theta(option_type.lower(), S, K, T, self.risk_free_rate, sigma) / 365
            except:
                pass
        
        # Manual calculation
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        d2 = self._d2(S, K, T, self.risk_free_rate, sigma)
        
        first_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type.lower() == 'c':
            second_term = -self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
            theta_annual = first_term + second_term
        else:
            second_term = self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
            theta_annual = first_term + second_term
        
        return theta_annual / 365  # Convert to daily
    
    def calculate_vega(self, S: float, K: float, T: float, sigma: float,
                       option_type: str = 'c') -> float:
        """
        Calculate Vega - sensitivity to volatility.
        
        Vega measures how much the option price changes for a 1% change
        in implied volatility. Same for calls and puts.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Vega value (per 1% IV change)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        if self.library == 'py_vollib':
            try:
                # py_vollib returns vega per 1 unit change, we want per 1%
                return bs_vega(option_type.lower(), S, K, T, self.risk_free_rate, sigma) / 100
            except:
                pass
        
        # Manual calculation
        d1 = self._d1(S, K, T, self.risk_free_rate, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100  # Per 1% change in IV
    
    def calculate_rho(self, S: float, K: float, T: float, sigma: float,
                      option_type: str = 'c') -> float:
        """
        Calculate Rho - sensitivity to interest rate.
        
        Rho measures how much the option price changes for a 1% change
        in the risk-free interest rate.
        
        Call Rho: Positive (higher rates increase call value)
        Put Rho: Negative (higher rates decrease put value)
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Rho value (per 1% rate change)
        """
        if T <= 0:
            return 0.0
        
        if self.library == 'py_vollib':
            try:
                # py_vollib returns rho per 1 unit change, we want per 1%
                return bs_rho(option_type.lower(), S, K, T, self.risk_free_rate, sigma) / 100
            except:
                pass
        
        # Manual calculation
        d2 = self._d2(S, K, T, self.risk_free_rate, sigma)
        
        if option_type.lower() == 'c':
            rho = K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)
        
        return rho / 100  # Per 1% change in rate
    
    def calculate_all_greeks(self, S: float, K: float, T: float, sigma: float,
                             option_type: str = 'c') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Implied volatility (as decimal)
            option_type: 'c' for call, 'p' for put
            
        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': self.calculate_delta(S, K, T, sigma, option_type),
            'gamma': self.calculate_gamma(S, K, T, sigma, option_type),
            'theta': self.calculate_theta(S, K, T, sigma, option_type),
            'vega': self.calculate_vega(S, K, T, sigma, option_type),
            'rho': self.calculate_rho(S, K, T, sigma, option_type)
        }
    
    def add_greeks_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Greeks for ATM Call and Put options to the DataFrame.
        
        Requires columns:
        - spot_close: Underlying spot price
        - opt_ce_atm_strike: ATM Call strike price
        - opt_pe_atm_strike: ATM Put strike price
        - opt_ce_atm_iv: ATM Call implied volatility (%)
        - opt_pe_atm_iv: ATM Put implied volatility (%)
        - opt_dte: Days to expiry
        
        Adds columns:
        - greeks_ce_atm_delta, greeks_ce_atm_gamma, etc.
        - greeks_pe_atm_delta, greeks_pe_atm_gamma, etc.
        """
        df = df.copy()
        
        # Initialize Greek columns
        greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for opt_type in ['ce', 'pe']:
            for greek in greek_names:
                df[f'greeks_{opt_type}_atm_{greek}'] = np.nan
        
        # Calculate Greeks for each row with options data
        rows_processed = 0
        
        for idx, row in df.iterrows():
            # Skip rows without options data
            if pd.isna(row.get('opt_ce_atm_iv')) or pd.isna(row.get('opt_dte')):
                continue
            
            S = row['spot_close']
            dte = row['opt_dte']
            T = max(dte, 1) / 365  # Time to expiry in years (minimum 1 day)
            
            # Calculate Greeks for ATM Call
            if not pd.isna(row.get('opt_ce_atm_strike')) and not pd.isna(row.get('opt_ce_atm_iv')):
                K_ce = row['opt_ce_atm_strike']
                sigma_ce = row['opt_ce_atm_iv'] / 100  # Convert from % to decimal
                
                greeks_ce = self.calculate_all_greeks(S, K_ce, T, sigma_ce, 'c')
                for greek, value in greeks_ce.items():
                    df.at[idx, f'greeks_ce_atm_{greek}'] = round(value, 6)
            
            # Calculate Greeks for ATM Put
            if not pd.isna(row.get('opt_pe_atm_strike')) and not pd.isna(row.get('opt_pe_atm_iv')):
                K_pe = row['opt_pe_atm_strike']
                sigma_pe = row['opt_pe_atm_iv'] / 100  # Convert from % to decimal
                
                greeks_pe = self.calculate_all_greeks(S, K_pe, T, sigma_pe, 'p')
                for greek, value in greeks_pe.items():
                    df.at[idx, f'greeks_pe_atm_{greek}'] = round(value, 6)
            
            rows_processed += 1
        
        print(f"  Greeks calculated for {rows_processed} rows with options data")
        
        return df
    
    def get_greeks_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for Greeks."""
        summary = {
            'risk_free_rate': f"{self.risk_free_rate * 100}%",
            'library_used': self.library,
            'rows_with_greeks': df['greeks_ce_atm_delta'].notna().sum(),
        }
        
        # ATM Call Greeks summary
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            col = f'greeks_ce_atm_{greek}'
            if col in df.columns:
                summary[f'ce_atm_{greek}_mean'] = round(df[col].mean(), 6)
                summary[f'ce_atm_{greek}_std'] = round(df[col].std(), 6)
        
        # ATM Put Greeks summary
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            col = f'greeks_pe_atm_{greek}'
            if col in df.columns:
                summary[f'pe_atm_{greek}_mean'] = round(df[col].mean(), 6)
                summary[f'pe_atm_{greek}_std'] = round(df[col].std(), 6)
        
        return summary


def add_greeks_to_dataset(input_path: Path, output_path: Path = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Add Options Greeks to a dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output (optional)
        
    Returns:
        Tuple of (DataFrame with Greeks, summary dict)
    """
    print("=" * 60)
    print("Options Greeks Calculator - Task 2.2")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Records: {len(df)}")
    
    # Initialize Greeks calculator
    print("\nInitializing Black-Scholes Greeks calculator...")
    greeks_calc = BlackScholesGreeks(risk_free_rate=0.065)
    
    # Add Greeks
    print("\nCalculating Greeks for ATM options...")
    df = greeks_calc.add_greeks_to_dataframe(df)
    
    # Get summary
    summary = greeks_calc.get_greeks_summary(df)
    
    print("\nGreeks Summary:")
    print(f"  Rows with Greeks: {summary['rows_with_greeks']}")
    print(f"\n  ATM Call Greeks (mean):")
    print(f"    Delta: {summary.get('ce_atm_delta_mean', 'N/A')}")
    print(f"    Gamma: {summary.get('ce_atm_gamma_mean', 'N/A')}")
    print(f"    Theta: {summary.get('ce_atm_theta_mean', 'N/A')} (daily)")
    print(f"    Vega:  {summary.get('ce_atm_vega_mean', 'N/A')} (per 1% IV)")
    print(f"    Rho:   {summary.get('ce_atm_rho_mean', 'N/A')} (per 1% rate)")
    print(f"\n  ATM Put Greeks (mean):")
    print(f"    Delta: {summary.get('pe_atm_delta_mean', 'N/A')}")
    print(f"    Gamma: {summary.get('pe_atm_gamma_mean', 'N/A')}")
    print(f"    Theta: {summary.get('pe_atm_theta_mean', 'N/A')} (daily)")
    print(f"    Vega:  {summary.get('pe_atm_vega_mean', 'N/A')} (per 1% IV)")
    print(f"    Rho:   {summary.get('pe_atm_rho_mean', 'N/A')} (per 1% rate)")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("OPTIONS GREEKS COMPLETE")
    print("=" * 60)
    
    return df, summary


def main():
    """Main function to add Greeks to dataset."""
    input_path = Path("data/processed/nifty_features_5min.csv")
    output_path = Path("data/processed/nifty_features_5min.csv")  # Overwrite with Greeks
    
    df, summary = add_greeks_to_dataset(input_path, output_path)
    
    # Print sample
    print("\nSample data with Greeks:")
    greek_cols = ['timestamp', 'spot_close', 'opt_ce_atm_iv', 'opt_dte',
                  'greeks_ce_atm_delta', 'greeks_ce_atm_gamma', 'greeks_ce_atm_theta',
                  'greeks_pe_atm_delta', 'greeks_pe_atm_gamma', 'greeks_pe_atm_theta']
    available_cols = [c for c in greek_cols if c in df.columns]
    print(df[available_cols].dropna().head(10))
    
    return df, summary


if __name__ == "__main__":
    main()
