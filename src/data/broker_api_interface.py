"""
Broker API Interface Module

This module provides abstract interfaces and implementations for connecting
to various Indian stock broker APIs for fetching NIFTY data.

Supported Brokers:
1. Zerodha Kite Connect
2. ICICI Breeze
3. Groww (via unofficial API)

Note: These require valid API credentials and active subscriptions.
The implementations below are templates - uncomment and configure as needed.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd


class BrokerAPIInterface(ABC):
    """Abstract base class for broker API implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker API."""
        pass
    
    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        pass
    
    @abstractmethod
    def fetch_option_chain(
        self,
        symbol: str,
        expiry: datetime
    ) -> pd.DataFrame:
        """Fetch options chain data."""
        pass
    
    @abstractmethod
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol."""
        pass


class ZerodhaKiteAPI(BrokerAPIInterface):
    """
    Zerodha Kite Connect API Implementation.
    
    Requirements:
    - pip install kiteconnect
    - Valid API key and secret from Kite Connect
    - Active Kite Connect subscription
    
    Documentation: https://kite.trade/docs/connect/v3/
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self.instruments = None
    
    def connect(self) -> bool:
        """
        Connect to Kite API.
        
        For first-time authentication:
        1. Generate login URL: kite.login_url()
        2. User logs in and gets request_token from redirect URL
        3. Generate access_token: kite.generate_session(request_token, api_secret)
        """
        try:
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=self.api_key)
            
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                # Fetch instruments for token lookup
                self.instruments = pd.DataFrame(self.kite.instruments("NFO"))
                return True
            else:
                print(f"Login URL: {self.kite.login_url()}")
                print("Please login and provide the request_token")
                return False
                
        except ImportError:
            print("kiteconnect not installed. Run: pip install kiteconnect")
            return False
        except Exception as e:
            print(f"Kite connection failed: {e}")
            return False
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for NIFTY instruments."""
        if self.instruments is None:
            return None
        
        # For NIFTY 50 index
        if symbol == "NIFTY 50":
            nse_instruments = pd.DataFrame(self.kite.instruments("NSE"))
            match = nse_instruments[nse_instruments['tradingsymbol'] == 'NIFTY 50']
            if not match.empty:
                return match.iloc[0]['instrument_token']
        
        # For futures/options
        match = self.instruments[self.instruments['tradingsymbol'] == symbol]
        if not match.empty:
            return match.iloc[0]['instrument_token']
        
        return None
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5minute"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Kite.
        
        Intervals: minute, 3minute, 5minute, 10minute, 15minute, 30minute, 
                   60minute, day, week, month
        """
        if self.kite is None:
            raise ConnectionError("Not connected to Kite API")
        
        token = self.get_instrument_token(symbol)
        if token is None:
            raise ValueError(f"Instrument token not found for {symbol}")
        
        data = self.kite.historical_data(
            instrument_token=token,
            from_date=start_date,
            to_date=end_date,
            interval=interval
        )
        
        df = pd.DataFrame(data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return df
    
    def fetch_option_chain(self, symbol: str = "NIFTY", expiry: datetime = None) -> pd.DataFrame:
        """Fetch options chain - Kite doesn't have direct option chain API."""
        # Would need to fetch individual option instruments
        raise NotImplementedError("Use fetch_historical_data for individual options")
    
    def get_nifty_futures_symbol(self, expiry: datetime) -> str:
        """Generate NIFTY futures symbol for given expiry."""
        return f"NIFTY{expiry.strftime('%y%b').upper()}FUT"
    
    def get_nifty_option_symbol(
        self, 
        strike: int, 
        option_type: str, 
        expiry: datetime
    ) -> str:
        """Generate NIFTY option symbol."""
        return f"NIFTY{expiry.strftime('%y%b').upper()}{strike}{option_type}"


class ICICIBreezeAPI(BrokerAPIInterface):
    """
    ICICI Breeze API Implementation.
    
    Requirements:
    - pip install breeze-connect
    - Valid API key and secret from ICICI Direct
    - Active trading account
    
    Documentation: https://api.icicidirect.com/
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, session_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_token = session_token
        self.breeze = None
    
    def connect(self) -> bool:
        """Connect to Breeze API."""
        try:
            from breeze_connect import BreezeConnect
            
            self.breeze = BreezeConnect(api_key=self.api_key)
            
            if self.session_token:
                self.breeze.generate_session(
                    api_secret=self.api_secret,
                    session_token=self.session_token
                )
                return True
            else:
                print("Please provide session_token from ICICI login")
                return False
                
        except ImportError:
            print("breeze-connect not installed. Run: pip install breeze-connect")
            return False
        except Exception as e:
            print(f"Breeze connection failed: {e}")
            return False
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Breeze uses stock_code instead of tokens."""
        return None
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5minute"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Breeze.
        
        Intervals: 1minute, 5minute, 30minute, 1day
        """
        if self.breeze is None:
            raise ConnectionError("Not connected to Breeze API")
        
        # Map interval format
        interval_map = {
            "5minute": "5minute",
            "5min": "5minute",
            "1minute": "1minute",
            "30minute": "30minute",
            "day": "1day"
        }
        
        data = self.breeze.get_historical_data_v2(
            interval=interval_map.get(interval, "5minute"),
            from_date=start_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            to_date=end_date.strftime("%Y-%m-%dT07:00:00.000Z"),
            stock_code=symbol,
            exchange_code="NFO" if "FUT" in symbol or "CE" in symbol or "PE" in symbol else "NSE",
            product_type="futures" if "FUT" in symbol else "options" if "CE" in symbol or "PE" in symbol else "cash"
        )
        
        if data['Success']:
            df = pd.DataFrame(data['Success'])
            df = df.rename(columns={
                'datetime': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return pd.DataFrame()
    
    def fetch_option_chain(self, symbol: str = "NIFTY", expiry: datetime = None) -> pd.DataFrame:
        """Fetch options chain from Breeze."""
        if self.breeze is None:
            raise ConnectionError("Not connected to Breeze API")
        
        data = self.breeze.get_option_chain_quotes(
            stock_code=symbol,
            exchange_code="NFO",
            expiry_date=expiry.strftime("%Y-%m-%d") if expiry else None
        )
        
        if data['Success']:
            return pd.DataFrame(data['Success'])
        
        return pd.DataFrame()


class DataFetcherFactory:
    """Factory class to create appropriate data fetcher based on configuration."""
    
    @staticmethod
    def create_fetcher(
        broker: str,
        credentials: Dict[str, str]
    ) -> Optional[BrokerAPIInterface]:
        """
        Create a broker API instance.
        
        Args:
            broker: One of 'zerodha', 'icici', 'groww'
            credentials: Dict with api_key, api_secret, access_token/session_token
        
        Returns:
            Configured broker API instance
        """
        broker = broker.lower()
        
        if broker == 'zerodha':
            return ZerodhaKiteAPI(
                api_key=credentials.get('api_key'),
                api_secret=credentials.get('api_secret'),
                access_token=credentials.get('access_token')
            )
        elif broker == 'icici':
            return ICICIBreezeAPI(
                api_key=credentials.get('api_key'),
                api_secret=credentials.get('api_secret'),
                session_token=credentials.get('session_token')
            )
        else:
            print(f"Unsupported broker: {broker}")
            return None


# Example usage template
"""
# Zerodha Kite Connect Example
from src.data.broker_api_interface import ZerodhaKiteAPI
import os

kite = ZerodhaKiteAPI(
    api_key=os.environ.get('KITE_API_KEY'),
    api_secret=os.environ.get('KITE_API_SECRET'),
    access_token=os.environ.get('KITE_ACCESS_TOKEN')
)

if kite.connect():
    # Fetch NIFTY spot data
    spot_data = kite.fetch_historical_data(
        symbol='NIFTY 50',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 1),
        interval='5minute'
    )
    
    # Fetch NIFTY futures
    futures_symbol = kite.get_nifty_futures_symbol(datetime(2025, 1, 30))
    futures_data = kite.fetch_historical_data(
        symbol=futures_symbol,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 1),
        interval='5minute'
    )


# ICICI Breeze Example
from src.data.broker_api_interface import ICICIBreezeAPI

breeze = ICICIBreezeAPI(
    api_key=os.environ.get('BREEZE_API_KEY'),
    api_secret=os.environ.get('BREEZE_API_SECRET'),
    session_token=os.environ.get('BREEZE_SESSION_TOKEN')
)

if breeze.connect():
    option_chain = breeze.fetch_option_chain(
        symbol='NIFTY',
        expiry=datetime(2025, 1, 30)
    )
"""
