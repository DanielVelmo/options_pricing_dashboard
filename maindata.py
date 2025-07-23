"""
Financial Data Publisher Program - Main Backend Engine
Handles data gathering, processing, and real-time broadcasting to dashboard
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import norm
import traceback

# External libraries
from supabase import create_client, Client
from realtime import AsyncRealtimeClient, RealtimeSubscribeStates
import ably
from ably import AblyRealtime
import requests
import websockets


# Configuration and Settings
@dataclass
class PublisherConfig:
    """Configuration settings for the publisher"""
    supabase_url: str
    supabase_key: str
    ably_api_key: Optional[str] = None
    log_level: str = "INFO"
    update_interval: float = 1.0  # seconds
    max_retries: int = 3
    database_batch_size: int = 100


# Data Models
@dataclass
class MarketData:
    """Standard market data structure"""
    symbol: str
    price: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    data_type: str = "price"
    source: str = "unknown"


@dataclass
class OptionData:
    """Options specific data structure"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'C' or 'P'
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    implied_volatility: Optional[float] = None
    timestamp: datetime
    source: str = "binance"


@dataclass
class ModelResult:
    """Model calculation results"""
    symbol: str
    model_name: str
    result: float
    confidence: Optional[float] = None
    parameters: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# Black-Scholes Model (from your original code)
def BlackScholes(r: float, S: float, K: float, T: float, sigma: float, tipo: str = 'C') -> float:
    """
    Black-Scholes option pricing model
    r : Interest Rate
    S : Spot Price
    K : Strike Price
    T : Time to expiration in years
    sigma : Annualized Volatility
    tipo : 'C' for Call, 'P' for Put
    """
    try:
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if tipo == 'C':
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif tipo == 'P':
            precio = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
        else:
            raise ValueError("Option type must be 'C' or 'P'")
            
        return precio
    except Exception as e:
        logging.error(f"Black-Scholes calculation error: {e}")
        return 0.0


# Database Manager
class DatabaseManager:
    """Handles all database operations with Supabase"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        self.logger = logging.getLogger(__name__)
    
    async def get_user_api_keys(self, user_id: str) -> Dict[str, str]:
        """Retrieve user's API keys from secure storage"""
        try:
            response = self.client.table('user_api_keys')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if response.data:
                return {
                    'binance_api_key': response.data[0].get('binance_api_key'),
                    'binance_secret_key': response.data[0].get('binance_secret_key'),
                    'alpaca_api_key': response.data[0].get('alpaca_api_key'),
                    'alpaca_secret_key': response.data[0].get('alpaca_secret_key'),
                    'fmp_api_key': response.data[0].get('fmp_api_key'),
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching API keys: {e}")
            return {}
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences and subscriptions"""
        try:
            response = self.client.table('user_preferences')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if response.data:
                return response.data[0]
            return {'symbols': [], 'models': ['black_scholes'], 'real_time_enabled': False}
        except Exception as e:
            self.logger.error(f"Error fetching user preferences: {e}")
            return {'symbols': [], 'models': ['black_scholes'], 'real_time_enabled': False}
    
    async def store_market_data(self, data_batch: List[MarketData]):
        """Store market data in batch"""
        try:
            records = [asdict(data) for data in data_batch]
            # Convert datetime to ISO string for JSON serialization
            for record in records:
                record['timestamp'] = record['timestamp'].isoformat()
            
            self.client.table('market_data').insert(records).execute()
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def store_model_results(self, results_batch: List[ModelResult]):
        """Store model calculation results"""
        try:
            records = [asdict(result) for result in results_batch]
            for record in records:
                record['timestamp'] = record['timestamp'].isoformat()
                if record['parameters']:
                    record['parameters'] = json.dumps(record['parameters'])
            
            self.client.table('model_results').insert(records).execute()
        except Exception as e:
            self.logger.error(f"Error storing model results: {e}")


# Data Providers - Base Class and Implementations
class DataProvider:
    """Base class for all data providers"""
    
    def __init__(self, name: str, api_keys: Dict[str, str]):
        self.name = name
        self.api_keys = api_keys
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_connected = False
    
    async def connect(self):
        """Establish connection to data provider"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Close connection to data provider"""
        raise NotImplementedError
    
    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        raise NotImplementedError
    
    async def get_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Get historical data for analysis"""
        raise NotImplementedError


class BinanceOptionsProvider(DataProvider):
    """Binance options data provider"""
    
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__("binance_options", api_keys)
        self.base_url = "https://eapi.binance.com"
        self.websocket_url = "wss://nbstream.binance.com/eoptions/ws"
        self.websocket = None
    
    async def connect(self):
        """Connect to Binance options API"""
        try:
            # Test connection with exchange info endpoint
            response = requests.get(f"{self.base_url}/eapi/v1/exchangeInfo")
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info("Connected to Binance Options API")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            self.is_connected = False
    
    async def get_option_chain(self, underlying: str = "BTCUSDT") -> List[OptionData]:
        """Get options chain for underlying asset"""
        try:
            if not self.is_connected:
                await self.connect()
            
            response = requests.get(f"{self.base_url}/eapi/v1/exchangeInfo")
            if response.status_code != 200:
                return []
            
            data = response.json()
            options = []
            
            for symbol_info in data.get('optionSymbols', []):
                if symbol_info.get('underlying') == underlying:
                    option = OptionData(
                        symbol=symbol_info['symbol'],
                        underlying=underlying,
                        strike=float(symbol_info['strikePrice']),
                        expiry=datetime.fromtimestamp(symbol_info['expiryDate'] / 1000, tz=timezone.utc),
                        option_type='C' if symbol_info['side'] == 'CALL' else 'P',
                        price=0.0,  # Will be updated with real-time data
                        timestamp=datetime.now(timezone.utc),
                        source="binance"
                    )
                    options.append(option)
            
            return options
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return []
    
    async def get_option_prices(self, symbols: List[str]) -> List[OptionData]:
        """Get current prices for option symbols"""
        try:
            if not symbols:
                return []
            
            # For now, return mock data - implement real API calls
            options = []
            for symbol in symbols:
                # Parse symbol to extract details (simplified)
                option = OptionData(
                    symbol=symbol,
                    underlying="BTCUSDT",
                    strike=50000.0,
                    expiry=datetime.now(timezone.utc),
                    option_type='C',
                    price=1000.0,  # Mock price
                    timestamp=datetime.now(timezone.utc),
                    source="binance"
                )
                options.append(option)
            
            return options
        except Exception as e:
            self.logger.error(f"Error getting option prices: {e}")
            return []


class AlpacaProvider(DataProvider):
    """Alpaca stocks/crypto data provider"""
    
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__("alpaca", api_keys)
        self.base_url = "https://paper-api.alpaca.markets"  # Use paper for testing
        self.data_url = "https://data.alpaca.markets"
    
    async def connect(self):
        """Connect to Alpaca API"""
        try:
            headers = {
                'APCA-API-KEY-ID': self.api_keys.get('alpaca_api_key', ''),
                'APCA-API-SECRET-KEY': self.api_keys.get('alpaca_secret_key', '')
            }
            
            response = requests.get(f"{self.base_url}/v2/account", headers=headers)
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info("Connected to Alpaca API")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.is_connected = False
    
    async def get_latest_quotes(self, symbols: List[str]) -> List[MarketData]:
        """Get latest quotes for symbols"""
        try:
            if not symbols or not self.is_connected:
                return []
            
            # Mock implementation - replace with real Alpaca API calls
            market_data = []
            for symbol in symbols:
                data = MarketData(
                    symbol=symbol,
                    price=100.0,  # Mock price
                    timestamp=datetime.now(timezone.utc),
                    source="alpaca"
                )
                market_data.append(data)
            
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting Alpaca quotes: {e}")
            return []


# Model Processing Engine
class ModelEngine:
    """Handles all financial model calculations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.models = {
            'black_scholes': self.calculate_black_scholes,
            # Future models can be added here
        }
    
    async def process_options_data(self, options_data: List[OptionData], 
                                 underlying_price: float, 
                                 risk_free_rate: float = 0.05) -> List[ModelResult]:
        """Process options data through various models"""
        results = []
        
        for option in options_data:
            # Calculate time to expiration
            time_to_expiry = (option.expiry - datetime.now(timezone.utc)).days / 365.0
            
            if time_to_expiry <= 0:
                continue  # Skip expired options
            
            # Estimate volatility (simplified - in practice, use more sophisticated methods)
            volatility = 0.25  # 25% annual volatility assumption
            
            # Calculate Black-Scholes theoretical price
            bs_result = self.calculate_black_scholes(
                underlying_price, option.strike, time_to_expiry, 
                risk_free_rate, volatility, option.option_type
            )
            
            if bs_result:
                results.append(bs_result)
        
        return results
    
    def calculate_black_scholes(self, spot: float, strike: float, time_to_expiry: float, 
                              risk_free_rate: float, volatility: float, 
                              option_type: str) -> Optional[ModelResult]:
        """Calculate Black-Scholes price"""
        try:
            theoretical_price = BlackScholes(
                risk_free_rate, spot, strike, time_to_expiry, volatility, option_type
            )
            
            return ModelResult(
                symbol=f"BS_{option_type}_{strike}",
                model_name="black_scholes",
                result=theoretical_price,
                parameters={
                    'spot': spot,
                    'strike': strike,
                    'time_to_expiry': time_to_expiry,
                    'risk_free_rate': risk_free_rate,
                    'volatility': volatility,
                    'option_type': option_type
                }
            )
        except Exception as e:
            self.logger.error(f"Black-Scholes calculation error: {e}")
            return None


# Real-time Communication Manager
class CommunicationManager:
    """Manages real-time communication with dashboard"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supabase_client = create_client(config.supabase_url, config.supabase_key)
        self.ably_client = None
        self.realtime_client = None
        
        # Initialize communication channels
        if config.ably_api_key:
            self.ably_client = AblyRealtime(config.ably_api_key)
    
    async def initialize_realtime(self):
        """Initialize Supabase realtime connection"""
        try:
            realtime_url = self.config.supabase_url.replace('https://', 'wss://') + '/realtime/v1'
            self.realtime_client = AsyncRealtimeClient(realtime_url, self.config.supabase_key)
            self.logger.info("Realtime client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize realtime: {e}")
    
    async def broadcast_market_data(self, data: List[MarketData], user_id: str = None):
        """Broadcast market data to subscribed users"""
        try:
            # Prepare data payload
            payload = {
                'type': 'market_data',
                'data': [asdict(d) for d in data],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Convert datetime objects to ISO strings
            for item in payload['data']:
                item['timestamp'] = item['timestamp'].isoformat() if hasattr(item['timestamp'], 'isoformat') else item['timestamp']
            
            # Broadcast via Supabase Realtime (Method 1)
            channel_name = f"market_data_{user_id}" if user_id else "market_data_global"
            
            # Insert into broadcast table (triggers realtime event)
            broadcast_data = {
                'channel': channel_name,
                'event': 'market_update',
                'payload': json.dumps(payload),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase_client.table('realtime_broadcasts').insert(broadcast_data).execute()
            
            # Broadcast via Ably (Method 2 - fallback/alternative)
            if self.ably_client:
                channel = self.ably_client.channels.get(channel_name)
                await channel.publish('market_update', payload)
            
            self.logger.debug(f"Broadcasted market data to {channel_name}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting market data: {e}")
    
    async def broadcast_model_results(self, results: List[ModelResult], user_id: str = None):
        """Broadcast model calculation results"""
        try:
            payload = {
                'type': 'model_results',
                'data': [asdict(r) for r in results],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Convert datetime objects and handle parameters
            for item in payload['data']:
                item['timestamp'] = item['timestamp'].isoformat() if hasattr(item['timestamp'], 'isoformat') else item['timestamp']
                if item['parameters']:
                    item['parameters'] = json.dumps(item['parameters']) if isinstance(item['parameters'], dict) else item['parameters']
            
            channel_name = f"model_results_{user_id}" if user_id else "model_results_global"
            
            # Supabase broadcast
            broadcast_data = {
                'channel': channel_name,
                'event': 'model_update',
                'payload': json.dumps(payload),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase_client.table('realtime_broadcasts').insert(broadcast_data).execute()
            
            # Ably broadcast
            if self.ably_client:
                channel = self.ably_client.channels.get(channel_name)
                await channel.publish('model_update', payload)
            
            self.logger.debug(f"Broadcasted model results to {channel_name}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting model results: {e}")


# Main Publisher Class
class FinancialDataPublisher:
    """Main publisher orchestrating all components"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config)
        self.comm_manager = CommunicationManager(config)
        self.model_engine = ModelEngine(self.db_manager)
        
        # Data providers
        self.providers: Dict[str, DataProvider] = {}
        
        # Runtime state
        self.active_users: Dict[str, Dict] = {}  # user_id -> preferences
        self.is_running = False
        self.data_batch = []
        self.results_batch = []
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing Financial Data Publisher...")
            
            # Initialize communication
            await self.comm_manager.initialize_realtime()
            
            # Load active users and their preferences
            await self.load_active_users()
            
            # Initialize data providers based on user needs
            await self.initialize_providers()
            
            self.logger.info("Publisher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    async def load_active_users(self):
        """Load active users and their preferences"""
        try:
            # Get users who have real-time enabled
            response = self.db_manager.client.table('user_preferences')\
                .select('user_id, symbols, models, real_time_enabled')\
                .eq('real_time_enabled', True)\
                .execute()
            
            self.active_users = {}
            for user_data in response.data:
                user_id = user_data['user_id']
                self.active_users[user_id] = {
                    'symbols': user_data.get('symbols', []),
                    'models': user_data.get('models', ['black_scholes']),
                    'api_keys': await self.db_manager.get_user_api_keys(user_id)
                }
            
            self.logger.info(f"Loaded {len(self.active_users)} active users")
            
        except Exception as e:
            self.logger.error(f"Error loading active users: {e}")
    
    async def initialize_providers(self):
        """Initialize data providers based on user needs"""
        try:
            # Check if any user needs Binance options data
            needs_binance = any(
                'BTC' in str(user_prefs.get('symbols', [])) or 
                'ETH' in str(user_prefs.get('symbols', []))
                for user_prefs in self.active_users.values()
            )
            
            if needs_binance:
                # Use the first user's API keys that has them, or empty dict
                api_keys = {}
                for user_prefs in self.active_users.values():
                    if user_prefs['api_keys'].get('binance_api_key'):
                        api_keys = user_prefs['api_keys']
                        break
                
                binance_provider = BinanceOptionsProvider(api_keys)
                await binance_provider.connect()
                self.providers['binance'] = binance_provider
            
            # Check if any user needs Alpaca data
            needs_alpaca = any(
                any(symbol in ['AAPL', 'TSLA', 'MSFT'] for symbol in user_prefs.get('symbols', []))
                for user_prefs in self.active_users.values()
            )
            
            if needs_alpaca:
                api_keys = {}
                for user_prefs in self.active_users.values():
                    if user_prefs['api_keys'].get('alpaca_api_key'):
                        api_keys = user_prefs['api_keys']
                        break
                
                alpaca_provider = AlpacaProvider(api_keys)
                await alpaca_provider.connect()
                self.providers['alpaca'] = alpaca_provider
            
            self.logger.info(f"Initialized {len(self.providers)} data providers")
            
        except Exception as e:
            self.logger.error(f"Error initializing providers: {e}")
    
    async def run(self):
        """Main execution loop"""
        self.is_running = True
        self.logger.info("Starting publisher main loop...")
        
        try:
            while self.is_running:
                start_time = datetime.now()
                
                # Process data for all active users
                await self.process_cycle()
                
                # Calculate sleep time to maintain update interval
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.config.update_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Publisher stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            await self.cleanup()
    
    async def process_cycle(self):
        """Single processing cycle for all users"""
        try:
            # Collect all unique symbols needed
            all_symbols = set()
            for user_prefs in self.active_users.values():
                all_symbols.update(user_prefs.get('symbols', []))
            
            if not all_symbols:
                return
            
            # Fetch data from providers
            market_data = []
            
            # Get Binance options data
            if 'binance' in self.providers:
                btc_options = await self.providers['binance'].get_option_chain('BTCUSDT')
                # Convert options to market data format for now
                for option in btc_options[:5]:  # Limit to first 5 for testing
                    market_data.append(MarketData(
                        symbol=option.symbol,
                        price=option.price,
                        timestamp=option.timestamp,
                        source=option.source
                    ))
            
            # Get Alpaca data
            if 'alpaca' in self.providers:
                stock_symbols = [s for s in all_symbols if s in ['AAPL', 'TSLA', 'MSFT']]
                if stock_symbols:
                    stock_data = await self.providers['alpaca'].get_latest_quotes(stock_symbols)
                    market_data.extend(stock_data)
            
            # Process through models
            model_results = []
            
            # Example: Calculate Black-Scholes for BTC options
            if 'binance' in self.providers:
                btc_price = 45000.0  # Mock BTC price - get from real data
                btc_options = await self.providers['binance'].get_option_chain('BTCUSDT')
                bs_results = await self.model_engine.process_options_data(
                    btc_options[:3], btc_price  # Limit for testing
                )
                model_results.extend(bs_results)
            
            # Store data in batches
            if market_data:
                self.data_batch.extend(market_data)
            
            if model_results:
                self.results_batch.extend(model_results)
            
            # Batch processing
            if len(self.data_batch) >= self.config.database_batch_size:
                await self.db_manager.store_market_data(self.data_batch)
                self.data_batch.clear()
            
            if len(self.results_batch) >= self.config.database_batch_size:
                await self.db_manager.store_model_results(self.results_batch)
                self.results_batch.clear()
            
            # Broadcast to users who have real-time enabled
            for user_id, user_prefs in self.active_users.items():
                user_market_data = [
                    d for d in market_data 
                    if d.symbol in user_prefs.get('symbols', [])
                ]
                user_model_results = [
                    r for r in model_results
                    if any(model in r.model_name for model in user_prefs.get('models', []))
                ]
                
                if user_market_data:
                    await self.comm_manager.broadcast_market_data(user_market_data, user_id)
                
                if user_model_results:
                    await self.comm_manager.broadcast_model_results(user_model_results, user_id)
            
            self.logger.debug(f"Processed {len(market_data)} market data points, {len(model_results)} model results")
            
        except Exception as e:
            self.logger.error(f"Error in process cycle: {e}")
            self.logger.error(traceback.format_exc())
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        # Disconnect from providers
        for provider in self.providers.values():
            try:
                await provider.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from {provider.name}: {e}")
        
        # Store remaining batched data
        if self.data_batch:
            await self.db_manager.store_market_data(self.data_batch)
        
        if self.results_batch:
            await self.db_manager.store_model_results(self.results_batch)
        
        self.logger.info("Cleanup completed")
    
    def stop(self):
        """Stop the publisher"""
        self.is_running = False


# Main entry point
async def main():
    """Main function to run the publisher"""
    
    # Load configuration from environment variables
    config = PublisherConfig(
        supabase_url=os.getenv('SUPABASE_URL', 'https://your-project.supabase.co'),
        supabase_key=os.getenv('SUPABASE_ANON_KEY', 'your-anon-key'),
        ably_api_key=os.getenv('ABLY_API_KEY'),  # Optional
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        update_interval=float(os.getenv('UPDATE_INTERVAL', '1.0')),
    )
    
    # Create and initialize publisher
    publisher = FinancialDataPublisher(config)
    
    try:
        await publisher.initialize()
        await publisher.run()
    except Exception as e:
        logging.error(f"Publisher failed: {e}")
        logging.error(traceback.format_exc())
    finally:
        await publisher.cleanup()


if __name__ == "__main__":
    # Required dependencies to install:
    """
    pip install supabase realtime-py ably requests websockets numpy scipy
    """
    
    print("Financial Data Publisher - Starting...")
    print("Required environment variables:")
    print("- SUPABASE_URL")
    print("- SUPABASE_ANON_KEY") 
    print("- ABLY_API_KEY (optional)")
    print("- LOG_LEVEL (optional, default: INFO)")
    print("- UPDATE_INTERVAL (optional, default: 1.0 seconds)")
    
    asyncio.run(main())
