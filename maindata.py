"""
Financial Data Publisher Program - Updated for Dashboard Integration
Handles data gathering, processing, and real-time broadcasting to dashboard
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import norm
import traceback
import re
import requests
import hashlib
import hmac
import time as time_module
from urllib.parse import urlencode

# External libraries
from supabase import create_client, Client
from realtime import AsyncRealtimeClient, RealtimeSubscribeStates
import ably
from ably import AblyRealtime
import websockets
import pandas as pd

# Configuration and Settings
@dataclass
class PublisherConfig:
    """Configuration settings for the publisher"""
    supabase_url: str
    supabase_key: str
    supabase_service_key: str  # Service key for admin operations
    ably_api_key: Optional[str] = None
    log_level: str = "INFO"
    update_interval: float = 1.0
    max_retries: int = 3
    database_batch_size: int = 100
    request_check_interval: float = 5.0  # Check for new requests every 5 seconds

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
    """Options specific data structure enhanced with greeks and derived metrics from Binance /eapi endpoints"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'C' or 'P'
    price: float
    timestamp: datetime  
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    implied_volatility: Optional[float] = None
    source: str = "binance"
    # Fields from /eapi/v1/mark
    mark_price: Optional[float] = None
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    mark_iv: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    risk_free_interest: Optional[float] = None
    # Derived fields per notebook logic
    exercise_price: Optional[float] = None
    contract_unit: Optional[int] = None
    bs_price: Optional[float] = None
    itm_probability: Optional[float] = None
    trading_fee: Optional[float] = None
    success_probability: Optional[float] = None

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

@dataclass
class DataRequest:
    """Data request from dashboard"""
    id: str
    user_id: str
    ticker: str
    request_type: str
    timestamp: datetime

# Black-Scholes Model
def BlackScholes(r: float, S: float, K: float, T: float, sigma: float, tipo: str = 'C') -> Tuple[float, float]:
    """Notebook-style Black-Scholes returning (price, itm_probability). T in years."""
    try:
        if T <= 0:
            precio = max(0, S - K) if tipo == 'C' else max(0, K - S)
            itm_p = 1.0 if (tipo == 'C' and S > K) or (tipo == 'P' and K > S) else 0.0
            return float(precio), float(itm_p)
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if tipo == 'C':
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
            itm_p = norm.cdf(d2)
        else:
            precio = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)
            itm_p = norm.cdf(-d2)
        return float(max(0, precio)), float(itm_p)
    except Exception as e:
        logging.error(f"Black-Scholes calculation error: {e}")
        return 0.0, 0.0

def calculate_trading_fee(exercise_price: float, last_price: float, fee_rate: float, option_size: int, contract_unit: int) -> float:
    try:
        transaction_fee = min(fee_rate * exercise_price * contract_unit, last_price * option_size) * option_size
        return float(transaction_fee)
    except Exception:
        return 0.0

def calculate_success_probability(option_type: str, strike: float, exercise_price: float, r: float, sigma: float, T: float, last_price: float, trading_fee: float) -> float:
    try:
        trading_cost = last_price + 2 * trading_fee
        St = exercise_price * np.exp((r + sigma**2 / 2) * T)
        if option_type == 'C':
            expected_relative_return = np.log(St / (strike + trading_cost))
        else:
            expected_relative_return = np.log(strike / (St + trading_cost))
        z = expected_relative_return / (sigma * np.sqrt(T)) - (sigma * np.sqrt(T))
        return float(norm.cdf(z))
    except Exception:
        return 0.0

# Database Manager
class DatabaseManager:
    """Handles all database operations with Supabase"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        self.admin_client: Client = create_client(config.supabase_url, config.supabase_service_key)
        self.logger = logging.getLogger(__name__)
    
    async def get_pending_requests(self) -> List[DataRequest]:
        """Get pending data requests from dashboard users"""
        try:
            # Get requests from the last 10 minutes that haven't been processed
            cutoff_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
            
            response = self.admin_client.table('data_requests')\
                .select('*')\
                .gt('timestamp', cutoff_time)\
                .order('timestamp', desc=True)\
                .execute()
            
            requests = []
            for req_data in response.data:
                request = DataRequest(
                    id=req_data['id'],
                    user_id=req_data['user_id'],
                    ticker=req_data['ticker'],
                    request_type=req_data['request_type'],
                    timestamp=datetime.fromisoformat(req_data['timestamp'].replace('Z', '+00:00'))
                )
                requests.append(request)
            
            return requests
            
        except Exception as e:
            self.logger.error(f"Error fetching pending requests: {e}")
            return []
    
    async def mark_request_processed(self, request_id: str):
        """Mark a request as processed"""
        try:
            # Avoid non-existent 'processed' column; update only safe field
            self.admin_client.table('data_requests')\
                .update({'processed_at': datetime.now(timezone.utc).isoformat()})\
                .eq('id', request_id)\
                .execute()
        except Exception as e:
            self.logger.warning(f"Skipping mark_request_processed due to schema mismatch or permissions: {e}")
    
    async def get_user_api_keys(self, user_id: str) -> Dict[str, str]:
        """Retrieve user's API keys from secure storage"""
        try:
            response = self.admin_client.table('user_api_keys')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if response.data:
                return {
                    'binance_api_key': response.data[0].get('binance_api_key', ''),
                    'binance_secret_key': response.data[0].get('binance_secret_key', ''),
                    'alpaca_api_key': response.data[0].get('alpaca_api_key', ''),
                    'alpaca_secret_key': response.data[0].get('alpaca_secret_key', ''),
                    'fmp_api_key': response.data[0].get('fmp_api_key', ''),
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching API keys: {e}")
            return {}
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's dashboard preferences and subscriptions"""
        try:
            response = self.admin_client.table('user_preferences')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if response.data:
                return response.data[0]
            return {'symbols': [], 'models': ['black_scholes'], 'real_time_enabled': False}
        except Exception as e:
            self.logger.error(f"Error fetching user preferences: {e}")
            return {'symbols': [], 'models': ['black_scholes'], 'real_time_enabled': False}
    
    async def store_options_data(self, options_batch: List[OptionData]):
        """Store options data in the database"""
        try:
            records = []
            for option in options_batch:
                record = {
                    'symbol': option.symbol,
                    'underlying': option.underlying,
                    'strike': float(option.strike),
                    'expiry': option.expiry.isoformat() if option.expiry else None,
                    'option_type': option.option_type,
                    'price': float(option.price),
                    'bid': float(option.bid) if option.bid else None,
                    'ask': float(option.ask) if option.ask else None,
                    'volume': float(option.volume) if option.volume else None,
                    'implied_volatility': float(option.implied_volatility) if option.implied_volatility else None,
                    'timestamp': option.timestamp.isoformat(),
                    'source': option.source
                }
                # Keep storage minimal to avoid schema mismatch errors
                records.append(record)
            
            # Clear old data for the same underlying first
            if records:
                underlying = records[0]['underlying']
                self.admin_client.table('options_data')\
                    .delete()\
                    .eq('underlying', underlying)\
                    .execute()
                
                # Insert new data
                self.admin_client.table('options_data').insert(records).execute()
                self.logger.info(f"Stored {len(records)} options for {underlying}")
                
        except Exception as e:
            self.logger.error(f"Error storing options data: {e}")
    
    async def store_market_data(self, data_batch: List[MarketData]):
        """Store market data in batch"""
        try:
            records = []
            for data in data_batch:
                record = {
                    'symbol': data.symbol,
                    'price': float(data.price),
                    'timestamp': data.timestamp.isoformat(),
                    'bid': float(data.bid) if data.bid else None,
                    'ask': float(data.ask) if data.ask else None,
                    'volume': float(data.volume) if data.volume else None,
                    'data_type': data.data_type,
                    'source': data.source
                }
                records.append(record)
            
            if records:
                self.admin_client.table('market_data').insert(records).execute()
                self.logger.debug(f"Stored {len(records)} market data points")
                
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
    
    async def store_model_results(self, results_batch: List[ModelResult]):
        """Store model calculation results"""
        try:
            records = []
            for result in results_batch:
                record = {
                    'symbol': result.symbol,
                    'model_name': result.model_name,
                    'result': float(result.result),
                    'confidence': float(result.confidence) if result.confidence else None,
                    'parameters': json.dumps(result.parameters) if result.parameters else None,
                    'timestamp': result.timestamp.isoformat()
                }
                records.append(record)
            
            if records:
                self.admin_client.table('model_results').insert(records).execute()
                self.logger.debug(f"Stored {len(records)} model results")
                
        except Exception as e:
            self.logger.error(f"Error storing model results: {e}")

# Enhanced Binance Options Provider
class BinanceOptionsProvider:
    """Enhanced Binance options data provider with real API integration"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_key = api_keys.get('binance_api_key', '')
        self.secret_key = api_keys.get('binance_secret_key', '')
        self.base_url = "https://eapi.binance.com"
        self.logger = logging.getLogger(f"{__name__}.binance_options")
        self.is_connected = False
        
    def _generate_signature(self, params: str) -> str:
        """Generate HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _create_headers(self):
        """Create headers for authenticated requests"""
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    async def connect(self):
        """Connect to Binance options API"""
        try:
            # Test connection with exchange info endpoint
            url = f"{self.base_url}/eapi/v1/exchangeInfo"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                self.is_connected = True
                self.logger.info("Connected to Binance Options API")
                return True
            else:
                self.logger.error(f"Connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            self.is_connected = False
            return False
    
    async def get_underlying_price(self, underlying: str) -> float:
        """Get current price of underlying asset"""
        try:
            # Convert to spot symbol (remove USDT and add back for spot)
            spot_symbol = underlying.replace('USDT', '') + 'USDT'
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': spot_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                self.logger.warning(f"Could not get price for {underlying}, using default")
                return 45000.0  # Default BTC price
                
        except Exception as e:
            self.logger.error(f"Error getting underlying price: {e}")
            return 45000.0  # Default
    
    async def get_options_chain(self, underlying: str = "BTCUSDT") -> List[OptionData]:
        """Get complete options chain for underlying asset"""
        try:
            # Use eAPI ticker and mark endpoints to enrich and compute notebook metrics
            _ = await self.connect()
            ticker_resp = requests.get(f"{self.base_url}/eapi/v1/ticker", timeout=15)
            mark_resp = requests.get(f"{self.base_url}/eapi/v1/mark", timeout=15)
            if ticker_resp.status_code != 200 or mark_resp.status_code != 200:
                self.logger.error(f"Failed to fetch ticker/mark: {ticker_resp.status_code}, {mark_resp.status_code}")
                return []

            ticker_data = ticker_resp.json()
            mark_data = mark_resp.json()
            mark_by_symbol = {m.get('symbol'): m for m in mark_data}

            options: List[OptionData] = []
            now_ts = datetime.now(timezone.utc)
            requested_underlying_symbol = underlying.replace('USDT', '')

            fee_rate = 0.0003
            option_size = 1

            for t in ticker_data:
                sym = t.get('symbol')
                if not sym or '-' not in sym:
                    continue
                u_sym = sym.split('-')[0]
                if u_sym != requested_underlying_symbol:
                    continue

                try:
                    last_price = float(t.get('lastPrice', 0))
                    bid_price = float(t.get('bidPrice', 0)) if t.get('bidPrice') is not None else None
                    ask_price = float(t.get('askPrice', 0)) if t.get('askPrice') is not None else None
                    volume = float(t.get('volume', 0)) if t.get('volume') is not None else None
                    strike_price = float(t.get('strikePrice', 0))
                    exercise_price = float(t.get('exercisePrice', 0))
                    contract_type = (t.get('contractType') or sym[-1]).upper()
                except Exception:
                    continue

                try:
                    exp_str = '20' + sym.split('-')[1]
                    expiry_dt = datetime.strptime(exp_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                days_to_exp = (expiry_dt - datetime.now(timezone.utc)).days
                if days_to_exp <= 0:
                    continue
                T = days_to_exp / 365.0

                md = mark_by_symbol.get(sym, {})
                mark_price = float(md.get('markPrice', 0)) if md.get('markPrice') is not None else None
                mark_iv = float(md.get('markIV', 0)) if md.get('markIV') is not None else None
                bid_iv = float(md.get('bidIV', 0)) if md.get('bidIV') is not None else None
                ask_iv = float(md.get('askIV', 0)) if md.get('askIV') is not None else None
                delta = float(md.get('delta', 0)) if md.get('delta') is not None else None
                theta = float(md.get('theta', 0)) if md.get('theta') is not None else None
                gamma = float(md.get('gamma', 0)) if md.get('gamma') is not None else None
                vega = float(md.get('vega', 0)) if md.get('vega') is not None else None
                r = float(md.get('riskFreeInterest', 0)) if md.get('riskFreeInterest') is not None else 0.0

                if u_sym in ['ETH', 'BTC', 'BNB', 'SOL']:
                    contract_unit = 1
                elif u_sym in ['XRP']:
                    contract_unit = 100
                elif u_sym in ['DOG']:
                    contract_unit = 1000
                else:
                    contract_unit = 1

                bs_price, itm_p = BlackScholes(r, exercise_price, strike_price, T, mark_iv or 0.0, tipo=contract_type)
                trading_fee = min(fee_rate * exercise_price * contract_unit, last_price * option_size) * option_size

                # Success probability per notebook
                try:
                    trading_cost = last_price + 2 * trading_fee
                    St = exercise_price * np.exp((r + (mark_iv or 0.0)**2 / 2) * T)
                    if contract_type == 'C':
                        expected_relative_return = np.log(St / (strike_price + trading_cost))
                    else:
                        expected_relative_return = np.log(strike_price / (St + trading_cost))
                    z = expected_relative_return / ((mark_iv or 1e-9) * np.sqrt(T)) - ((mark_iv or 0.0) * np.sqrt(T))
                    success_prob = float(norm.cdf(z))
                except Exception:
                    success_prob = 0.0

                options.append(OptionData(
                    symbol=sym,
                    underlying=underlying,
                    strike=strike_price,
                    expiry=expiry_dt,
                    option_type=contract_type,
                    price=last_price,
                    bid=bid_price,
                    ask=ask_price,
                    volume=volume,
                    implied_volatility=mark_iv,
                    timestamp=now_ts,
                    source="binance",
                    mark_price=mark_price,
                    bid_iv=bid_iv,
                    ask_iv=ask_iv,
                    mark_iv=mark_iv,
                    delta=delta,
                    theta=theta,
                    gamma=gamma,
                    vega=vega,
                    risk_free_interest=r,
                    exercise_price=exercise_price,
                    contract_unit=contract_unit,
                    bs_price=bs_price,
                    itm_probability=itm_p,
                    trading_fee=float(trading_fee),
                    success_probability=success_prob
                ))

            self.logger.info(f"Retrieved {len(options)} options for {underlying}")
            return options
            
        except Exception as e:
            self.logger.error(f"Error getting options chain: {e}")
            return []
    
    async def get_option_prices(self, symbols: List[str]) -> List[Dict]:
        """Get current prices for option symbols"""
        try:
            if not symbols:
                return []
            
            # Get ticker prices for all symbols at once
            url = f"{self.base_url}/eapi/v1/ticker"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                all_tickers = response.json()
                # Filter for our symbols
                return [ticker for ticker in all_tickers if ticker['symbol'] in symbols]
            else:
                self.logger.error(f"Failed to get option prices: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting option prices: {e}")
            return []

# Model Processing Engine
class ModelEngine:
    """Enhanced model processing engine"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.models = {
            'black_scholes': self.calculate_black_scholes_batch,
        }
    
    async def process_options_request(self, request: DataRequest, options_data: List[OptionData]) -> List[ModelResult]:
        """Process options data through various models for a specific request"""
        try:
            if not options_data:
                return []
            
            # Get user preferences to know which models to run
            user_prefs = await self.db_manager.get_user_preferences(request.user_id)
            enabled_models = user_prefs.get('models', ['black_scholes'])
            
            # Get underlying price
            underlying = options_data[0].underlying
            binance_provider = BinanceOptionsProvider({})
            underlying_price = await binance_provider.get_underlying_price(underlying)
            
            # Get risk parameters from user preferences
            risk_free_rate = user_prefs.get('risk_free_rate', 0.05)
            default_volatility = user_prefs.get('default_volatility', 0.25)
            
            results = []
            
            # Run Black-Scholes if enabled
            if 'black_scholes' in enabled_models:
                bs_results = await self.calculate_black_scholes_batch(
                    options_data, underlying_price, risk_free_rate, default_volatility
                )
                results.extend(bs_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing options request: {e}")
            return []
    
    async def calculate_black_scholes_batch(self, options_data: List[OptionData], 
                                          underlying_price: float, 
                                          risk_free_rate: float, 
                                          default_volatility: float) -> List[ModelResult]:
        """Calculate Black-Scholes prices for a batch of options"""
        results = []
        
        for option in options_data:
            try:
                # Calculate time to expiration
                time_to_expiry = (option.expiry - datetime.now(timezone.utc)).days / 365.0
                
                if time_to_expiry <= 0:
                    continue  # Skip expired options
                
                # Prefer mark_iv/exercise_price when present
                volatility = (getattr(option, 'mark_iv', None) or option.implied_volatility or default_volatility)
                S_for_bs = getattr(option, 'exercise_price', None) or underlying_price
                theoretical_price, itm_p = BlackScholes(
                    risk_free_rate, S_for_bs, option.strike, 
                    time_to_expiry, volatility, option.option_type
                )
                
                result = ModelResult(
                    symbol=option.symbol,
                    model_name="black_scholes",
                    result=theoretical_price,
                    confidence=0.85,  # Static confidence for BS model
                    parameters={
                        'underlying_price': S_for_bs,
                        'strike': option.strike,
                        'time_to_expiry': time_to_expiry,
                        'risk_free_rate': risk_free_rate,
                        'volatility': volatility,
                        'option_type': option.option_type,
                        'market_price': option.price,
                        'itm_probability': itm_p
                    }
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error calculating BS for {option.symbol}: {e}")
                continue
        
        return results

# Real-time Communication Manager
class CommunicationManager:
    """Enhanced communication manager with better real-time support"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supabase_client = create_client(config.supabase_url, config.supabase_service_key)
        self.ably_client = None
        self.realtime_client = None
        
        # Initialize communication channels
        if config.ably_api_key:
            try:
                self.ably_client = AblyRealtime(config.ably_api_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Ably: {e}")
    
    async def initialize_realtime(self):
        """Initialize Supabase realtime connection"""
        try:
            realtime_url = self.config.supabase_url.replace('https://', 'wss://') + '/realtime/v1'
            self.realtime_client = AsyncRealtimeClient(realtime_url, self.config.supabase_service_key)
            self.logger.info("Realtime client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize realtime: {e}")
    
    async def notify_data_processed(self, request: DataRequest, options_count: int, results_count: int):
        """Notify dashboard that data has been processed"""
        try:
            # Create notification payload
            payload = {
                'type': 'data_processed',
                'request_id': request.id,
                'ticker': request.ticker,
                'options_count': options_count,
                'results_count': results_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Send via Supabase broadcast
            channel_name = f"data_updates_{request.user_id}"
            broadcast_data = {
                'channel': channel_name,
                'event': 'data_processed',
                'payload': json.dumps(payload),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase_client.table('realtime_broadcasts').insert(broadcast_data).execute()
            
            # Send via Ably if available
            if self.ably_client:
                try:
                    channel = self.ably_client.channels.get(channel_name)
                    await channel.publish('data_processed', payload)
                except Exception as e:
                    self.logger.warning(f"Ably broadcast failed: {e}")
            
            self.logger.info(f"Notified user {request.user_id} about processed data for {request.ticker}")
            
        except Exception as e:
            self.logger.error(f"Error notifying data processed: {e}")
    
    async def broadcast_model_results(self, results: List[ModelResult], user_id: str):
        """Broadcast model calculation results to specific user"""
        try:
            if not results:
                return
                
            payload = {
                'type': 'model_results',
                'data': [asdict(r) for r in results],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Convert datetime objects and handle parameters
            for item in payload['data']:
                item['timestamp'] = item['timestamp'].isoformat() if hasattr(item['timestamp'], 'isoformat') else item['timestamp']
                if item['parameters'] and isinstance(item['parameters'], dict):
                    item['parameters'] = json.dumps(item['parameters'])
            
            channel_name = f"model_results_{user_id}"
            
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
                try:
                    channel = self.ably_client.channels.get(channel_name)
                    await channel.publish('model_update', payload)
                except Exception as e:
                    self.logger.warning(f"Ably model results broadcast failed: {e}")
            
            self.logger.debug(f"Broadcasted {len(results)} model results to user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting model results: {e}")

# Main Publisher Class - Enhanced
class FinancialDataPublisher:
    """Enhanced main publisher orchestrating all components"""
    
    def __init__(self, config: PublisherConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config)
        self.comm_manager = CommunicationManager(config)
        self.model_engine = ModelEngine(self.db_manager)
        
        # Data providers cache
        self.providers_cache: Dict[str, Dict[str, Any]] = {}
        
        # Runtime state
        self.is_running = False
        self.processed_requests = set()  # Track processed requests
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('publisher.log')
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing Enhanced Financial Data Publisher...")
            
            # Initialize communication
            await self.comm_manager.initialize_realtime()
            
            self.logger.info("Publisher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    async def create_binance_provider(self, user_id: str) -> Optional[BinanceOptionsProvider]:
        """Create Binance provider for specific user"""
        try:
            # Check cache first
            if user_id in self.providers_cache and 'binance' in self.providers_cache[user_id]:
                return self.providers_cache[user_id]['binance']
            
            # Get user's API keys
            api_keys = await self.db_manager.get_user_api_keys(user_id)
            
            if not api_keys.get('binance_api_key'):
                self.logger.warning(f"No Binance API keys found for user {user_id}")
                return None
            
            # Create provider
            provider = BinanceOptionsProvider(api_keys)
            
            # Cache it
            if user_id not in self.providers_cache:
                self.providers_cache[user_id] = {}
            self.providers_cache[user_id]['binance'] = provider
            
            return provider
            
        except Exception as e:
            self.logger.error(f"Error creating Binance provider for user {user_id}: {e}")
            return None
    
    async def process_data_request(self, request: DataRequest):
        """Process a single data request from dashboard"""
        try:
            self.logger.info(f"Processing request: {request.ticker} for user {request.user_id}")
            
            if request.request_type == 'options_data':
                await self.process_options_request(request)
            else:
                self.logger.warning(f"Unknown request type: {request.request_type}")
            
            # Mark as processed
            await self.db_manager.mark_request_processed(request.id)
            self.processed_requests.add(request.id)
            
        except Exception as e:
            self.logger.error(f"Error processing request {request.id}: {e}")
            self.logger.error(traceback.format_exc())
    
    async def process_options_request(self, request: DataRequest):
        """Process options data request"""
        try:
            # Create Binance provider for this user
            binance_provider = await self.create_binance_provider(request.user_id)
            
            if not binance_provider:
                self.logger.warning(f"Cannot process options request - no Binance provider for user {request.user_id}")
                return
            
            # Fetch options data
            self.logger.info(f"Fetching options chain for {request.ticker}")
            options_data = await binance_provider.get_options_chain(request.ticker)
            
            if not options_data:
                self.logger.warning(f"No options data found for {request.ticker}")
                return
            
            # Store options data in database
            await self.db_manager.store_options_data(options_data)
            
            # Process through models
            model_results = await self.model_engine.process_options_request(request, options_data)
            
            # Store model results
            if model_results:
                await self.db_manager.store_model_results(model_results)
                
                # Broadcast results to user
                await self.comm_manager.broadcast_model_results(model_results, request.user_id)
            
            # Notify dashboard that processing is complete
            await self.comm_manager.notify_data_processed(
                request, len(options_data), len(model_results)
            )
            
            self.logger.info(f"Successfully processed {request.ticker}: {len(options_data)} options, {len(model_results)} results")
            
        except Exception as e:
            self.logger.error(f"Error processing options request: {e}")
            self.logger.error(traceback.format_exc())
    
    async def check_for_requests(self):
        """Check for new data requests from dashboard"""
        try:
            # Get pending requests
            pending_requests = await self.db_manager.get_pending_requests()
            
            # Filter out already processed requests
            new_requests = [
                req for req in pending_requests 
                if req.id not in self.processed_requests
            ]
            
            if new_requests:
                self.logger.info(f"Found {len(new_requests)} new data requests")
                
                # Process each request
                for request in new_requests:
                    try:
                        await self.process_data_request(request)
                    except Exception as e:
                        self.logger.error(f"Failed to process request {request.id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error checking for requests: {e}")
    
    async def run(self):
        """Enhanced main execution loop"""
        self.is_running = True
        self.logger.info("Starting enhanced publisher main loop...")
        
        try:
            while self.is_running:
                start_time = datetime.now()
                
                # Check for new requests from dashboard
                await self.check_for_requests()
                
                # Calculate sleep time
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.config.request_check_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Publisher stopped by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Enhanced cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        # Clear provider cache
        self.providers_cache.clear()
        
        self.logger.info("Cleanup completed")
    
    def stop(self):
        """Stop the publisher"""
        self.is_running = False

# Main entry point
async def main():
    """Main function to run the enhanced publisher"""
    
    # Load configuration from environment variables
    config = PublisherConfig(
        supabase_url=('https://pcfqzrzelgvutthbijzg.supabase.co'),
        supabase_key=('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBjZnF6cnplbGd2dXR0aGJpanpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2MDY4ODYsImV4cCI6MjA2ODE4Mjg4Nn0.zVUs0K7vNIUvwxJCesUsVhjpZn5vTm0VrCoiuVCo07k'),
        supabase_service_key=('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBjZnF6cnplbGd2dXR0aGJpanpnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjYwNjg4NiwiZXhwIjoyMDY4MTgyODg2fQ._84Caw717UmbLHScCLUc3LvQdXHUv5inyJnKusT2zyE'),
        ably_api_key=('ABLY_API_KEY'),  # Optional
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        update_interval=float(os.getenv('UPDATE_INTERVAL', '1.0')),
        request_check_interval=float(os.getenv('REQUEST_CHECK_INTERVAL', '5.0')),
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
    pip install supabase realtime-py ably requests websockets numpy scipy pandas
    """
    
    print("Enhanced Financial Data Publisher - Starting...")
    print("Required environment variables:")
    print("- SUPABASE_URL")
    print("- SUPABASE_ANON_KEY")
    print("- SUPABASE_SERVICE_KEY (for admin operations)")
    print("- ABLY_API_KEY (optional)")
    print("- LOG_LEVEL (optional, default: INFO)")
    print("- UPDATE_INTERVAL (optional, default: 1.0 seconds)")
    print("- REQUEST_CHECK_INTERVAL (optional, default: 5.0 seconds)")
    
    asyncio.run(main())
