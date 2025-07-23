"""
Financial Dashboard - Streamlit Frontend
Real-time options trading dashboard with Supabase authentication and P&L analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta, timezone
import json
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

# External libraries
from supabase import create_client, Client
from st_supabase_connection import SupabaseConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console/terminal
)


# Set page configuration
st.set_page_config(
    page_title="Financial Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Black-Scholes Model (from original code)
from scipy.stats import norm


def BlackScholes(
    r: float, S: float, K: float, T: float, sigma: float, tipo: str = "C"
) -> float:
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
        if T <= 0:
            return max(0, S - K) if tipo == "C" else max(0, K - S)

        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if tipo == "C":
            precio = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(
                d2, 0, 1
            )
        elif tipo == "P":
            precio = K * np.exp(-r * T) * norm.cdf(
                -d2, 0, 1
            ) - S * norm.cdf(-d1, 0, 1)
        else:
            raise ValueError("Option type must be 'C' or 'P'")

        return max(0, precio)
    except Exception as e:
        logging.error(f"Black-Scholes calculation error: {e}", exc_info=True)
        st.error(f"Black-Scholes calculation error: {e}")
        return 0.0


def HeatMapMatrix(
    Spot_Prices, Volatilities, Strike, Interest_Rate, Days_to_Exp, type="C"
):
    """Generate heatmap matrix for Black-Scholes pricing"""
    M = np.zeros(shape=(len(Spot_Prices), len(Volatilities)))
    T = Days_to_Exp / 365
    for i in range(len(Spot_Prices)):
        for j in range(len(Volatilities)):
            BS_result = BlackScholes(
                Interest_Rate, Spot_Prices[i], Strike, T, Volatilities[j], type
            )
            M[i, j] = round(BS_result, 2)
    return M


# Authentication and Database Management
@dataclass
class UserSession:
    user_id: str
    email: str
    api_keys: Dict[str, str]
    preferences: Dict[str, Any]
    is_authenticated: bool = False


class AuthManager:
    """Handle user authentication with Supabase"""

    def __init__(self):
        self.supabase_url = "https://pcfqzrzelgvutthbijzg.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBjZnF6cnplbGd2dXR0aGJpanpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2MDY4ODYsImV4cCI6MjA2ODE4Mjg4Nn0.zVUs0K7vNIUvwxJCesUsVhjpZn5vTm0VrCoiuVCo07k"

        if not self.supabase_url or not self.supabase_key:
            st.error(
                "Please configure SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets"
            )
            st.stop()

        self.client = create_client(self.supabase_url, self.supabase_key)

        # Initialize session state
        if "user_session" not in st.session_state:
            st.session_state.user_session = None
        if "auth_token" not in st.session_state:
            st.session_state.auth_token = None

    def show_auth_page(self):
        """Display login/signup form"""
        st.title("🔐 Login to Financial Dashboard")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.subheader("Login")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")

                if login_button and email and password:
                    if self.login(email, password):
                        st.success("Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab2:
            st.subheader("Sign Up")
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email")
                password = st.text_input(
                    "Password", type="password", key="signup_password"
                )
                confirm_password = st.text_input(
                    "Confirm Password", type="password"
                )
                signup_button = st.form_submit_button("Sign Up")

                if signup_button and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif self.signup(email, password):
                        st.success("Account created! Please login.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Signup failed. Email might already exist.")

    def login(self, email: str, password: str) -> bool:
        """Authenticate user"""
        logging.info(f"Login attempt for {email}")
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            if response.user:
                user_id = response.user.id
                st.session_state.auth_token = response.session.access_token

                # Load user preferences and API keys
                api_keys = self.get_user_api_keys(user_id)
                preferences = self.get_user_preferences(user_id)

                st.session_state.user_session = UserSession(
                    user_id=user_id,
                    email=email,
                    api_keys=api_keys,
                    preferences=preferences,
                    is_authenticated=True,
                )
                logging.info(f"Login successful for user {user_id}")
                return True
        except Exception as e:
            logging.error(f"Login error for {email}: {e}", exc_info=True)
            st.error(f"Login error: {e}")
            return False

    def signup(self, email: str, password: str) -> bool:
        """Create new user account"""
        logging.info(f"Signup attempt for {email}")
        try:
            response = self.client.auth.sign_up(
                {"email": email, "password": password}
            )

            if response.user:
                # Initialize default preferences
                user_id = response.user.id
                self.save_user_preferences(
                    user_id,
                    {
                        "symbols": [],
                        "models": ["black_scholes"],
                        "real_time_enabled": False,
                        "risk_free_rate": 0.05,
                        "default_volatility": 0.25,
                    },
                )
                logging.info(f"Signup successful for user {user_id}")
                return True
        except Exception as e:
            logging.error(f"Signup error for {email}: {e}", exc_info=True)
            st.error(f"Signup error: {e}")
            return False

    def get_user_api_keys(self, user_id: str) -> Dict[str, str]:
        """Get user's stored API keys """
        try:
            response = (
                self.client.table("user_api_keys")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                api_data = response.data[0]
                return {
                    "binance_api_key": api_data.get("binance_api_key") or "",
                    "binance_secret_key": api_data.get("binance_secret_key") or "",
                    "alpaca_api_key": api_data.get("alpaca_api_key") or "",
                    "alpaca_secret_key": api_data.get("alpaca_secret_key") or "",
                    "fmp_api_key": api_data.get("fmp_api_key") or "",
                }
            return {
                "binance_api_key": "",
                "binance_secret_key": "",
                "alpaca_api_key": "",
                "alpaca_secret_key": "",
                "fmp_api_key": "",
            }
        except Exception as e:
            logging.error(
                f"Error loading API keys for user {user_id}: {e}",
                exc_info=True,
            )
            st.error(f"Error loading API keys: {e}")
            return {
                "binance_api_key": "",
                "binance_secret_key": "",
                "alpaca_api_key": "",
                "alpaca_secret_key": "",
                "fmp_api_key": "",
            }

    def save_user_api_keys(self, user_id: str, api_keys: Dict[str, str]):
        """Save user's API keys """
        try:
            # First check if a record exists
            existing = (
                self.client.table("user_api_keys")
                .select("user_id")
                .eq("user_id", user_id)
                .execute()
            )
            
            # Prepare the data
            data = {
                "user_id": user_id,
                "binance_api_key": api_keys.get("binance_api_key", ""),
                "binance_secret_key": api_keys.get("binance_secret_key", ""),
                "alpaca_api_key": api_keys.get("alpaca_api_key", ""),
                "alpaca_secret_key": api_keys.get("alpaca_secret_key", ""),
                "fmp_api_key": api_keys.get("fmp_api_key", ""),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            if existing.data and len(existing.data) > 0:
                # Update existing record
                self.client.table("user_api_keys").update(data).eq("user_id", user_id).execute()
            else:
                # Insert new record
                self.client.table("user_api_keys").insert(data).execute()

            # Update session state
            if st.session_state.user_session:
                st.session_state.user_session.api_keys = api_keys

            st.success("API keys saved successfully!")
            logging.info(f"API keys saved for user {user_id}")
        except Exception as e:
            logging.error(
                f"Error saving API keys for user {user_id}: {e}", exc_info=True
            )
            st.error(f"Error saving API keys: {e}")

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            response = (
                self.client.table("user_preferences")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            if response.data:
                return response.data[0]

            # Return defaults if no preferences found
            return {
                "symbols": [],
                "models": ["black_scholes"],
                "real_time_enabled": False,
                "risk_free_rate": 0.05,
                "default_volatility": 0.25,
            }
        except Exception as e:
            logging.error(
                f"Error loading preferences for user {user_id}: {e}",
                exc_info=True,
            )
            st.error(f"Error loading preferences: {e}")
            return {}

    def save_user_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ):
        """Save user preferences"""
        try:
            self.client.table("user_preferences").upsert(
                {
                    "user_id": user_id,
                    **preferences,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()

            # Update session state
            if st.session_state.user_session:
                st.session_state.user_session.preferences = preferences

            st.success("Preferences saved!")
            logging.info(f"Preferences saved for user {user_id}")
        except Exception as e:
            logging.error(
                f"Error saving preferences for user {user_id}: {e}",
                exc_info=True,
            )
            st.error(f"Error saving preferences: {e}")

    def logout(self):
        """Logout user"""
        try:
            user_email = st.session_state.user_session.email
            self.client.auth.sign_out()
            st.session_state.user_session = None
            st.session_state.auth_token = None
            logging.info(f"User {user_email} logged out.")
            st.rerun()
        except Exception as e:
            logging.error(f"Logout error: {e}", exc_info=True)
            st.error(f"Logout error: {e}")


# Real-time Communication Manager 
class DashboardCommunicator:
    """Handle real-time communication with the publisher """

    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
        self.supabase_client = auth_manager.client
        self.ably_client = None
        self._ably_initialized = False

        # Remove automatic Ably initialization to prevent event loop errors
        # We'll initialize it when needed in a proper async context

    def _initialize_ably_if_needed(self):
        """Initialize Ably client safely"""
        if self._ably_initialized or self.ably_client is not None:
            return

        ably_key = st.secrets.get("ABLY_API_KEY")
        if not ably_key:
            logging.info("No Ably API key found, real-time features will be limited")
            return

        try:
            # Import Ably here to avoid initialization issues
            from ably import AblyRest
            
            # Use REST client instead of Realtime to avoid event loop issues
            self.ably_client = AblyRest(ably_key)
            self._ably_initialized = True
            logging.info("Ably REST client initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize Ably REST client: {e}")
            st.warning(f"Real-time features limited: {e}")

    def request_ticker_data(self, ticker: str):
        """Request data for a specific ticker from the publisher"""
        if not st.session_state.user_session:
            return

        try:
            user_id = st.session_state.user_session.user_id
            logging.info(f"User {user_id} requesting data for {ticker}")

            # Save the requested ticker to user preferences
            current_symbols = st.session_state.user_session.preferences.get(
                "symbols", []
            )
            if ticker not in current_symbols:
                current_symbols.append(ticker)

                # Update preferences
                updated_preferences = (
                    st.session_state.user_session.preferences.copy()
                )
                updated_preferences["symbols"] = current_symbols
                updated_preferences["real_time_enabled"] = True

                self.auth_manager.save_user_preferences(
                    user_id, updated_preferences
                )

            # Send request to publisher via Supabase
            request_data = {
                "user_id": user_id,
                "ticker": ticker,
                "request_type": "options_data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.supabase_client.table("data_requests").insert(
                request_data
            ).execute()

            # Try to send via Ably REST if available
            self._initialize_ably_if_needed()
            if self.ably_client:
                try:
                    channel = self.ably_client.channels.get("data_requests")
                    channel.publish("new_request", request_data)
                    logging.info(f"Request sent via Ably REST for {ticker}")
                except Exception as e:
                    logging.warning(f"Failed to send via Ably: {e}")

            st.success(
                f"Requested data for {ticker}. Publisher will fetch options data shortly."
            )

        except Exception as e:
            logging.error(
                f"Error requesting ticker data for {ticker}: {e}",
                exc_info=True,
            )
            st.error(f"Error requesting ticker data: {e}")

    def get_options_data(self, ticker: str) -> List[Dict]:
        """Get options data for a ticker"""
        try:
            logging.info(f"Fetching options data for {ticker}")
            response = (
                self.supabase_client.table("options_data")
                .select("*")
                .eq("underlying", ticker.upper())
                .order("strike")
                .limit(50)
                .execute()
            )

            return response.data
        except Exception as e:
            logging.error(
                f"Error fetching options data for {ticker}: {e}", exc_info=True
            )
            st.error(f"Error fetching options data: {e}")
            return []

    def get_model_results(self, ticker: str) -> List[Dict]:
        """Get model calculation results for a ticker"""
        try:
            response = (
                self.supabase_client.table("model_results")
                .select("*")
                .ilike("symbol", f"%{ticker.upper()}%")
                .order("timestamp", desc=True)
                .limit(100)
                .execute()
            )

            return response.data
        except Exception as e:
            logging.error(
                f"Error fetching model results for {ticker}: {e}",
                exc_info=True,
            )
            st.error(f"Error fetching model results: {e}")
            return []


# P&L Probability Calculator
class PLProbabilityCalculator:
    """Calculate positive P&L probability for options"""

    @staticmethod
    def calculate_positive_pl_probability(
        option_type: str,
        strike: float,
        current_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_purchase_price: float,
        transaction_cost: float = 0.0,
        num_simulations: int = 10000,
    ) -> Dict[str, float]:
        """
        Calculate the probability of positive P&L using Monte Carlo simulation
        """
        try:
            if time_to_expiry <= 0:
                return {
                    "positive_pl_probability": 0.0,
                    "breakeven_price": strike + option_purchase_price
                    if option_type == "C"
                    else strike - option_purchase_price,
                    "expected_pl": -option_purchase_price - transaction_cost,
                    "max_loss": option_purchase_price + transaction_cost,
                }

            dt = time_to_expiry / 365

            # Generate random price paths using geometric Brownian motion
            Z = np.random.normal(0, 1, num_simulations)
            final_prices = current_price * np.exp(
                (risk_free_rate - 0.5 * volatility**2) * dt
                + volatility * np.sqrt(dt) * Z
            )

            # Calculate option values at expiration
            if option_type.upper() == "C":
                option_values = np.maximum(final_prices - strike, 0)
                breakeven = strike + option_purchase_price + transaction_cost
            else:  # Put
                option_values = np.maximum(strike - final_prices, 0)
                breakeven = strike - option_purchase_price - transaction_cost

            # Calculate P&L
            pl_values = option_values - option_purchase_price - transaction_cost

            # Calculate statistics
            positive_pl_count = np.sum(pl_values > 0)
            positive_pl_probability = positive_pl_count / num_simulations
            expected_pl = np.mean(pl_values)
            max_loss = option_purchase_price + transaction_cost

            return {
                "positive_pl_probability": positive_pl_probability,
                "breakeven_price": breakeven,
                "expected_pl": expected_pl,
                "max_loss": max_loss,
                "simulated_pl_values": pl_values,
            }

        except Exception as e:
            logging.error(
                f"Error calculating P&L probability: {e}", exc_info=True
            )
            st.error(f"Error calculating P&L probability: {e}")
            return {
                "positive_pl_probability": 0.0,
                "breakeven_price": 0.0,
                "expected_pl": 0.0,
                "max_loss": 0.0,
            }


# Dashboard Pages
def show_dashboard_header():
    """Display dashboard header with user info"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("📊 Financial Trading Dashboard")

    with col2:
        if st.session_state.user_session:
            st.write(f"Welcome, {st.session_state.user_session.email}")

    with col3:
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.auth_manager.logout()


def show_api_keys_setup():
    """Show API keys configuration page"""
    st.header("🔑 API Keys Setup")

    if not st.session_state.user_session:
        st.error("Please login first")
        return

    st.write("Configure your API keys for data providers:")

    current_keys = st.session_state.user_session.api_keys

    with st.form("api_keys_form"):
        st.subheader("Binance API Keys")
        binance_api_key = st.text_input(
            "Binance API Key",
            value=current_keys.get("binance_api_key", ""),
            type="password",
        )
        binance_secret_key = st.text_input(
            "Binance Secret Key",
            value=current_keys.get("binance_secret_key", ""),
            type="password",
        )

        st.subheader("Alpaca API Keys")
        alpaca_api_key = st.text_input(
            "Alpaca API Key",
            value=current_keys.get("alpaca_api_key", ""),
            type="password",
        )
        alpaca_secret_key = st.text_input(
            "Alpaca Secret Key",
            value=current_keys.get("alpaca_secret_key", ""),
            type="password",
        )

        st.subheader("Financial Modeling Prep API Key")
        fmp_api_key = st.text_input(
            "FMP API Key",
            value=current_keys.get("fmp_api_key", ""),
            type="password",
        )

        if st.form_submit_button("💾 Save API Keys"):
            api_keys = {
                "binance_api_key": binance_api_key,
                "binance_secret_key": binance_secret_key,
                "alpaca_api_key": alpaca_api_key,
                "alpaca_secret_key": alpaca_secret_key,
                "fmp_api_key": fmp_api_key,
            }

            st.session_state.auth_manager.save_user_api_keys(
                st.session_state.user_session.user_id, api_keys
            )


def show_options_analysis():
    """Show options analysis page with P&L probability """
    st.header("📈 Options Analysis & P&L Probability")

    if not st.session_state.user_session:
        st.error("Please login first")
        return

    # Check if user has Binance API keys
    api_keys = st.session_state.user_session.api_keys
    has_binance_keys = api_keys.get(
        "binance_api_key"
    ) and api_keys.get("binance_secret_key")

    if not has_binance_keys:
        st.warning(
            "⚠️ Please configure your Binance API keys in the Settings to access full functionality."
        )

    # Ticker input
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Enter ticker symbol (e.g., BTCUSDT, ETHUSDT):",
            placeholder="BTCUSDT",
            key="ticker_input",
        )

    with col2:
        if st.button("🔍 Request Data", disabled=not ticker_input):
            if ticker_input:
                st.session_state.dashboard_comm.request_ticker_data(
                    ticker_input.upper()
                )

    # Show options data if available
    if ticker_input:
        logging.info(
            f"Rendering options analysis for ticker: {ticker_input}"
        )
        options_data = st.session_state.dashboard_comm.get_options_data(
            ticker_input
        )

        if options_data:
            st.subheader(f"Options Chain for {ticker_input.upper()}")

            # Convert to DataFrame for better display
            df = pd.DataFrame(options_data)

            if not df.empty:
                # Get current underlying price (you might want to fetch this from your market data)
                underlying_price = st.number_input(
                    f"Current {ticker_input.upper()} Price:",
                    value=45000.0,  # Default value
                    min_value=0.0,
                    key="underlying_price",
                )

                # Risk parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    risk_free_rate = st.number_input(
                        "Risk-free Rate",
                        value=0.05,
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    )
                with col2:
                    default_volatility = st.number_input(
                        "Default Volatility",
                        value=0.25,
                        min_value=0.0,
                        max_value=5.0,
                        format="%.4f",
                    )
                with col3:
                    transaction_cost = st.number_input(
                        "Transaction Cost", value=10.0, min_value=0.0
                    )

                # Calculate P&L probabilities
                calculator = PLProbabilityCalculator()

                enhanced_data = []
                for _, row in df.iterrows():
                    try:
                        # Parse expiry date
                        if "expiry" in row and row["expiry"]:
                            expiry_date = pd.to_datetime(row["expiry"])
                            time_to_expiry = (
                                expiry_date - datetime.now(timezone.utc)
                            ).days
                        else:
                            time_to_expiry = 30  # Default to 30 days

                        # Use implied volatility if available, otherwise default
                        volatility = (
                            row.get("implied_volatility", default_volatility)
                            or default_volatility
                        )

                        # Calculate theoretical Black-Scholes price
                        bs_price = BlackScholes(
                            risk_free_rate,
                            underlying_price,
                            row["strike"],
                            time_to_expiry / 365,
                            volatility,
                            row["option_type"],
                        )

                        # Use market price if available, otherwise BS price
                        market_price = row.get("price", bs_price) or bs_price

                        # Calculate P&L probability
                        pl_stats = calculator.calculate_positive_pl_probability(
                            option_type=row["option_type"],
                            strike=row["strike"],
                            current_price=underlying_price,
                            time_to_expiry=time_to_expiry,
                            volatility=volatility,
                            risk_free_rate=risk_free_rate,
                            option_purchase_price=market_price,
                            transaction_cost=transaction_cost,
                        )

                        # Simplified enhanced row with only key metrics
                        enhanced_row = {
                            "Symbol": row["symbol"],
                            "Type": row["option_type"],
                            "Strike": row["strike"],
                            "Market Price": market_price,
                            "BS Price": bs_price,
                            "Days to Expiry": time_to_expiry,
                            "Volatility": volatility,
                            "P&L Probability": pl_stats['positive_pl_probability'],
                            "Breakeven": pl_stats['breakeven_price'],
                            "Volume": row.get("volume", 0),
                            "Bid": row.get("bid", 0),
                            "Ask": row.get("ask", 0),
                        }
                        enhanced_data.append(enhanced_row)

                    except Exception as e:
                        logging.warning(
                            f"Error processing option {row.get('symbol', 'unknown')}: {e}",
                            exc_info=True,
                        )
                        st.warning(
                            f"Error processing option {row.get('symbol', 'unknown')}: {e}"
                        )
                        continue

                if enhanced_data:
                    enhanced_df = pd.DataFrame(enhanced_data)

                    # Display interactive table
                    st.subheader("📋 Options Analysis Table")

                    # Filter controls
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        option_type_filter = st.selectbox(
                            "Filter by Type:", ["All", "C", "P"]
                        )
                    with col2:
                        min_probability = st.slider(
                            "Min P&L Probability:", 0.0, 1.0, 0.0, 0.05
                        )
                    with col3:
                        min_days = st.slider(
                            "Min Days to Expiry:", 0, 365, 0
                        )

                    # Apply filters
                    filtered_df = enhanced_df.copy()
                    if option_type_filter != "All":
                        filtered_df = filtered_df[
                            filtered_df["Type"] == option_type_filter
                        ]

                    # Filter by probability
                    filtered_df = filtered_df[
                        filtered_df["P&L Probability"] >= min_probability
                    ]
                    filtered_df = filtered_df[
                        filtered_df["Days to Expiry"] >= min_days
                    ]

                    # Create styled dataframe with colors
                    def style_dataframe(df):
                        """Apply color styling to highlight best contracts"""
                        styled_df = df.style
                        
                        # Color-code P&L Probability column
                        styled_df = styled_df.background_gradient(
                            subset=['P&L Probability'],
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=1
                        )
                        
                        # Format columns
                        styled_df = styled_df.format({
                            'Market Price': '${:,.2f}',
                            'BS Price': '${:,.2f}',
                            'P&L Probability': '{:.1%}',
                            'Breakeven': '${:,.2f}',
                            'Volatility': '{:.1%}',
                            'Strike': '${:,.2f}',
                            'Bid': '${:,.2f}',
                            'Ask': '${:,.2f}'
                        })
                        
                        return styled_df

                    # Display the styled dataframe
                    if len(filtered_df) > 0:
                        st.dataframe(
                            style_dataframe(filtered_df),
                            use_container_width=True,
                            height=400,
                        )

                        # Charts
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("P&L Probability by Strike")
                        
                            fig_prob = px.bar(
                                filtered_df,
                                x="Strike",
                                y="P&L Probability",
                                color="Type",
                                title="Positive P&L Probability by Strike Price",
                            )
                            # Use update_layout instead of update_yaxis
                            fig_prob.update_layout(
                                yaxis=dict(tickformat=".1%")
                            )
                            st.plotly_chart(
                                fig_prob, use_container_width=True
                            )

                        with col2:
                            st.subheader("Contract Volume Distribution")
                            fig_vol = px.histogram(
                                filtered_df,
                                x="Volume",
                                nbins=20,
                                title="Volume Distribution",
                            )
                            st.plotly_chart(fig_vol, use_container_width=True)

                    else:
                        st.warning("No contracts match your filter criteria")
                else:
                    st.warning("No valid options data to display")
            else:
                st.warning("No options data found in the database")
        else:
            st.info(
                f"No options data found for {ticker_input}. Please request data using the button above."
            )


def show_black_scholes_heatmap():
    """Show Black-Scholes heatmap analysis """
    st.header("🔥 Black-Scholes Options")

    # Sidebar parameters 
    with st.sidebar:
        st.header("Option Parameters")
        Underlying_price = st.number_input("Spot Price", value=100)
        trade_type = st.segmented_control(
            "Contract type", ["Call", "Put"], default="Call"
        )
        SelectedStrike = st.number_input("Strike/Exercise Price", value=80)
        days_to_maturity = st.number_input(
            "Time to Maturity (days)", value=365
        )
        Risk_Free_Rate = st.number_input(
            "Risk-Free Interest Rate", value=0.1
        )
        volatility = st.number_input("Annualized Volatility", value=0.2)

        st.subheader("P&L Parameters")
        option_purchase_price = st.number_input("Option's Price", value=0.0)
        transaction_cost = st.number_input(
            "Opening/Closing Cost", value=0.0
        )

        st.subheader("Heatmap Parameters")
        min_spot_price = st.number_input("Min Spot price", value=50)
        max_spot_price = st.number_input("Max Spot price", value=110)
        min_vol = st.slider("Min Volatility", 0.01, 1.00, 0.1)
        max_vol = st.slider("Max Volatility", 0.01, 1.00, 1.00)
        grid_size = st.slider("Grid size (nxn)", 5, 20, 10)

    # Variables
    SpotPrices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
    Volatilities_space = np.linspace(min_vol, max_vol, grid_size)

    # Calculate option prices
    call_price = BlackScholes(
        Risk_Free_Rate,
        Underlying_price,
        SelectedStrike,
        days_to_maturity / 365,
        volatility,
    )
    put_price = BlackScholes(
        Risk_Free_Rate,
        Underlying_price,
        SelectedStrike,
        days_to_maturity / 365,
        volatility,
        "P",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Call Value", f"${call_price:.3f}")
    with col2:
        st.metric("Put Value", f"${put_price:.3f}")

    # Create tabs (preserved from original)
    tab1, tab2, tab3 = st.tabs(
        [
            "Option's Fair Value Heatmap",
            "Option's P&L Heatmap",
            "Expected Underlying Distribution",
        ]
    )

    # Calculate matrices
    output_matrix_C = HeatMapMatrix(
        SpotPrices_space,
        Volatilities_space,
        SelectedStrike,
        Risk_Free_Rate,
        days_to_maturity,
    )
    output_matrix_P = HeatMapMatrix(
        SpotPrices_space,
        Volatilities_space,
        SelectedStrike,
        Risk_Free_Rate,
        days_to_maturity,
        type="P",
    )

    with tab1:
        st.write(
            "Explore different contract's values given variations in Spot Prices and Annualized Volatilities"
        )

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        sns.heatmap(
            output_matrix_C.T,
            annot=True,
            fmt=".1f",
            xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
            yticklabels=[str(round(i, 2)) for i in Volatilities_space],
            ax=axs[0],
            cbar_kws={"label": "Call Value"},
        )
        axs[0].set_title("Call Heatmap", fontsize=16)
        axs[0].set_xlabel("Spot Price")
        axs[0].set_ylabel("Annualized Volatility")

        sns.heatmap(
            output_matrix_P.T,
            annot=True,
            fmt=".1f",
            xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
            yticklabels=[str(round(i, 2)) for i in Volatilities_space],
            ax=axs[1],
            cbar_kws={"label": "Put Value"},
        )
        axs[1].set_title("Put Heatmap", fontsize=16)
        axs[1].set_xlabel("Spot Price")
        axs[1].set_ylabel("Annualized Volatility")

        st.pyplot(fig)

    with tab2:
        st.write(
            "Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility"
        )

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))

        call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
        put_PL = output_matrix_P.T - option_purchase_price - 2 * transaction_cost
        PL_options = [call_PL, put_PL]
        selection = 0 if trade_type == "Call" else 1

        cal_contract_prices = [call_price, put_price]
        specific_contract_pl = (
            cal_contract_prices[selection]
            - option_purchase_price
            - 2 * transaction_cost
        )

        st.metric(
            "Expected P&L (Current Parameters)", f"${specific_contract_pl:.2f}"
        )

        mapping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)
        sns.heatmap(
            PL_options[selection],
            annot=True,
            fmt=".1f",
            xticklabels=[str(round(i, 2)) for i in SpotPrices_space],
            yticklabels=[str(round(i, 2)) for i in Volatilities_space],
            ax=axs,
            cmap=mapping_color,
            center=0,
        )
        axs.set_title(f"{trade_type} Expected P&L", fontsize=16)
        axs.set_xlabel("Spot Price")
        axs.set_ylabel("Annualized Volatility")

        st.pyplot(fig)

    with tab3:
        st.write(
            "Calculate the expected distribution of the underlying asset price, option premium and P&L from trading"
        )

        # Simulation parameters (preserved from original)
        col1, col2, col3 = st.columns(3)
        with col1:
            NS = st.slider("Number of simulations", 100, 10000, 1000, 10)
        with col2:
            s_selection = st.radio(
                "Select time interval", ["Days", "Hours", "Minutes"], horizontal=True
            )
        with col3:
            timeshot = st.slider(
                "Select chart's timestamp (days/year)",
                0.0,
                days_to_maturity / 365,
                days_to_maturity / 365,
            )

        # Simulation logic (preserved from original)
        if s_selection == "Days":
            step = days_to_maturity
        elif s_selection == "Hours":
            step = days_to_maturity * 24
        elif s_selection == "Minutes":
            step = days_to_maturity * 24 * 60

        @st.cache_data
        def simulate(NS, days_to_maturity, s, volatility, Risk_Free_Rate):
            dt = (days_to_maturity / 365) / s
            Z = np.random.normal(0, np.sqrt(dt), (s, NS))
            paths = np.vstack(
                [
                    np.ones(NS),
                    np.exp(
                        (Risk_Free_Rate - 0.5 * volatility**2) * dt
                        + volatility * Z
                    ),
                ]
            ).cumprod(axis=0)
            return paths

        simulation_paths = Underlying_price * simulate(
            NS, days_to_maturity, step, volatility, Risk_Free_Rate
        )

        # Display results (preserved from original logic)
        dynamic_index = -int(
            step - timeshot * 365 * (step / days_to_maturity) + 1
        )

        if trade_type == "Call":
            option_prices = np.maximum(
                simulation_paths[dynamic_index, :] - SelectedStrike, 0
            )
        else:
            option_prices = np.maximum(
                SelectedStrike - simulation_paths[dynamic_index, :], 0
            )

        pl_results = (
            option_prices - option_purchase_price - 2 * transaction_cost
        )

        otm_probability = round(sum(option_prices == 0) / len(option_prices), 2)
        itm_probability = round(1 - otm_probability, 2)
        positive_pl_proba = round(sum(pl_results > 0) / len(pl_results), 2)

        col1, col2, col3 = st.columns(3)
        col1.metric("ITM Probability", f"{itm_probability:.1%}")
        col2.metric("OTM Probability", f"{otm_probability:.1%}")
        col3.metric("Positive P&L Probability", f"{positive_pl_proba:.1%}")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig_price = px.histogram(
                simulation_paths[dynamic_index, :],
                nbins=50,
                title=f"Expected Price Distribution at Day {int(timeshot * 365)}",
            )
            fig_price.add_vline(
                x=SelectedStrike, line_color="red", annotation_text="Strike Price"
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with col2:
            fig_pl = px.histogram(
                pl_results,
                nbins=50,
                title=f"Expected P&L Distribution at Day {int(timeshot * 365)}",
            )
            fig_pl.add_vline(
                x=0, line_color="red", annotation_text="Breakeven"
            )
            st.plotly_chart(fig_pl, use_container_width=True)


def show_user_preferences():
    """Show user preferences and settings"""
    st.header("⚙️ User Preferences")

    if not st.session_state.user_session:
        st.error("Please login first")
        return

    preferences = st.session_state.user_session.preferences

    with st.form("preferences_form"):
        st.subheader("Trading Preferences")

        # Default risk parameters
        risk_free_rate = st.number_input(
            "Default Risk-Free Rate",
            value=preferences.get("risk_free_rate", 0.05),
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
        )

        default_volatility = st.number_input(
            "Default Volatility",
            value=preferences.get("default_volatility", 0.25),
            min_value=0.0,
            max_value=5.0,
            format="%.4f",
        )

        # Real-time data preferences
        real_time_enabled = st.checkbox(
            "Enable Real-Time Data Updates",
            value=preferences.get("real_time_enabled", False),
        )

        # Watchlist symbols
        st.subheader("Watchlist")
        current_symbols = preferences.get("symbols", [])
        symbols_text = st.text_area(
            "Symbols (one per line)",
            value="\n".join(current_symbols),
            help="Enter ticker symbols you want to monitor, one per line",
        )

        # Model preferences
        st.subheader("Model Preferences")
        available_models = ["black_scholes", "monte_carlo", "neural_network"]
        selected_models = st.multiselect(
            "Enabled Models",
            available_models,
            default=preferences.get("models", ["black_scholes"]),
        )

        if st.form_submit_button("💾 Save Preferences"):
            symbols_list = [
                s.strip().upper() for s in symbols_text.split("\n") if s.strip()
            ]

            updated_preferences = {
                "risk_free_rate": risk_free_rate,
                "default_volatility": default_volatility,
                "real_time_enabled": real_time_enabled,
                "symbols": symbols_list,
                "models": selected_models,
            }

            st.session_state.auth_manager.save_user_preferences(
                st.session_state.user_session.user_id, updated_preferences
            )


# Real-time data fragment - SIMPLIFIED TO AVOID EVENT LOOP ISSUES
@st.fragment(run_every=10.0)  # Update every 10 seconds to reduce load
def real_time_data_fragment():
    """Fragment for real-time data updates - Simplified to avoid event loop issues"""
    if not st.session_state.get(
        "user_session"
    ) or not st.session_state.user_session.preferences.get(
        "real_time_enabled", False
    ):
        return

    symbols = st.session_state.user_session.preferences.get("symbols", [])
    if not symbols:
        return

    logging.info("Real-time fragment updating...")
    st.subheader("📊 Live Data Feed")

    for symbol in symbols[:3]:  # Limit to first 3 symbols
        try:
            # Get latest model results using synchronous calls only
            model_results = st.session_state.dashboard_comm.get_model_results(
                symbol
            )

            if model_results:
                latest_result = model_results[0]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        f"{symbol} - Model",
                        latest_result.get("model_name", "N/A"),
                    )
                with col2:
                    result_value = latest_result.get("result", 0)
                    try:
                        formatted_result = f"${float(result_value):.2f}"
                    except (ValueError, TypeError):
                        formatted_result = str(result_value)
                    st.metric("Result", formatted_result)
                with col3:
                    timestamp = pd.to_datetime(
                        latest_result.get("timestamp", datetime.now())
                    )
                    st.metric("Last Update", timestamp.strftime("%H:%M:%S"))
        except Exception as e:
            logging.warning(
                f"Error in real-time fragment for {symbol}: {e}",
                exc_info=True,
            )
            # Silently handle errors in real-time updates


# Main Application
def main():
    """Main application function"""
    logging.info("Application starting...")

    # Initialize managers
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthManager()

    if "dashboard_comm" not in st.session_state:
        st.session_state.dashboard_comm = DashboardCommunicator(
            st.session_state.auth_manager
        )

    # Check authentication
    if (
        not st.session_state.user_session
        or not st.session_state.user_session.is_authenticated
    ):
        st.session_state.auth_manager.show_auth_page()
        return

    # Show dashboard header
    show_dashboard_header()

    # Real-time data fragment (only if enabled)
    if st.session_state.user_session.preferences.get(
        "real_time_enabled", False
    ):
        real_time_data_fragment()

    # Navigation
    st.sidebar.title("📱 Navigation")

    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "📈 Options Analysis",
            "🔥 Black-Scholes",
            "🔑 API Keys Setup",
            "⚙️ Preferences",
        ],
    )

    # Show selected page
    if page == "🏠 Home":
        st.header("🏠 Welcome to Your Financial Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Quick Stats")
            symbols = st.session_state.user_session.preferences.get(
                "symbols", []
            )
            st.metric("Watchlist Symbols", len(symbols))
            st.metric(
                "Real-time Updates",
                "On"
                if st.session_state.user_session.preferences.get(
                    "real_time_enabled"
                )
                else "Off",
            )

        with col2:
            st.subheader("🔧 Your Settings")
            risk_free_rate = st.session_state.user_session.preferences.get(
                "risk_free_rate", 0.05
            )
            st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")
            volatility = st.session_state.user_session.preferences.get(
                "default_volatility", 0.25
            )
            st.metric("Default Volatility", f"{volatility:.2%}")

        with col3:
            st.subheader("🚀 Quick Actions")
            if st.button("📊 View Options Analysis", use_container_width=True):
                st.session_state["nav_page"] = "📈 Options Analysis"
                

            if st.button("🔥 Open Heatmap", use_container_width=True):
                st.session_state["nav_page"] = "🔥 Black-Scholes"
                

        # Recent activity
        if symbols:
            st.subheader("📋 Your Watchlist")
            for symbol in symbols:
                with st.expander(f"📈 {symbol}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            f"Request {symbol} Data", key=f"req_{symbol}"
                        ):
                            st.session_state.dashboard_comm.request_ticker_data(
                                symbol
                            )
                    with col2:
                        options_count = len(
                            st.session_state.dashboard_comm.get_options_data(
                                symbol
                            )
                        )
                        st.write(f"Options available: {options_count}")

    elif page == "📈 Options Analysis":
        show_options_analysis()

    elif page == "🔥 Black-Scholes Heatmap":
        show_black_scholes_heatmap()

    elif page == "🔑 API Keys Setup":
        show_api_keys_setup()

    elif page == "⚙️ Preferences":
        show_user_preferences()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Tips:**")
    st.sidebar.markdown("- Configure API keys for live data")
    st.sidebar.markdown("- Enable real-time updates in preferences")
    st.sidebar.markdown("- Add symbols to your watchlist")



if __name__ == "__main__":
    main()
