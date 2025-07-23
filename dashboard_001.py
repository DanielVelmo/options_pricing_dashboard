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

            # Calculate option values at expirat