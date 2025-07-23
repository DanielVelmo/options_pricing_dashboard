# 📈 Autonomous Options Trading Dashboard & Publisher

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](#license) [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]() [![Streamlit](https://img.shields.io/badge/Streamlit-%3E%3D0.75-blue.svg)]() [![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-blue.svg)]() [![Ably](https://img.shields.io/badge/Ably-Realtime-blue.svg)]()

An end-to-end solution for autonomous options trading on Binance Options API—fetching live option chains, computing theoretical prices (Black-Scholes & beyond), storing & broadcasting data in real time, and visualizing insights through a rich Streamlit dashboard.

---

## 📋 Table of Contents

- [🚀 Features](#-features)  
- [🖥️ Architecture](#️-architecture)  
- [📂 Directory Structure](#-directory-structure)  
- [⚙️ Installation & Configuration](#️-installation--configuration)  
- [▶️ Usage](#️-usage)  
- [🔬 Theoretical Models](#-theoretical-models)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## 🚀 Features

- **Data Publisher**  
  - Polls user requests from Supabase  
  - Fetches live option chains via Binance Options API  
  - Calculates theoretical option prices (Black-Scholes)  
  - Stores market, options, and model data in Supabase  
  - Broadcasts real-time updates via Supabase Realtime & Ably  

- **Streamlit Dashboard**  
  - Secure Supabase-powered authentication  
  - Request new tickers & manage your watchlist  
  - Interactive Options Chain table with P&L probability  
  - Heatmaps: fair value & expected P&L surfaces  
  - Simulations for price & P&L distributions  

- **Extensible Model Engine**  
  - Batch Black-Scholes processing  
  - Plug-and-play support for additional models (e.g., Monte Carlo, neural nets)  
  - Confidence scoring & parameter logging  

---

## 🖥️ Architecture

### Publisher Workflow

```mermaid
flowchart TB
  subgraph PublisherComponents [Publisher Components]
    direction LR
    DBM[(DatabaseManager)]
    BOP[(BinanceOptionsProvider)]
    ME[(ModelEngine)]
    CM[(CommunicationManager)]
  end

  DBM --> BOP
  BOP --> ME
  ME --> CM
  CM --> SB1[Supabase Realtime]
  CM --> AB1[AblyRealtime]

  style DBM fill:#1F78B4,stroke:#333,stroke-width:2px
  style BOP fill:#1F78B4,stroke:#333,stroke-width:2px
  style ME fill:#1F78B4,stroke:#333,stroke-width:2px
  style CM fill:#1F78B4,stroke:#333,stroke-width:2px
  style SB1 fill:#33A02C,stroke:#333,stroke-width:1px
  style AB1 fill:#FF7F00,stroke:#333,stroke-width:1px
```

### Dashboard Workflow

```mermaid
flowchart TB
  subgraph DashboardComponents [Dashboard Components]
    direction LR
    AM[(AuthManager)]
    DC[(Dashboard)]
    UI[(Streamlit UI)]
  end

  AM --> DC
  UI --> DC
  DC --> SB2[Supabase Realtime]
  DC --> AB2[AblyRealtime]

  style AM  fill:#E31A1C,stroke:#333,stroke-width:2px
  style DC  fill:#E31A1C,stroke:#333,stroke-width:2px
  style UI  fill:#E31A1C,stroke:#333,stroke-width:2px
  style SB2 fill:#33A02C,stroke:#333,stroke-width:1px
  style AB2 fill:#FF7F00,stroke:#333,stroke-width:1px
```

---

## 📂 Directory Structure

```
.
├── maindata.py         # 🎛️ FinancialDataPublisher (core)
├── computes.py         # ⚙️ Previous dashboard version 
├── dashboard_001.py    # 🖥️ Streamlit frontend
├── requirements.txt    # 📦 Python dependencies
├── LICENSE             # 📄 MIT License
└── README.md           # 📘 Project overview
```

---

## ⚙️ Installation & Configuration

1. **Clone the repo**  
   ```bash
   git clone https://github.com/DanielVelmo/options_trading_dashboard.git
   cd options_trading_dashboard
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (e.g., in `.env` or your shell):
   ```bash
   # Supabase
   export SUPABASE_URL="https://your.supabase.co"
   export SUPABASE_ANON_KEY="your_anon_key"
   export SUPABASE_SERVICE_KEY="your_service_role_key"

   # Ably (optional)
   export ABLY_API_KEY="your_ably_api_key"

   # Logging & timing
   export LOG_LEVEL="INFO"
   export UPDATE_INTERVAL="1.0"
   export REQUEST_CHECK_INTERVAL="5.0"
   ```

4. **(Optional) Streamlit secrets**  
   In `.streamlit/secrets.toml`:
   ```toml
   SUPABASE_URL = "https://your.supabase.co"
   SUPABASE_ANON_KEY = "your_anon_key"
   ABLY_API_KEY = "your_ably_api_key"
   ```

---

## ▶️ Usage

### 1. Start the Publisher
```bash
python maindata.py
```
It will:
- Poll new requests every 5 s  
- Fetch & store options data  
- Compute & store model results  
- Broadcast updates

### 2. Launch the Dashboard
```bash
streamlit run dashboard_001.py
```
- Sign up / Log in via Supabase Auth  
- Configure your API keys  
- Request new tickers & explore your personalized dashboard

---

## 🔬 Theoretical Models

### Black-Scholes Formula

The Black-Scholes call price \(C\) and put price \(P\) are given by

$$
d_1 = \frac{\ln\bigl(\tfrac{S}{K}\bigr) + \bigl(r + \tfrac{\sigma^2}{2}\bigr)T}{\sigma\sqrt{T}}, 
\quad
d_2 = d_1 - \sigma\sqrt{T},
$$

$$
C = S\Phi(d_1) - K e^{-rT}\Phi(d_2), 
\quad
P = K e^{-rT}\Phi(-d_2) - S\Phi(-d_1),
$$

where:  
$$
S = \text{Spot price}, \quad K = \text{Strike}, \quad r = \text{Risk Free Rate} , \quad \sigma = \text{Volatility} 
$$
$$
T  = \text{time to expiry}, \quad \Phi = \text{CDF of the standard normal distribution}
$$ 

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/amazing-model
   ```
3. Commit your changes  
4. Push & open a PR  
5. 🎉 We’ll review & merge!

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
