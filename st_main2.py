import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from scipy.optimize import minimize
from typing import Tuple, List, Dict

# ==============================================================================
# 📝 STREAMLIT CONFIGURATION
# ==============================================================================
# st.set_page_config: Configures the page title, icon, and layout (Wide mode).
# This must be the very first Streamlit command used.
st.set_page_config(
    page_title="Financial Analysis Toolkit", page_icon="💸", layout="wide"
)

# ==============================================================================
# 📚 HELPER FUNCTIONS (Refactored from uploaded modules)
# ==============================================================================


# @st.cache_data: A decorator that caches the output of this function.
# If the app reruns, it won't scrape Wikipedia again unless the cache is cleared.
# This speeds up the app significantly.
@st.cache_data
def get_sp500_symbols() -> List[str]:
    """Scrapes S&P 500 symbols from Wikipedia (from SP500_Symbol_checker.py)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "StreamlitFinancialApp/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find table with 'Symbol' in headers
        tables = soup.find_all("table")
        target_table = None
        for t in tables:
            if "Symbol" in str(t) or "symbol" in str(t):
                target_table = t
                break

        if not target_table:
            return []

        # Extract symbols
        symbols = []
        rows = target_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all(["td", "th"])
            if cols:
                sym = cols[0].text.strip()
                symbols.append(sym)
        return symbols
    except Exception as e:
        st.error(f"Error scraping S&P 500: {e}")
        return []


# @st.cache_data: Caches the stock data download so changing a visualization
# setting doesn't trigger a re-download of the same data.
@st.cache_data
def download_stock_data(tickers: List[str], start_date, end_date, interval):
    """Downloads data using yfinance."""
    if not tickers:
        return pd.DataFrame()

    # st.spinner: Shows a temporary "Running..." spinner in the UI while this executes.
    with st.spinner(f"Downloading data for {len(tickers)} stocks..."):
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if "Adj Close" in data:
                return data["Adj Close"]
            elif isinstance(data, pd.DataFrame) and not data.empty:
                # Handle cases where yfinance returns a single level column if 1 stock
                if len(tickers) == 1:
                    return data["Adj Close"] if "Adj Close" in data else data
                return data
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Download error: {e}")
            return pd.DataFrame()


def determine_interval_coefficient(interval: str) -> int:
    """Returns the annualization coefficient based on the interval."""
    # From interval.py
    INTERVALS_coefficient = {
        "1m": 98280,
        "2m": 49140,
        "5m": 19656,
        "15m": 6552,
        "30m": 3276,
        "60m": 1638,
        "90m": 1092,
        "1d": 252,
        "5d": 52,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    return INTERVALS_coefficient.get(interval, 252)


def calculate_sharpe(weights, mean_returns, cov_matrix, rf_rate, coeff):
    """Portfolio Sharpe Ratio Calculation."""
    port_return = np.sum(mean_returns * weights) * coeff
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(coeff)
    return (port_return - rf_rate) / port_vol if port_vol != 0 else 0


def negative_sharpe(weights, mean_returns, cov_matrix, rf_rate, coeff):
    """Objective function for optimization."""
    return -calculate_sharpe(weights, mean_returns, cov_matrix, rf_rate, coeff)


# ==============================================================================
# 🖥️ UI LAYOUT & INPUTS
# ==============================================================================

# st.title: Main heading of the app.
st.title("💸 Financial Analysis Toolkit")
st.markdown(
    "Perform **Individual Stock Analysis** or **Portfolio Optimization** using real-time data."
)

# st.sidebar: This creates the sidebar on the left for controls.
st.sidebar.header("🍪 Configuration")

# 1. Mode Selection
# st.sidebar.radio: Radio buttons for mutually exclusive options.
mode = st.sidebar.radio(
    "Select Analysis Mode", ["Individual Stock Analysis", "Portfolio Optimization"]
)

# 2. Stock Selection
# Load S&P 500 symbols for easy selection
sp500_symbols = get_sp500_symbols()
default_symbols = ["AAPL", "MSFT", "GOOG"] if not sp500_symbols else sp500_symbols[:3]

# st.sidebar.multiselect: A dropdown that allows selecting multiple items.
selected_stocks = st.sidebar.multiselect(
    "Select Stocks (S&P 500)", options=sp500_symbols, default=default_symbols
)

# 3. Date Selection
st.sidebar.subheader("📅 Date Range")
today = date.today()
default_start = today - timedelta(days=365)

# st.sidebar.date_input: Date picker widgets.
start_date = st.sidebar.date_input("Start Date", default_start, max_value=today)
end_date = st.sidebar.date_input("End Date", today, max_value=today)

if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")
    st.stop()

# 4. Interval Selection
# Determine valid intervals based on logic from interval.py
duration_days = (end_date - start_date).days
valid_intervals = []

if duration_days <= 7:
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d"]
elif duration_days <= 60:
    valid_intervals = ["2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk"]
else:
    valid_intervals = ["1d", "5d", "1wk", "1mo", "3mo"]

# st.sidebar.selectbox: A standard dropdown menu.
interval = st.sidebar.selectbox(
    "Select Data Interval",
    valid_intervals,
)

# Risk Free Rate
rf_rate_percent = st.sidebar.number_input("Risk-Free Rate (%)", value=4.25, step=0.01)
rf_rate_decimal = rf_rate_percent / 100.0

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================

# Fetch Data
df_adj_close = download_stock_data(selected_stocks, start_date, end_date, interval)
coefficient = determine_interval_coefficient(interval)

if df_adj_close.empty:
    st.error(
        "No data found for the selected parameters. Please check symbols or dates."
    )
    st.stop()

# Display Raw Data (Optional expander)
# st.expander: Creates a collapsible container to hide/show details.
with st.expander("🔎 View Raw Data"):
    # FIX: Added unique key="raw_data_toggle"
    show_full_raw = st.checkbox("Show full data", value=False, key="raw_data_toggle")

    if show_full_raw:
        st.dataframe(df_adj_close)
    else:
        st.dataframe(df_adj_close.head())

# Calculate Simple Returns
df_simple_return = df_adj_close.pct_change() * 100
df_simple_return.dropna(inplace=True)

with st.expander("🔎 View Simple Return Data Frame"):
    # FIX: Added unique key="simple_ret_toggle"
    show_full_ret = st.checkbox("Show full data", value=False, key="simple_ret_toggle")

    if show_full_ret:
        st.dataframe(df_simple_return)
    else:
        st.dataframe(df_simple_return.head())
# ------------------------------------------------------------------------------
# MODE 1: INDIVIDUAL STOCK ANALYSIS
# ------------------------------------------------------------------------------
if mode == "Individual Stock Analysis":
    st.header("📚 Individual Stock Analysis")

    # st.checkbox: A boolean toggle.
    use_rolling = st.checkbox("Enable Rolling Window Analysis?")

    if not use_rolling:
        # --- Full Period Analysis ---
        st.subheader("📊 Full Period Metrics")

        # Calculate Metrics
        mean_ret = df_simple_return.mean()
        volatility = df_simple_return.std()

        # Risk Free per period
        rf_per_period = (1 + rf_rate_decimal) ** (1 / coefficient) - 1
        rf_per_period_perc = rf_per_period * 100

        sharpe = (mean_ret - rf_per_period_perc) / volatility * np.sqrt(coefficient)

        metrics_df = pd.DataFrame(
            {
                "Mean Return (%)": mean_ret,
                "Volatility (%)": volatility,
                "Sharpe Ratio": sharpe,
            }
        )

        # st.dataframe: Displays interactive tables.
        # use_container_width=True makes it fit the layout.
        st.dataframe(metrics_df, use_container_width=True)

        # Visualizations using Streamlit Native Charts
        st.subheader("📈 Visualizations")

        # st.line_chart: Native interactive line chart.
        # Accepts a DataFrame where index=X-axis and columns=Lines.
        st.caption("Price History")
        st.line_chart(df_adj_close)

        # st.bar_chart: Native interactive bar chart.
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Volatility Comparison")
            st.bar_chart(metrics_df["Volatility (%)"], color="#FF4B4B")  # Red color
        with col2:
            st.caption("Sharpe Ratio Comparison")
            st.bar_chart(metrics_df["Sharpe Ratio"], color="#0068C9")  # Blue color

    else:
        # --- Rolling Analysis ---
        st.subheader("🕛🕧🕐🕜🕑🕝 Rolling Metrics")

        col1, col2 = st.columns(2)
        # st.number_input: Input widget for numbers with min/max validation.
        # Ensure default window is valid (at least 5, max 100 or data length)
        data_len = len(df_simple_return)
        default_window = max(5, min(data_len, 100))

        # st.number_input: Input widget for numbers with min/max validation.
        time_frame = col1.number_input("Window Size", min_value=5, value=default_window)

        # Ensure default step is integer and at least 1
        # We use // for integer division to ensure the widget remains in "integer mode"
        default_step = max(1, int(time_frame // 2))

        if time_frame > len(df_simple_return):
            st.error("Window size is larger than available data points.")
        else:
            # Calculate Rolling Metrics
            rolling_mean = df_simple_return.rolling(window=time_frame).mean()
            rolling_vol = df_simple_return.rolling(window=time_frame).std()

            rf_per_period = (1 + rf_rate_decimal) ** (1 / coefficient) - 1
            rf_per_period_perc = rf_per_period * 100

            rolling_sharpe = (
                (rolling_mean - rf_per_period_perc) / rolling_vol * np.sqrt(coefficient)
            )

            # Step logic
            rolling_mean = rolling_mean.iloc[time_frame - 1 :: time_step].dropna()
            rolling_vol = rolling_vol.iloc[time_frame - 1 :: time_step].dropna()
            rolling_sharpe = rolling_sharpe.iloc[time_frame - 1 :: time_step].dropna()

            # Display Charts
            st.caption("Rolling Mean Return")
            st.line_chart(rolling_mean)

            st.caption("Rolling Volatility")
            st.line_chart(rolling_vol)

            st.caption("Rolling Sharpe Ratio")
            st.line_chart(rolling_sharpe)


# ------------------------------------------------------------------------------
# MODE 2: PORTFOLIO OPTIMIZATION
# ------------------------------------------------------------------------------
elif mode == "Portfolio Optimization":
    st.header("📖 Portfolio Optimization")

    if len(selected_stocks) < 2:
        st.warning("⚠️ Portfolio optimization requires 2 or more stocks.")
        st.stop()

    investment_amount = st.number_input(
        "Investment Amount ($)", value=1000.0, step=100.0
    )

    # 1. Weights Input
    st.subheader("⚖️ Portfolio Weights")
    weight_mode = st.radio("Weight Distribution", ["Equal Weights", "Custom Weights"])

    weights = np.array(
        [1.0 / len(selected_stocks)] * len(selected_stocks)
    )  # Default equal

    if weight_mode == "Custom Weights":
        # Create columns dynamically for inputs
        cols = st.columns(len(selected_stocks))
        custom_weights = []
        for i, sym in enumerate(selected_stocks):
            # Place inputs in columns
            w = cols[i % 4].number_input(
                f"{sym} Weight",
                value=1.0 / len(selected_stocks),
                min_value=0.0,
                step=0.05,
            )
            custom_weights.append(w)

        # Normalize
        total_w = sum(custom_weights)
        if total_w == 0:
            st.error("Total weight cannot be zero.")
            st.stop()
        weights = np.array(custom_weights) / total_w
        st.info(f"Weights normalized to sum to 1.0: {weights}")

    # 2. Portfolio Performance
    # Calculate daily portfolio returns
    port_daily_ret = df_simple_return.dot(weights)

    # Calculate Value over time
    # Cumulative product of (1 + return/100)
    port_cum_ret = (1 + port_daily_ret / 100).cumprod()
    port_value = port_cum_ret * investment_amount

    st.subheader("💰 Portfolio Performance")
    st.caption("Portfolio Value Over Time")
    st.line_chart(port_value)

    # 3. Metrics
    mean_ret_dec = df_simple_return.mean() / 100.0
    cov_matrix_dec = df_simple_return.cov() / 10000.0

    # Annualized Metrics
    port_ann_ret = np.sum(mean_ret_dec * weights) * coefficient
    port_ann_vol = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix_dec, weights))
    ) * np.sqrt(coefficient)
    curr_sharpe = (port_ann_ret - rf_rate_decimal) / port_ann_vol

    # st.metric: Displays a bold metric with optional delta/change.
    m1, m2, m3 = st.columns(3)
    m1.metric("Annualized Return", f"{port_ann_ret*100:.2f}%")
    m2.metric("Annualized Volatility", f"{port_ann_vol*100:.2f}%")
    m3.metric("Sharpe Ratio", f"{curr_sharpe:.4f}")

    # 4. Optimization
    st.subheader("🧠 Markowitz Optimization (Max Sharpe)")
    if st.button("Run Optimization"):
        with st.spinner("Optimizing..."):
            bounds = tuple((0, 1) for _ in range(len(selected_stocks)))
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            init_w = np.array([1.0 / len(selected_stocks)] * len(selected_stocks))

            result = minimize(
                negative_sharpe,
                init_w,
                args=(mean_ret_dec, cov_matrix_dec, rf_rate_decimal, coefficient),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                opt_weights = result.x
                opt_sharpe = -result.fun

                opt_ret = np.sum(mean_ret_dec * opt_weights) * coefficient
                opt_vol = np.sqrt(
                    np.dot(opt_weights.T, np.dot(cov_matrix_dec, opt_weights))
                ) * np.sqrt(coefficient)

                st.success("Optimization Successful!")

                # Show Comparison
                col_a, col_b = st.columns(2)

                # Create DataFrame for bar chart comparison
                df_compare = pd.DataFrame(
                    {
                        "Stock": selected_stocks,
                        "Current Weights": weights,
                        "Optimal Weights": opt_weights,
                    }
                ).set_index("Stock")

                with col_a:
                    st.write("**Optimal Metrics**")
                    st.write(f"Sharpe Ratio: `{opt_sharpe:.4f}`")
                    st.write(f"Return: `{opt_ret*100:.2f}%`")
                    st.write(f"Volatility: `{opt_vol*100:.2f}%`")

                with col_b:
                    st.caption("Weight Allocation Comparison")
                    st.bar_chart(df_compare)
            else:
                st.error(f"Optimization failed: {result.message}")
