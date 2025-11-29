# ==============================================================================
# 📚 1. Standard Library Imports
# ==============================================================================
import sys
import re
from datetime import date, datetime
from typing import List, Set, Tuple

# ==============================================================================
# 📚 2. Third-Party Library Imports
# ==============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st
import requests
from bs4 import BeautifulSoup

# ==============================================================================
# ⚙️ 3. Global Constants (Risk-Free Rate)
# ==============================================================================
RISK_FREE_RATE_PERCENT = 4.25
RISK_FREE_RATE_DECIMAL = 4.25 / 100.0


# ==============================================================================
# 🧩 4. Helper/Utility Function Integration
# ==============================================================================


# --- SP500_Symbol_checker Class Integration ---
class SP500_Symbol_checker:
    """A utility class for handling S&P 500 stock symbols (adapted for Streamlit)."""

    def __init__(self, Symbol, Number) -> None:
        self.Symbol = Symbol
        self.Number = Number

    def Symbols_check(self):
        """Check if the given symbol exists in the S&P 500 list."""
        try:
            df = pd.read_csv("Symbols.csv")
            if "Symbol" not in df.columns:
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")
            set_symbols = set(df["Symbol"])
        except FileNotFoundError:
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = set(df["Symbol"])
        except Exception:
            return False

        return self.Symbol.upper() in set_symbols

    def Symbols_random_gen(self):
        """Generate a random set of S&P 500 stock symbols."""
        try:
            df = pd.read_csv("Symbols.csv")
            if "Symbol" not in df.columns:
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")
            set_symbols = list(df["Symbol"])
        except FileNotFoundError:
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = list(df["Symbol"])
        except Exception:
            return set()

        if len(set_symbols) < self.Number:
            st.warning(
                f"Only {len(set_symbols)} symbols available. Generating {len(set_symbols)} random symbols."
            )
            self.Number = len(set_symbols)

        random_stocks = np.random.choice(set_symbols, self.Number, replace=False)
        return {str(i) for i in random_stocks}

    def Symbols_df_maker(self):
        """Scrape the S&P 500 list from Wikipedia and save it as Symbols.csv."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "StreamlitFinancialTool/1.0"}

        try:
            page = requests.get(url, headers=headers, timeout=10)
            page.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(
                f"Error fetching S&P 500 data from Wikipedia: {e}. Cannot validate/generate random symbols."
            )
            return

        soup = BeautifulSoup(page.text, "html.parser")
        tables = soup.find_all("table")

        target_table = None
        for table in tables:
            if re.search("symbol", str(table), re.IGNORECASE) and re.search(
                "security", str(table), re.IGNORECASE
            ):
                target_table = table
                break

        if target_table is None:
            st.error(
                "Could not locate the S&P 500 symbols table on the Wikipedia page."
            )
            return

        df_list = pd.read_html(str(target_table))
        if df_list and "Symbol" in df_list[0].columns:
            df = df_list[0]
            df["Symbol"] = df["Symbol"].astype(str).str.strip()
            df[["Symbol"]].to_csv("Symbols.csv", index=False)
        else:
            st.error("Scraped table does not contain a 'Symbol' column.")


# --- Date Checker Integration ---
@st.cache_data(ttl=3600)
def get_min_valid_date(stock_symbols: Set[str]) -> date:
    """Fetches the LATEST (maximum) of individual stocks' start dates."""
    min_dates = []

    # Initialize the symbol checker to ensure Symbols.csv is available
    SP500_Symbol_checker(None, 0).Symbols_df_maker()

    st.info("Checking minimum valid history date for all selected stocks...")
    progress_bar = st.progress(0)

    for i, symbol in enumerate(stock_symbols):
        try:
            symbol_temp = yf.Ticker(symbol)
            hist = symbol_temp.history(period="max", auto_adjust=True, timeout=10)

            if hist.empty:
                st.warning(f"Could not fetch data for {symbol}. Skipping...")
                continue

            start_date_ts = hist.index.min().to_pydatetime().date()
            min_dates.append(start_date_ts)
            progress_bar.progress((i + 1) / len(stock_symbols))

        except Exception as e:
            st.error(f"Error fetching date for {symbol}: {e}")
            continue

    progress_bar.empty()

    if not min_dates:
        raise ValueError("Could not find a valid start date for any of the stocks.")

    common_min_date = max(min_dates)
    st.success(f"Minimum common valid start date is: **{common_min_date}**")
    return common_min_date


# --- Interval Setter Integration ---
def set_interval(S_date_str: str, E_date_str: str):
    """Calculates the date difference and returns available intervals."""
    S_date = np.datetime64(S_date_str)
    E_date = np.datetime64(E_date_str)
    diff_days = (E_date - S_date).astype(int)

    if diff_days <= 7:
        valid_intervals = ["1m"]
    elif 7 < diff_days <= 60:
        valid_intervals = [
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
    else:  # diff_days > 60
        valid_intervals = ["1d", "5d", "1wk", "1mo", "3mo"]

    VALID_INTERVALS = {
        "1m": "1 minute (~98280 periods/yr)",
        "2m": "2 minute (~49140 periods/yr)",
        "5m": "5 minute (~19656 periods/yr)",
        "15m": "15 minute (~6552 periods/yr)",
        "30m": "30 minute (~3276 periods/yr)",
        "60m": "60 minute (~1638 periods/yr)",
        "90m": "90 minute (~1092 periods/yr)",
        "1d": "1 day (252 periods/yr)",
        "5d": "5 day (52 periods/yr)",
        "1wk": "1 week (52 periods/yr)",
        "1mo": "1 month (12 periods/yr)",
        "3mo": "3 month (4 periods/yr)",
    }

    filtered_intervals = {
        k: VALID_INTERVALS[k] for k in valid_intervals if k in VALID_INTERVALS
    }

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

    interval_options = list(filtered_intervals.keys())
    if not interval_options:
        st.error("No valid intervals available for the selected date range.")
        return None, None, None

    st.subheader("📅 Data Interval Selection")
    st.caption(f"Difference between Start and End Date: {diff_days} days.")

    interval = st.selectbox(
        "Select Data Interval",
        options=interval_options,
        format_func=lambda x: f"{x} ({filtered_intervals[x].split('(')[1].split(')')[0]})",
        key="interval_select",
    )

    coefficient = INTERVALS_coefficient.get(interval)
    return interval, coefficient, filtered_intervals


# ==============================================================================
# 🧮 5. Calculation Functions
# ==============================================================================


def calculate_rolling_metrics_optimized(
    df: pd.DataFrame, time_frame: int, time_step: int, coefficient: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_frame > len(df):
        raise ValueError("Time frame is larger than the total number of data points.")

    df_simple_returns = df.pct_change() * 100
    df_simple_returns = df_simple_returns.dropna()

    df_rolling_mean = df_simple_returns.rolling(window=time_frame).mean()
    df_rolling_vol = df_simple_returns.rolling(window=time_frame).std()

    Rf_per_period = (1 + RISK_FREE_RATE_PERCENT / 100) ** (1 / coefficient) - 1
    Rf_per_period_perc = Rf_per_period * 100

    df_rolling_sharpe = (
        (df_rolling_mean - Rf_per_period_perc) / df_rolling_vol * np.sqrt(coefficient)
    )

    start_index = time_frame - 1
    df_mean_stepped = df_rolling_mean.iloc[start_index::time_step].dropna(how="all")
    df_vol_stepped = df_rolling_vol.iloc[start_index::time_step].dropna(how="all")
    df_sharpe_stepped = df_rolling_sharpe.iloc[start_index::time_step].dropna(how="all")

    return (df_mean_stepped, df_vol_stepped, df_sharpe_stepped)


def calculate_sharpe_ratio(
    weights,
    mean_returns_decimal,
    cov_matrix_decimal,
    risk_free_rate_annual,
    coefficient,
):
    annual_return = np.sum(mean_returns_decimal * weights) * coefficient
    annual_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix_decimal, weights))
    ) * np.sqrt(coefficient)

    if annual_volatility == 0:
        return 0.0

    sharpe_ratio = (annual_return - risk_free_rate_annual) / annual_volatility
    return sharpe_ratio


def negative_sharpe_ratio(
    weights,
    mean_returns_decimal,
    cov_matrix_decimal,
    risk_free_rate_annual,
    coefficient,
):
    sharpe = calculate_sharpe_ratio(
        weights,
        mean_returns_decimal,
        cov_matrix_decimal,
        risk_free_rate_annual,
        coefficient,
    )
    return -sharpe


# ==============================================================================
# 📈 6. Visualization Function
# ==============================================================================


def plot_metrics_st(
    df: pd.DataFrame,
    title: str,
    is_rolling: bool,
    kind: str = "line",
    time_frame: int = None,
    time_step: int = None,
    value_label: str = None,
):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        full_title = title
        x_label = "Date"
        color = None

        if is_rolling:
            full_title += f" (Window Size: {time_frame}, Step: {time_step})"
            x_label = "End Date of Rolling Window"

        if kind == "bar":
            if "Volatility" in title:
                color = "red"
            elif "Sharpe" in title:
                color = "blue"
            elif "Mean" in title:
                color = "green"

            df.plot(
                kind="bar",
                color=color,
                alpha=0.8,
                edgecolor="black",
                legend=False,
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            x_label = "Symbols"
            ax.tick_params(axis="x", rotation=0)

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
        else:
            df.plot(ax=ax)
            x_label = "Date"

        y_label = value_label if value_label else "Value"

        ax.set_title(full_title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        if kind == "line":
            ax.legend(
                df.columns, title="Symbols", bbox_to_anchor=(1.05, 1), loc="upper left"
            )

        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"⚠️ Could not generate plot for '{title}'. Error: {e}")


# ==============================================================================
# 7. Data Acquisition and Filtering
# ==============================================================================


@st.cache_data(ttl="1h", show_spinner="Downloading stock data...")
def download_data(list_stocks: List[str], S_date: str, E_date: str, interval: str):
    try:
        data = yf.download(
            list_stocks,
            start=S_date,
            end=E_date,
            interval=interval,
            auto_adjust=False,
            timeout=15,
        )
        if data.empty:
            st.error("No data returned from Yahoo Finance.")
            return None
    except Exception as e:
        st.error(f"🛑 Error during data download: {e}")
        return None

    # Handle multi-index columns if strictly necessary, but yfinance usually returns 'Adj Close'
    # if single ticker, it might be a Series.
    if isinstance(data, pd.DataFrame) and "Adj Close" in data:
        DF_Adj_Close = data["Adj Close"].copy()
    else:
        # Fallback if structure is different (e.g. single ticker sometimes)
        DF_Adj_Close = data

    DF_Adj_Close.dropna(inplace=True)

    if DF_Adj_Close.empty or len(DF_Adj_Close) < 2:
        st.error("🛑 Insufficient valid data after cleaning. Aborting.")
        return None

    return DF_Adj_Close


# ==============================================================================
# 8. Streamlit App Structure
# ==============================================================================


def execute_analysis():
    st.set_page_config(
        page_title="Financial Analysis Toolkit",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚀 Financial Analysis Toolkit V1.0")

    # --- SIDEBAR: Configuration ---
    st.sidebar.header("⚙️ Data Configuration")

    # Stock Input
    st.sidebar.markdown("### Stock Selection")

    # Widget: N_stocks
    N_stocks_input = st.sidebar.number_input(
        "Number of Stocks", min_value=1, max_value=20, value=2, key="N_stocks"
    )

    # Widget: Mode Selection
    mode_selection_input = st.sidebar.radio(
        "Select Mode",
        options=["Individual Stock Analysis", "Portfolio Optimization"],
        key="mode_select",
    )

    if mode_selection_input == "Portfolio Optimization" and N_stocks_input < 2:
        st.sidebar.warning(
            "Portfolio optimization typically requires 2 or more stocks."
        )

    random_test = st.sidebar.checkbox(
        f"Generate {N_stocks_input} random S&P 500 symbols for testing",
        key="random_test",
    )

    List_stocks_local: List[str] = []

    if random_test:
        checker = SP500_Symbol_checker(None, N_stocks_input)
        Set_stocks = checker.Symbols_random_gen()
        List_stocks_local = list(Set_stocks)
        st.sidebar.markdown(f"**Random Symbols:** {', '.join(List_stocks_local)}")
    else:
        st.sidebar.markdown("Enter stock symbols (comma-separated):")
        symbols_input = st.sidebar.text_input(
            "Symbols (e.g., AAPL, GOOGL)", value="AAPL, GOOGL", key="symbols_input"
        )
        # Process input and validate
        input_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        Set_stocks = set()
        for symbol in input_list:
            if SP500_Symbol_checker(symbol, 0).Symbols_check():
                Set_stocks.add(symbol)
            else:
                st.sidebar.error(f"Symbol '{symbol}' is invalid or not in S&P 500.")

        List_stocks_local = list(Set_stocks)

    # Validation logic before running
    valid_n_stocks = len(List_stocks_local)
    if valid_n_stocks == 0:
        st.error("Please enter at least one valid stock symbol or choose random test.")
        return

    # Date Input
    st.sidebar.markdown("### Date Range")
    try:
        min_common_date = get_min_valid_date(Set_stocks)
    except ValueError as e:
        st.error(f"🛑 Error: {e}")
        return

    today_date = date.today()

    S_date_input = st.sidebar.date_input(
        "Start Date",
        value=max(min_common_date, today_date.replace(year=today_date.year - 1)),
        min_value=min_common_date,
        max_value=today_date,
        key="S_date",
    )
    E_date_input = st.sidebar.date_input(
        "End Date",
        value=today_date,
        min_value=S_date_input,
        max_value=today_date,
        key="E_date",
    )

    st.sidebar.markdown(f"**Selected Stocks:** {', '.join(List_stocks_local)}")
    st.sidebar.markdown(f"**Date Range:** {S_date_input} to {E_date_input}")

    # Interval Selection
    interval, coefficient, _ = set_interval(str(S_date_input), str(E_date_input))

    if not interval or not coefficient:
        return

    # --- ACTION BUTTON ---
    if st.button("Fetch Data and Run Analysis", key="run_analysis_button"):

        DF_Adj_Close = download_data(
            List_stocks_local, str(S_date_input), str(E_date_input), interval
        )

        if DF_Adj_Close is None:
            return

        # ✅ FIX: SAVE TO NEW, SEPARATE SESSION STATE KEYS
        # We do not overwrite widget keys like 'N_stocks'.
        st.session_state["analysis_submitted"] = True
        st.session_state["analysis_DF"] = DF_Adj_Close
        st.session_state["analysis_list_stocks"] = List_stocks_local
        st.session_state["analysis_interval"] = interval
        st.session_state["analysis_coefficient"] = coefficient
        st.session_state["analysis_mode"] = mode_selection_input
        st.session_state["analysis_N_stocks"] = valid_n_stocks
        st.session_state["analysis_S_date"] = str(S_date_input)
        st.session_state["analysis_E_date"] = str(E_date_input)

        st.balloons()

    # --- CHECK IF ANALYSIS IS SUBMITTED ---
    if not st.session_state.get("analysis_submitted", False):
        st.info(
            "Configure the data in the sidebar and click 'Fetch Data and Run Analysis' to begin."
        )
        return

    # ✅ FIX: RETRIEVE FROM SEPARATE KEYS
    DF_Adj_Close = st.session_state["analysis_DF"]
    List_stocks = st.session_state["analysis_list_stocks"]
    coefficient = st.session_state["analysis_coefficient"]
    N_stocks = st.session_state["analysis_N_stocks"]
    mode_selection = st.session_state["analysis_mode"]

    # --- DISPLAY LOGIC ---
    st.header("📊 Stock Price Data")
    st.markdown("Adjusted Close Price Time Series (Sample)")
    st.dataframe(DF_Adj_Close.head())

    plot_metrics_st(
        DF_Adj_Close,
        "Adjusted Close Price Time Series",
        is_rolling=False,
        kind="line",
        value_label="Price (Dollars)",
    )

    st.markdown("---")

    # ==============================================================================
    # MODE 1: Individual Stock Analysis
    # ==============================================================================
    if mode_selection == "Individual Stock Analysis":
        st.header("🔵 Individual Stock Analysis")

        DF_simple_return = DF_Adj_Close.pct_change() * 100
        DF_simple_return.dropna(inplace=True)

        tab1, tab2 = st.tabs(["Full-Period Metrics", "Rolling Metrics"])

        with tab1:
            st.subheader("Full Period Metrics")

            df_mean_full = DF_simple_return.mean().to_frame("Mean Return (%)")
            df_vol_full = DF_simple_return.std().to_frame("Volatility (%)")

            Rf_per_period = (1 + RISK_FREE_RATE_PERCENT / 100) ** (1 / coefficient) - 1
            Rf_per_period_perc = Rf_per_period * 100

            df_sharpe_full = (
                (df_mean_full["Mean Return (%)"] - Rf_per_period_perc)
                / df_vol_full["Volatility (%)"]
                * np.sqrt(coefficient)
            ).to_frame("Sharpe Ratio")

            df_results = pd.concat([df_mean_full, df_vol_full, df_sharpe_full], axis=1)
            st.dataframe(df_results.style.format("{:.4f}"))

            col1, col2 = st.columns(2)
            with col1:
                plot_metrics_st(
                    df_vol_full, "Full Period Volatility Comparison", False, "bar"
                )
            with col2:
                plot_metrics_st(
                    df_sharpe_full, "Full Period Sharpe Ratio Comparison", False, "bar"
                )

        with tab2:
            st.subheader("Rolling Metrics")

            max_window = len(DF_Adj_Close) - 1
            if max_window < 2:
                st.warning("Not enough data points for rolling calculations.")
            else:
                # Rolling inputs are local to this view, no need to freeze them
                time_frame = st.slider(
                    "Rolling Window Size (Periods)",
                    min_value=2,
                    max_value=max_window,
                    value=min(20, max_window),
                    step=1,
                )
                time_step = st.slider(
                    "Rolling Step Size (Periods)",
                    min_value=1,
                    max_value=time_frame,
                    value=min(5, time_frame),
                    step=1,
                )

                try:
                    with st.spinner("Calculating rolling metrics..."):
                        (df_mean_roll, df_vol_roll, df_sharpe_roll) = (
                            calculate_rolling_metrics_optimized(
                                DF_Adj_Close, time_frame, time_step, coefficient
                            )
                        )

                    st.success("Rolling metrics calculated.")

                    st.subheader("Rolling Simple Mean Return")
                    plot_metrics_st(
                        df_mean_roll,
                        "Rolling Simple Mean Return",
                        True,
                        "line",
                        time_frame,
                        time_step,
                        "Percent per period",
                    )

                    st.subheader("Rolling Volatility")
                    plot_metrics_st(
                        df_vol_roll,
                        "Rolling Volatility",
                        True,
                        "line",
                        time_frame,
                        time_step,
                        "Percent per period",
                    )

                    st.subheader("Rolling Sharpe Ratio")
                    plot_metrics_st(
                        df_sharpe_roll,
                        "Rolling Sharpe Ratio",
                        True,
                        "line",
                        time_frame,
                        time_step,
                    )

                except ValueError as e:
                    st.error(f"🛑 Error in rolling calculation: {e}")

    # ==============================================================================
    # MODE 2: Portfolio Analysis
    # ==============================================================================
    elif mode_selection == "Portfolio Optimization":
        st.header("🟣 Portfolio Optimization")

        if N_stocks < 2:
            st.warning("Portfolio analysis requires at least 2 stocks.")
        else:
            # 1. Weights Input
            st.subheader("1. Portfolio Weights & Initial Amount")

            col1, col2 = st.columns(2)
            with col1:
                total_amount = st.number_input(
                    "Total Amount to Invest (€/D/£)",
                    min_value=1.0,
                    value=1000.0,
                    step=100.0,
                    format="%.2f",
                )

            with col2:
                weight_type = st.radio(
                    "Weight Allocation",
                    options=["Equal Weight", "Custom Weights"],
                    index=0,
                    key="weight_type",
                )

            weights = np.zeros(N_stocks)
            if weight_type == "Custom Weights":
                st.markdown("Enter custom weights (must sum to 1.0):")
                custom_weights = []
                total_w = 0.0
                for i, symbol in enumerate(List_stocks):
                    w = st.number_input(
                        f"Weight of {symbol}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        key=f"w_{symbol}",
                    )
                    custom_weights.append(w)
                    total_w += w

                if not np.isclose(total_w, 1.0):
                    st.error(
                        f"Custom weights sum to {total_w:.4f}. They must sum to 1.0."
                    )
                    return
                weights = np.array(custom_weights)
            else:
                weights = np.array([1.0 / N_stocks] * N_stocks)
                st.markdown(
                    f"**Using Equal Weights**: {1.0/N_stocks:.4f} for each stock."
                )

            normal_w = weights

            # 2. Portfolio Value Calculation
            st.subheader("2. Portfolio Performance")

            DF_simple_return = DF_Adj_Close.pct_change() * 100
            DF_simple_return.dropna(inplace=True)

            with st.spinner("Calculating portfolio daily value..."):
                portfolio_daily_returns_perc: pd.Series = DF_simple_return.dot(normal_w)

                DF_Prt_value_daily = pd.DataFrame(
                    index=portfolio_daily_returns_perc.index,
                    columns=["The daily value of the portfolio"],
                )
                current_portfolio_value = total_amount
                for i in portfolio_daily_returns_perc.index:
                    current_portfolio_value = current_portfolio_value * (
                        1 + portfolio_daily_returns_perc.loc[i] / 100.0
                    )
                    DF_Prt_value_daily.loc[i] = current_portfolio_value

            st.dataframe(DF_Prt_value_daily.head())
            plot_metrics_st(
                DF_Prt_value_daily,
                "Portfolio Value Time Series",
                False,
                "line",
                value_label="Value (Currency)",
            )

            # 3. Metrics Calculation
            st.subheader("3. Current Portfolio Metrics (Annualized)")

            df_cov = DF_simple_return.cov()
            cov_matrix = df_cov.to_numpy()
            cov_matrix_decimal = cov_matrix / 10000.0  # Convert from %^2 to decimal^2

            volatility_annual_decimal = np.sqrt(
                np.dot(normal_w.T, np.dot(cov_matrix_decimal, normal_w))
            ) * np.sqrt(coefficient)
            volatility_annual_perc = volatility_annual_decimal * 100.0
            st.info(
                f"Current Annualized Portfolio Volatility: **{volatility_annual_perc:.4f}%**"
            )

            mean_returns_decimal = DF_simple_return.mean() / 100.0
            portfolio_return_annual_decimal = (
                np.sum(mean_returns_decimal * normal_w) * coefficient
            )

            sharpe_ratio_full = calculate_sharpe_ratio(
                normal_w,
                mean_returns_decimal,
                cov_matrix_decimal,
                RISK_FREE_RATE_DECIMAL,
                coefficient,
            )

            df_sharpe_full = pd.DataFrame(
                {"Portfolio": sharpe_ratio_full}, index=["Sharpe Ratio"]
            ).T
            st.success(f"Current Annualized Sharpe Ratio: **{sharpe_ratio_full:.4f}**")
            st.dataframe(df_sharpe_full.style.format("{:.4f}"))

            # 4. Optimization
            st.subheader("4. Markowitz Optimization (Maximize Sharpe Ratio)")

            if st.button("Run Optimization"):
                with st.spinner("Starting Markowitz Optimization..."):
                    num_assets = len(List_stocks)
                    # Bounds: Weights must be between 0 and 1
                    bounds = tuple((0, 1) for _ in range(num_assets))
                    # Constraint: Sum of weights must equal 1
                    constraints = {
                        "type": "eq",
                        "fun": lambda weights: np.sum(weights) - 1,
                    }
                    initial_weights = normal_w

                    optimal_results = minimize(
                        negative_sharpe_ratio,
                        initial_weights,
                        args=(
                            mean_returns_decimal,
                            cov_matrix_decimal,
                            RISK_FREE_RATE_DECIMAL,
                            coefficient,
                        ),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                    )

                if optimal_results.success:
                    st.success("✅ Optimization Successful!")
                    optimal_weights = optimal_results.x
                    optimal_sharpe = -optimal_results.fun

                    optimal_return_annual = (
                        np.sum(mean_returns_decimal * optimal_weights) * coefficient
                    )
                    optimal_volatility_annual = np.sqrt(
                        np.dot(
                            optimal_weights.T,
                            np.dot(cov_matrix_decimal, optimal_weights),
                        )
                    ) * np.sqrt(coefficient)

                    st.markdown(
                        f"**Optimal Annualized Sharpe Ratio**: **{optimal_sharpe:.4f}**"
                    )
                    st.markdown(
                        f"Optimal Annualized Return: **{optimal_return_annual*100:.4f}%**"
                    )
                    st.markdown(
                        f"Optimal Annualized Volatility: **{optimal_volatility_annual*100:.4f}%**"
                    )

                    st.markdown("\n**Optimal Weights (Decimal):**")

                    optimal_weights_df = pd.DataFrame(
                        {
                            "Symbol": List_stocks,
                            "Optimal Weight": optimal_weights,
                            "Percentage": optimal_weights * 100,
                        }
                    ).set_index("Symbol")

                    st.dataframe(
                        optimal_weights_df.style.format(
                            {"Optimal Weight": "{:.4f}", "Percentage": "{:.2f}%"}
                        )
                    )

                    # Plotting optimal weights
                    fig_opt, ax_opt = plt.subplots(figsize=(10, 5))
                    optimal_weights_df["Optimal Weight"].plot(
                        kind="bar", ax=ax_opt, color="purple", alpha=0.8
                    )
                    ax_opt.set_title("Optimal Portfolio Weights for Max Sharpe Ratio")
                    ax_opt.set_ylabel("Weight (Decimal)")
                    ax_opt.tick_params(axis="x", rotation=0)
                    st.pyplot(fig_opt)
                    plt.close(fig_opt)

                else:
                    st.error(
                        f"❌ Optimization Failed. Status: {optimal_results.message}"
                    )


if __name__ == "__main__":
    execute_analysis()
