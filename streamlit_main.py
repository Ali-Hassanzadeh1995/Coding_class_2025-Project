import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from datetime import date, datetime, timedelta
from typing import List, Set, Tuple, Callable

# ==============================================================================
# ⚙️ 1. Global Constants for Finance and Streamlit Config
# ==============================================================================
ANNUALIZATION_FACTOR = 252  # Standard trading days for daily data
RISK_FREE_RATE = 0.045  # Placeholder for Annual Risk-Free Rate (e.g., 4.5%)
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


# Set Streamlit page configuration
st.set_page_config(
    page_title="Financial Asset & Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 📉 2. Markowitz Portfolio Optimization Functions
# ==============================================================================


def portfolio_performance(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame
) -> Tuple[float, float]:
    """Return portfolio expected return and volatility given weights."""
    w = np.array(weights)
    port_return = np.dot(w, mu)
    port_vol = np.sqrt(w.T @ cov @ w)
    return port_return, port_vol


def minimize_volatility(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame) -> float:
    """Objective: portfolio volatility (to minimize)."""
    return portfolio_performance(weights, mu, cov)[1]


def negative_sharpe(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float
) -> float:
    """Objective: negative Sharpe ratio (to minimize)."""
    ret, vol = portfolio_performance(weights, mu, cov)
    if vol == 0:
        return 1e6
    return -(ret - risk_free_rate) / vol


def solve_min_variance(
    mu: pd.Series, cov: pd.DataFrame, allow_short: bool = False
) -> Tuple[np.ndarray, Tuple[float, float], sco.OptimizeResult]:
    """Finds the weights for the minimum variance portfolio."""
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    res = sco.minimize(
        minimize_volatility,
        x0,
        args=(mu, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x, portfolio_performance(res.x, mu, cov), res


def solve_max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float, allow_short: bool = False
) -> Tuple[np.ndarray, Tuple[float, float], sco.OptimizeResult]:
    """Finds the weights for the maximum Sharpe ratio portfolio (Tangent portfolio)."""
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    res = sco.minimize(
        negative_sharpe,
        x0,
        args=(mu, cov, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x, portfolio_performance(res.x, mu, cov), res


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, points: int = 50, allow_short: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs points on the efficient frontier by minimizing volatility for target returns."""
    n = len(mu)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    results = []

    ret_min = min(mu)
    ret_max = max(mu)
    target_returns = np.linspace(ret_min, ret_max, points)

    for r_target in target_returns:
        x0 = np.repeat(1 / n, n)
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x, r_target=r_target: np.dot(x, mu) - r_target,
            },
        ]
        res = sco.minimize(
            minimize_volatility,
            x0,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            w = res.x
            ret, vol = portfolio_performance(w, mu, cov)
            results.append((ret, vol, w))

    rets = np.array([r for r, v, w in results])
    vols = np.array([v for r, v, w in results])
    weights = np.array([w for r, v, w in results])
    return rets, vols, weights


# ==============================================================================
# ⬇️ 3. Optimized Rolling Metric Function
# ==============================================================================


def calculate_rolling_metrics_optimized(
    df: pd.DataFrame, time_frame: int, time_step: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculates rolling statistics using efficient pandas vectorized operations."""
    if time_frame > len(df):
        raise ValueError("Time frame is larger than the total number of data points.")

    df_simple_returns = df.pct_change()
    df_log_returns = np.log(df / df.shift(1))

    df_rolling_simple_mean_full = df_simple_returns.rolling(window=time_frame).mean()
    df_rolling_simple_volatility_full = df_simple_returns.rolling(
        window=time_frame
    ).std()
    df_rolling_log_volatility_full = df_log_returns.rolling(window=time_frame).std()

    P_start = df.shift(time_frame - 1)
    df_rolling_abs_log_return_full = np.log(df / P_start)

    start_index = time_frame - 1

    df_abs_log_return_stepped = df_rolling_abs_log_return_full.iloc[
        start_index::time_step
    ].dropna(how="all")
    df_log_volatility_stepped = df_rolling_log_volatility_full.iloc[
        start_index::time_step
    ].dropna(how="all")
    df_simple_mean_stepped = df_rolling_simple_mean_full.iloc[
        start_index::time_step
    ].dropna(how="all")
    df_simple_volatility_stepped = df_rolling_simple_volatility_full.iloc[
        start_index::time_step
    ].dropna(how="all")

    return (
        df_abs_log_return_stepped,
        df_log_volatility_stepped,
        df_simple_mean_stepped,
        df_simple_volatility_stepped,
    )


# ==============================================================================
# 📈 4. Visualization Functions
# ==============================================================================


def plot_metrics(
    df: pd.DataFrame, title: str, time_frame: int, time_step: int
) -> plt.Figure:
    """Generates a Matplotlib figure for rolling metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(ax=ax)
    ax.set_title(
        f"{title}\n(Window Size: {time_frame} Periods, Step: {time_step} Periods)"
    )
    ax.set_xlabel("End Date of Window")
    ax.set_ylabel("Value")
    ax.legend(title="Ticker")
    ax.grid(True, linestyle="--", alpha=0.7)
    return fig


def plot_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    ef_rets: np.ndarray,
    ef_vols: np.ndarray,
    ret_minvar: float,
    vol_minvar: float,
    ret_tan: float,
    vol_tan: float,
) -> plt.Figure:
    """Plots the Efficient Frontier along with key portfolios and individual assets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot Efficient Frontier
    ax.plot(ef_vols, ef_rets, "b--", lw=2, label="Efficient Frontier")

    # 2. Plot Key Portfolios
    ax.scatter(
        vol_minvar,
        ret_minvar,
        marker="*",
        s=200,
        label="Minimum-Variance Portfolio",
        color="green",
        zorder=3,
    )
    ax.scatter(
        vol_tan,
        ret_tan,
        marker="*",
        s=200,
        label="Max-Sharpe (Tangent) Portfolio",
        color="red",
        zorder=3,
    )

    # 3. Plot Individual Assets
    indiv_vol = np.sqrt(np.diag(cov))
    ax.scatter(
        indiv_vol,
        mu,
        marker="o",
        s=50,
        label="Individual Assets",
        color="black",
        zorder=2,
    )

    for i, t in enumerate(mu.index):
        ax.annotate(t, (indiv_vol[i], mu[i]), xytext=(6, 0), textcoords="offset points")

    ax.set_title("Markowitz Efficient Frontier (Annualized)")
    ax.set_xlabel("Annualized Volatility ($\sigma_p$)")
    ax.set_ylabel("Annualized Expected Return ($E[R_p]$)")
    ax.legend()
    ax.grid(True)
    return fig


# ==============================================================================
# 💾 5. Caching and Data Processing
# ==============================================================================


@st.cache_data(show_spinner="Downloading and processing data...")
def download_and_calculate_returns(
    List_stocks: List[str], S_date: date, E_date: date, interval: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Downloads data and calculates full-period returns and Markowitz inputs."""

    try:
        data = yf.download(
            List_stocks,
            start=S_date,
            end=E_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            st.error("No data returned. Check symbols, dates, and interval limits.")
            return None, None, None, None, None

    except Exception as e:
        st.error(f"Error during data download: {e}")
        return None, None, None, None, None

    DF_Adj_Close = data["Adj Close"].copy()
    DF_Adj_Close.dropna(inplace=True)

    if DF_Adj_Close.empty or len(DF_Adj_Close) < 2:
        st.error("Insufficient valid data after cleaning. Aborting analysis.")
        return None, None, None, None, None

    # Full Period Returns
    DF_simple_return = DF_Adj_Close.pct_change().dropna()
    DF_log_return = np.log(DF_Adj_Close / DF_Adj_Close.shift(1)).dropna()

    # Markowitz Inputs (Annualized Simple Returns and Covariance)
    mu_annual = DF_simple_return.mean() * ANNUALIZATION_FACTOR
    cov_annual = DF_simple_return.cov() * ANNUALIZATION_FACTOR

    return DF_Adj_Close, DF_simple_return, DF_log_return, mu_annual, cov_annual


# ==============================================================================
# 🏁 6. Streamlit Main App Function
# ==============================================================================


def streamlit_app():
    st.title("Financial Asset & Portfolio Analysis App 📈")

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("1. Asset Selection")
        ticker_input = st.text_input(
            "Enter stock tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):",
            value=",".join(DEFAULT_TICKERS),
        )
        List_stocks = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        N_stocks = len(List_stocks)
        st.write(f"**Number of stocks selected:** {N_stocks}")

        st.header("2. Data Range & Interval")
        today = date.today()
        default_start_date = today - timedelta(days=365 * 3)

        S_date = st.date_input(
            "Start Date", value=default_start_date, max_value=today - timedelta(days=1)
        )
        E_date = st.date_input(
            "End Date",
            value=today,
            min_value=S_date + timedelta(days=1),
            max_value=today,
        )

        interval = st.selectbox(
            "Interval (yfinance format)",
            options=["1d", "1wk", "1mo"],
            index=0,
            help="1d=Daily, 1wk=Weekly, 1mo=Monthly",
        )

        st.header("3. Rolling Window Analysis")
        time_frame = st.number_input(
            "Window Size (periods)",
            min_value=2,
            value=20,
            step=1,
            help="Number of data points in the rolling window.",
        )
        time_step = st.number_input(
            "Step Size (periods)",
            min_value=1,
            value=5,
            step=1,
            help="How many periods to step the window forward.",
        )

    # --- Execution Button ---
    if st.button("Run Analysis", type="primary") and N_stocks > 0:

        DF_Adj_Close, DF_simple_return, DF_log_return, mu_annual, cov_annual = (
            download_and_calculate_returns(List_stocks, S_date, E_date, interval)
        )

        if DF_Adj_Close is None:
            return  # Exit if data download failed

        st.success(
            f"Successfully downloaded {len(DF_Adj_Close)} data points for {', '.join(List_stocks)}."
        )

        # ======================================================================
        # PART A: Full-Period Data and Markowitz Optimization
        # ======================================================================
        st.header("A. Full Period & Portfolio Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Adjusted Close Prices")
            st.dataframe(DF_Adj_Close.tail())

        with col2:
            st.subheader("Annualized Simple Mean Returns (μ)")
            st.dataframe(
                mu_annual.to_frame("Annualized Mean Return").style.format("{:.4f}")
            )
            st.info(f"Using Risk-Free Rate (R_f): {RISK_FREE_RATE:.2%}")

        if N_stocks > 1:
            st.markdown("---")
            st.subheader("Markowitz Portfolio Optimization (Long-Only)")

            # 1. Run Optimization
            w_minvar, (ret_minvar, vol_minvar), _ = solve_min_variance(
                mu_annual, cov_annual, allow_short=False
            )
            w_tan, (ret_tan, vol_tan), _ = solve_max_sharpe(
                mu_annual, cov_annual, RISK_FREE_RATE, allow_short=False
            )
            ef_rets, ef_vols, ef_weights = efficient_frontier(
                mu_annual, cov_annual, points=60, allow_short=False
            )

            # 2. Display Results
            sharpe_tan = (ret_tan - RISK_FREE_RATE) / vol_tan

            col_mv, col_tan, col_weights = st.columns(3)

            with col_mv:
                st.metric(
                    "Minimum-Variance Portfolio",
                    value=f"{ret_minvar:.2%}",
                    delta=f"Volatility: {vol_minvar:.2%}",
                )

            with col_tan:
                st.metric(
                    "Max-Sharpe Portfolio",
                    value=f"{ret_tan:.2%}",
                    delta=f"Sharpe Ratio: {sharpe_tan:.4f}",
                )

            with col_weights:
                df_weights = pd.DataFrame(
                    {"MinVar Weight": w_minvar, "Tangent Weight": w_tan},
                    index=mu_annual.index,
                )
                st.caption("**Optimal Weights Comparison**")
                st.dataframe(df_weights.style.format("{:.2%}"))

            # 3. Plot
            fig_ef = plot_efficient_frontier(
                mu_annual,
                cov_annual,
                ef_rets,
                ef_vols,
                ret_minvar,
                vol_minvar,
                ret_tan,
                vol_tan,
            )
            st.pyplot(fig_ef)

        # ======================================================================
        # PART B: Rolling Window Metrics
        # ======================================================================
        st.markdown("---")
        st.header("B. Rolling Window Metrics Analysis")

        if len(DF_Adj_Close) < time_frame:
            st.warning(
                f"Data length ({len(DF_Adj_Close)}) is less than the window size ({time_frame}). Skipping rolling analysis."
            )
            return

        # Calculate Rolling Metrics
        (
            df_abs_log_return_roll,
            df_log_volatility_roll,
            df_simple_mean_roll,
            df_simple_volatility_roll,
        ) = calculate_rolling_metrics_optimized(DF_Adj_Close, time_frame, time_step)

        # 1. Display DataFrames
        col_data_1, col_data_2 = st.columns(2)
        with col_data_1:
            st.subheader("Rolling Simple Mean Return")
            st.dataframe(df_simple_mean_roll.tail())
        with col_data_2:
            st.subheader("Rolling Simple Volatility")
            st.dataframe(df_simple_volatility_roll.tail())

        # 2. Display Plots
        st.subheader("Time Series Plots")
        st.pyplot(
            plot_metrics(
                df_simple_mean_roll, "Rolling Simple Mean Return", time_frame, time_step
            )
        )
        st.pyplot(
            plot_metrics(
                df_simple_volatility_roll,
                "Rolling Simple Volatility (Std Dev)",
                time_frame,
                time_step,
            )
        )
        st.pyplot(
            plot_metrics(
                df_abs_log_return_roll,
                "Rolling Absolute Logarithmic Return Over Period",
                time_frame,
                time_step,
            )
        )
        st.pyplot(
            plot_metrics(
                df_log_volatility_roll,
                "Rolling Log Volatility (Std Dev)",
                time_frame,
                time_step,
            )
        )


# Call the main function
if __name__ == "__main__":
    streamlit_app()
