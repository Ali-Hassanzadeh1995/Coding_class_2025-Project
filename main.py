# ==============================================================================
# 📚 1. Standard Library Imports
# ==============================================================================
from datetime import date, datetime
import sys
from typing import List, Set, Tuple

# ==============================================================================
# 📚 2. Third-Party Library Imports
# ==============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from IPython.display import display

# ==============================================================================
# 📚 3. Helper Module Imports
# ==============================================================================
# Assuming these exist in your local environment
from input_number_symbols import get_integer_input, get_valid_symbols
from date_checker import get_min_valid_date, get_valid_date_input
from interval import set_interval

# ==============================================================================
# ⚙️ 4. Global Constants (Risk-Free Rate)
# ==============================================================================
# Using 4.25% Annual Risk-Free Rate as a default proxy
RISK_FREE_RATE_ANNUAL = 4.25


# ==============================================================================
# ⬇️ 5. Optimized Rolling Metric Function
# ==============================================================================


def calculate_rolling_metrics_optimized(
    df: pd.DataFrame, time_frame: int, time_step: int, coefficient: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates rolling Mean, Volatility, and Sharpe Ratio.
    """
    if time_frame > len(df):
        raise ValueError("Time frame is larger than the total number of data points.")

    # Calculate simple returns
    df_simple_returns = df.pct_change() * 100
    df_simple_returns = df_simple_returns.dropna()

    # 1. Calculate Rolling Mean & Volatility
    df_rolling_mean = df_simple_returns.rolling(window=time_frame).mean()
    df_rolling_vol = df_simple_returns.rolling(window=time_frame).std()

    # 2. Calculate Rolling Sharpe Ratio
    # Formula: (Rolling Mean - Daily Risk Free) / Rolling Volatility
    Rf_per_period = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / coefficient) - 1
    df_rolling_sharpe = (
        (df_rolling_mean - Rf_per_period) / df_rolling_vol * np.sqrt(coefficient)
    )

    # 3. Step the results to reduce data density for plotting
    start_index = time_frame - 1
    df_mean_stepped = df_rolling_mean.iloc[start_index::time_step].dropna(how="all")
    df_vol_stepped = df_rolling_vol.iloc[start_index::time_step].dropna(how="all")
    df_sharpe_stepped = df_rolling_sharpe.iloc[start_index::time_step].dropna(how="all")

    return (df_mean_stepped, df_vol_stepped, df_sharpe_stepped)


# ==============================================================================
# 📈 6. Visualization Function
# ==============================================================================


def plot_metrics(
    df: pd.DataFrame,
    title: str,
    is_rolling: bool,
    kind: str = "line",
    time_frame: int = None,
    time_step: int = None,
    value_label: str = None,
):
    """Generates and displays a plot (Line or Bar) for the given DataFrame."""
    try:
        # Create figure
        plt.figure(figsize=(12, 6))

        full_title = title
        x_label = "Date"

        if is_rolling:
            full_title += f"\n(Window Size: {time_frame}, Step: {time_step})"
            x_label = "End Date of Rolling Window"

        # Plot Logic
        if kind == "bar":
            # Bar chart for scalar comparisons (Full Period)
            # We use a distinct color (e.g., orange) for volatility if indicated in title, otherwise default
            if "Volatility" in title:
                color = "red"
            elif "Sharpe" in title:
                color = "blue"

            df.plot(
                kind="bar", figsize=(12, 6), color=color, alpha=0.8, edgecolor="black"
            )
            plt.axhline(0, color="black", linewidth=0.8)  # Add zero line
            x_label = "Symbols"
            plt.xticks(rotation=0)
        else:
            # Line chart for time series (Rolling)
            plt.plot(df)
            x_label = "Date"
        if value_label:
            y_label = value_label
        else:
            y_label = "Value"

        plt.title(full_title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        if kind == "line":
            plt.legend(
                df.columns, title="Symbols", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
        else:
            # For bar charts, the legend is often redundant if the x-axis labels are clear,
            # but we keep it for consistency or disable if single metric.
            pass

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not generate plot for '{title}'. Error: {e}")


# ==============================================================================
# 🏁 7. Main Execution Block
# ==============================================================================


def execute_analysis():
    # --- Input Collection ---
    N_stocks = get_integer_input("Please enter the number of stocks: ")
    Set_stocks: Set[str] = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        sys.exit()

    List_stocks: List[str] = list(Set_stocks)
    print(f"\n**Selected Stocks:** {List_stocks}\n")

    min_common_date = get_min_valid_date(Set_stocks)
    today_date = date.today()
    S_date = get_valid_date_input(
        f"Enter Start Date (YYYY-MM-DD, min: {min_common_date}): ",
        min_date=min_common_date,
        max_date=today_date,
    )
    S_date_dt = datetime.strptime(S_date, "%Y-%m-%d").date()
    E_date = get_valid_date_input(
        f"Enter End Date (YYYY-MM-DD, min: {S_date}): ",
        min_date=S_date_dt,
        max_date=today_date,
    )
    interval, coefficient = set_interval(S_date, E_date)
    print(f"\n**Data Range:** {S_date} to {E_date}, Interval: {interval}")

    # --- Data Download & Pre-processing ---
    print("\n⬇️ Downloading data from Yahoo Finance...")
    try:
        data = yf.download(
            List_stocks, start=S_date, end=E_date, interval=interval, auto_adjust=False
        )
        if data.empty:
            raise ValueError("No data returned.")
    except Exception as e:
        print(f"🛑 Error during data download: {e}")
        sys.exit()

    # Extract Adjusted Close
    DF_Adj_Close: pd.DataFrame = data["Adj Close"].copy()
    DF_Adj_Close.dropna(inplace=True)
    display(DF_Adj_Close)
    plot_metrics(
        DF_Adj_Close,
        "Adjust close price Time Series (Full Period)",
        is_rolling=False,
        kind="line",
        value_label="Dollars",
    )

    if DF_Adj_Close.empty or len(DF_Adj_Close) < 2:
        print("🛑 Insufficient valid data. Aborting.")
        sys.exit()

    # 1. Calculate Simple Returns
    DF_simple_return: pd.DataFrame = DF_Adj_Close.pct_change() * 100
    DF_simple_return = DF_simple_return.dropna()
    print("\n📊 Simple Returns (Sample):")
    display(DF_simple_return.head())

    # 2. Choose Analysis Mode
    print(
        "\n🌕🌔🌓🌒🌑 Choose Analysis Mode: Full Period (Y) or Rolling Time Frame (N)."
    )
    option = str(input("Do you want full-period results? (Y/N): "))

    if option.upper() == "Y":
        # === FULL PERIOD ANALYSIS ===

        # Calculate Mean and Volatility
        df_mean_full = DF_simple_return.mean().to_frame("Mean Return")
        df_vol_full = DF_simple_return.std().to_frame("Volatility")

        # Calculate Sharpe Ratio (Vectorized)
        Rf_per_period = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / coefficient) - 1
        df_sharpe_full = (
            (df_mean_full["Mean Return"] - Rf_per_period)
            / df_vol_full["Volatility"]
            * np.sqrt(coefficient)
        )
        df_sharpe_full = df_sharpe_full.to_frame("Sharpe Ratio")

        # Combine for Display
        df_results = pd.concat([df_mean_full, df_vol_full, df_sharpe_full], axis=1)

        print("\n📊 Full Period Metrics:")
        display(df_results)

        # Plot 1: Simple Return Time Series (Line)
        plot_metrics(
            DF_simple_return,
            "Simple Return Time Series (Full Period)",
            is_rolling=False,
            kind="line",
        )

        # Plot 2: Volatility Comparison (Bar - NEW)
        print("\n📊 Visualizing Volatility Comparison...")
        plot_metrics(
            df_vol_full,
            "Full Period Volatility Comparison",
            is_rolling=False,
            kind="bar",
        )

        # Plot 3: Sharpe Ratio Comparison (Bar)
        print("\n📊 Visualizing Sharpe Ratio Comparison...")
        plot_metrics(
            df_sharpe_full,
            "Full Period Sharpe Ratio Comparison",
            is_rolling=False,
            kind="bar",
        )

    else:
        # === ROLLING TIME FRAME ANALYSIS ===

        print(
            f"\nGiven your data has **{len(DF_Adj_Close)}** periods, enter a time_frame and a time_step."
        )
        time_frame = get_integer_input(
            "Enter time_frame (window size, e.g., 20 periods):"
        )
        time_step = get_integer_input(
            "Enter time_step (periods to step, e.g., 5 periods):"
        )

        try:
            (df_mean_roll, df_vol_roll, df_sharpe_roll) = (
                calculate_rolling_metrics_optimized(
                    DF_Adj_Close, time_frame, time_step, coefficient
                )
            )

            # Visualization
            print("\n💹 Generating plots for rolling metrics...")

            # Line Chart: Mean Return
            print("\n📈 Rolling Simple Mean Return:")
            display(df_mean_roll)

            plot_metrics(
                df_mean_roll,
                "Rolling Simple Mean Return",
                is_rolling=True,
                kind="line",
                time_frame=time_frame,
                time_step=time_step,
                value_label="Percent per period",
            )

            # Line Chart: Volatility
            print("\n📉 Rolling volatility:")
            display(df_vol_roll)
            plot_metrics(
                df_vol_roll,
                "Rolling Volatility",
                is_rolling=True,
                kind="line",
                time_frame=time_frame,
                time_step=time_step,
                value_label="Percent per period",
            )

            # Line Chart: Sharpe Ratio
            print("\n⚖️ Rolling Sharpe Ratio:")
            display(df_sharpe_roll)
            plot_metrics(
                df_sharpe_roll,
                "Rolling Sharpe Ratio",
                is_rolling=True,
                kind="line",
                time_frame=time_frame,
                time_step=time_step,
            )

        except ValueError as e:
            print(f"\n🛑 Error in rolling calculation: {e}")
            sys.exit()


if __name__ == "__main__":
    execute_analysis()
