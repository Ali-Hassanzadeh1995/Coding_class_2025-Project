# ==============================================================================
# 📚 1. Standard Library Imports
# ==============================================================================
from datetime import date, datetime
import sys

# ==============================================================================
# 📚 2. Third-Party Library Imports
# ==============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Set, Tuple

# ==============================================================================
# 📚 3. Helper Module Imports (Assuming these exist and work correctly)
# ==============================================================================
from input_number_symbols import get_integer_input, get_valid_symbols
from date_checker import get_min_valid_date, get_valid_date_input
from interval import set_interval


# ==============================================================================
# ⬇️ 4. Data Acquisition and Full-Period Metrics Calculation
# ==============================================================================


def execute_analysis():
    # --- Input Collection ---
    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Get the set of stock symbols
    Set_stocks = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        # Use sys.exit() instead of exit() for cleaner script termination
        sys.exit()

    List_stocks: List[str] = list(Set_stocks)
    print(f"\n**Selected Stocks:** {List_stocks}\n")

    # Get and validate Start Date
    min_common_date = get_min_valid_date(Set_stocks)
    today_date = date.today()
    print(f"Today's date is: {today_date}")

    S_date = get_valid_date_input(
        f"Enter Start Date (YYYY-MM-DD, min: {min_common_date}): ",
        min_date=min_common_date,
        max_date=today_date,
    )

    # Get and validate End Date
    S_date_dt = datetime.strptime(S_date, "%Y-%m-%d").date()
    E_date = get_valid_date_input(
        f"Enter End Date (YYYY-MM-DD, min: {S_date}): ",
        min_date=S_date_dt,
        max_date=today_date,
    )

    # Get and validate Interval
    interval = set_interval(S_date, E_date)
    print(f"\n**Data Range:** {S_date} to {E_date}, Interval: {interval}")

    # --- Data Download ---
    print("\n⬇️ Downloading data from Yahoo Finance...")
    try:
        data = yf.download(
            List_stocks,
            start=S_date,
            end=E_date,
            interval=interval,
            auto_adjust=False,
        )
        if data.empty:
            raise ValueError(
                "No data returned. Check symbols, dates, and interval limits."
            )

    except Exception as e:
        print(f"🛑 Error during data download: {e}")
        sys.exit()

    # Extract Adjusted Close prices
    DF_Adj_Close: pd.DataFrame = data["Adj Close"].copy()

    # Drop rows where any stock has missing data (important for portfolio analysis)
    DF_Adj_Close.dropna(inplace=True)

    if DF_Adj_Close.empty or len(DF_Adj_Close) < 2:
        print("🛑 Insufficient valid data after cleaning. Aborting.")
        sys.exit()

    # --- Full Period Metrics ---
    DF_simple_return: pd.DataFrame = DF_Adj_Close.pct_change().dropna()
    DF_log_return: pd.DataFrame = np.log(DF_Adj_Close / DF_Adj_Close.shift(1)).dropna()

    print("\n--- Full Period Statistics ---")
    print(f"Total Periods Available: {len(DF_Adj_Close)}")
    print(f"First 5 Simple Returns:\n{DF_simple_return.head()}")

    DF_log_volatility = DF_log_return.std()
    DF_simple_mean = DF_simple_return.mean()
    DF_simple_volatility = DF_simple_return.std()

    print("\n**Full Period Log Volatility (Std Dev):**\n", DF_log_volatility)
    print("\n**Full Period Simple Mean Return:**\n", DF_simple_mean)
    print("\n**Full Period Simple Volatility (Std Dev):**\n", DF_simple_volatility)

    # --- Rolling Window Analysis Setup ---
    print(
        f"\nGiven your data has **{len(DF_Adj_Close)}** periods, enter a time_frame and a time_step."
    )
    time_frame = get_integer_input("Enter time_frame (window size, e.g., 20 periods):")
    time_step = get_integer_input("Enter time_step (periods to step, e.g., 5 periods):")

    # Run the rolling calculation
    try:
        (
            df_abs_log_return_roll,
            df_log_volatility_roll,
            df_simple_mean_roll,
            df_simple_volatility_roll,
        ) = calculate_rolling_metrics_optimized(DF_Adj_Close, time_frame, time_step)

        print("\n**Rolling Metrics (First 5 Windows):**")
        print(f"Absolute Log Returns:\n{df_abs_log_return_roll.head()}")
        print(f"Log Volatility:\n{df_log_volatility_roll.head()}")
        print(f"Simple Mean Returns:\n{df_simple_mean_roll.head()}")
        print(f"Simple Volatility:\n{df_simple_volatility_roll.head()}")

    except ValueError as e:
        print(f"\n🛑 Error in rolling calculation: {e}")
        sys.exit()

    # --- Visualization ---
    print("\n🖼️ Generating plots for rolling metrics...")
    plot_metrics(
        df_abs_log_return_roll,
        "Rolling Absolute Logarithmic Return Over Period",
        time_frame,
        time_step,
    )
    plot_metrics(
        df_log_volatility_roll,
        "Rolling Log Volatility (Standard Deviation)",
        time_frame,
        time_step,
    )
    plot_metrics(
        df_simple_mean_roll, "Rolling Simple Mean Return", time_frame, time_step
    )
    plot_metrics(
        df_simple_volatility_roll,
        "Rolling Simple Volatility (Standard Deviation)",
        time_frame,
        time_step,
    )


# ==============================================================================
# 📊 5. Optimized Rolling Metric Function (Replaces original manual loop)
# ==============================================================================


def calculate_rolling_metrics_optimized(
    df: pd.DataFrame, time_frame: int, time_step: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates rolling statistics using efficient pandas vectorized operations.
    The results are stepped by time_step to provide non-overlapping or custom-stepped windows.

    Returns:
        tuple: (df_rolling_abs_log_return, df_rolling_log_volatility,
                df_rolling_simple_mean, df_rolling_simple_volatility)
    """

    if time_frame > len(df):
        raise ValueError("Time frame is larger than the total number of data points.")

    # Calculate daily returns once
    df_simple_returns = df.pct_change()
    df_log_returns = np.log(df / df.shift(1))

    # --- Full Rolling Metrics (Calculated for every day) ---

    # Rolling Simple Mean & Volatility (Standard Deviation)
    df_rolling_simple_mean_full = df_simple_returns.rolling(window=time_frame).mean()
    df_rolling_simple_volatility_full = df_simple_returns.rolling(
        window=time_frame
    ).std()

    # Rolling Log Volatility
    df_rolling_log_volatility_full = df_log_returns.rolling(window=time_frame).std()

    # Rolling Absolute Log Return (ln(P_end / P_start))
    # P_end is the price at the current index, P_start is the price (time_frame - 1) indices before
    P_start = df.shift(time_frame - 1)
    df_rolling_abs_log_return_full = np.log(df / P_start)

    # --- Step the Results ---

    # Rolling results are valid starting from index (time_frame - 1).
    start_index = time_frame - 1

    # Use .iloc slicing to select the values at the desired time_step intervals
    # .dropna(how='all') ensures no columns with only NaNs are included if the final step is incomplete
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
# 📈 6. Visualization Function
# ==============================================================================


def plot_metrics(df: pd.DataFrame, title: str, time_frame: int, time_step: int):
    """Generates and displays a plot for the given DataFrame."""
    # Ensure index is used for plotting (if it's a date index from the stepping)
    # The current index is just a step count, which is fine for plotting.
    df.plot(figsize=(12, 6))
    plt.title(f"{title}\n(Window Size: {time_frame}, Step: {time_step})")
    plt.xlabel("End Date of Window")
    plt.ylabel("Value")
    plt.legend(title="Ticker")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# ==============================================================================
# 🏁 7. Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    execute_analysis()
