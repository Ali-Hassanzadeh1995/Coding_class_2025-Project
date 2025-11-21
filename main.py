# Get number of stocks and creating the List of stocks

from input_number_symbols import get_integer_input, get_valid_symbols

N_stocks = get_integer_input("Please enter the number of stocks: ")

# Get the set of stock symbols (either random test or user-entered validated)
Set_stocks = get_valid_symbols(N_stocks)

if not Set_stocks:
    print("Execution aborted due to symbol entry error.")
    exit()

# Convert set to a list for yfinance download
List_stocks = list(Set_stocks)
print(f"\n**Selected Stocks:** {List_stocks}\n")


# Get and validate the Start Date
from datetime import date, datetime
from date_checker import get_min_valid_date, get_valid_date_input

min_common_date = get_min_valid_date(Set_stocks)
today_date = date.today()
print(f"Today's date is: {today_date}")

S_date = get_valid_date_input(
    f"Enter Start Date (YYYY-MM-DD, e.g., 2025-01-01): ",
    min_date=min_common_date,
    max_date=today_date,  # Start date should also not be in the future
)

# Get and validate the End Date
E_date = get_valid_date_input(
    f"Enter End Date (YYYY-MM-DD, e.g., 2025-11-07): ",
    min_date=datetime.strptime(
        S_date, "%Y-%m-%d"
    ).date(),  # End date must be >= Start Date
    max_date=today_date,
)

# Check for a sensible duration (Start Date <= End Date) is handled by min_date in E_date check.

print(f"\n**Data Range:** {S_date} to {E_date}")

# Interval

from interval import set_interval

interval = set_interval(S_date, E_date)

## ⬇️ 4. Data Download and Initial Return Calculation

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
import matplotlib.pyplot as plt

print("\n⬇️ Downloading data from Yahoo Finance...")

# Download data for all stocks in the specified range and interval
try:
    data = yf.download(
        List_stocks,
        start=S_date,
        end=E_date,
        interval=interval,
        # auto_adjust=False is fine to collect Adj close prise, but yfinance default is True
        auto_adjust=False,
    )
    # Check if data was actually returned
    if data.empty:
        raise ValueError("No data returned. Check symbols, dates, and interval limits.")

except Exception as e:
    print(f"🛑 Error during data download: {e}")
    exit()

# Extract the Adjusted Close prices (most common for return analysis)
DF_Adj_Close = data["Adj Close"]

# Calculate Simple Returns: R_t = (P_t / P_{t-1}) - 1
DF_simple_return = DF_Adj_Close.pct_change().dropna()
print("\n--- Simple Returns (First 5 Rows) ---")
print(DF_simple_return.head())

# Calculate Logarithmic Returns: r_t = ln(P_t / P_{t-1})
DF_log_return = np.log(DF_Adj_Close / DF_Adj_Close.shift(1)).dropna()
DF_log_volatility = DF_log_return.std()
print("\n--- Log Returns (First 5 Rows) ---")
print(DF_log_return.head())
print("\n**Full Period log Volatility (Std Dev):**")
print(DF_log_volatility)

# Calculate full period statistics
DF_simple_mean = DF_simple_return.mean()
DF_simple_volatility = DF_simple_return.std()
print("\n**Full Period Simple Mean Return:**")
print(DF_simple_mean)
print("\n**Full Period Simple Volatility (Std Dev):**")
print(DF_simple_volatility)


## 📊 5. Rolling Window Analysis
# Note: The logic for calculating len_day based on interval[0] was complex
# and potentially inaccurate, so I removed it and simplified the user prompt
# based on the actual length of the fetched DataFrame.


def calculate_rolling_metrics(
    df: pd.DataFrame, time_frame: int, time_step: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates rolling log returns, simple mean, and simple volatility
    over the given time_frame, stepped by time_step.

    Args:
        df (pd.DataFrame): DataFrame of Adjusted Close prices.
        time_frame (int): The window size for the rolling calculation.
        time_step (int): How many periods to step the window forward.

    Returns:
        tuple: (df_rolling_log_return, df_rolling_simple_mean, df_rolling_simple_volatility)
    """
    # Calculate the number of steps possible
    total_periods = len(df)
    if time_frame > total_periods:
        raise ValueError("Time frame is larger than the total number of data points.")

    # Calculate initial offset for a cleaner rolling window if the remainder is not 0
    residual = (total_periods - time_frame) % time_step
    steps = (
        total_periods - time_frame - residual
    ) // time_step + 1  # +1 to include the last window

    # Initialize DataFrames for results
    columns = df.columns.tolist()
    df_rolling_log_return = pd.DataFrame(columns=columns, index=range(steps))
    df_rolling_log_volatility = pd.DataFrame(columns=columns, index=range(steps))
    df_rolling_simple_mean = pd.DataFrame(columns=columns, index=range(steps))
    df_rolling_simple_volatility = pd.DataFrame(columns=columns, index=range(steps))

    # Iterate through the data for the rolling calculation
    for i in range(steps):
        # Calculate the start and end index for the current window
        start_idx = i * time_step
        end_idx = start_idx + time_frame

        # Select the data for the current time frame
        # We start from the remainder (residual) to ensure clean steps
        Temp_window = df.iloc[start_idx:end_idx]

        # 1. Rolling Log Return: ln(P_end / P_start) - Note: This is an *absolute* log return over the period,
        # not the sum/mean of daily log returns.
        # Temp_window.iloc[-1] is the last price, Temp_window.iloc[0] is the first price.
        # np.log(P_end / P_start) calculates the log price change over the window.
        df_rolling_log_return.iloc[i] = np.log(
            Temp_window.iloc[-1] / Temp_window.iloc[0]
        )
        Temp_log_return = np.log(Temp_window / Temp_window.shift(1)).dropna()
        df_rolling_log_volatility.iloc[i] = Temp_log_return.std()

        # 2. Calculate Simple Returns for the window
        Temp_simple_return = Temp_window.pct_change().dropna()

        # 3. Rolling Simple Mean Return
        df_rolling_simple_mean.iloc[i] = Temp_simple_return.mean()

        # 4. Rolling Simple Volatility (Standard Deviation)
        df_rolling_simple_volatility.iloc[i] = Temp_simple_return.std()

    return (
        df_rolling_log_return,
        df_rolling_log_volatility,
        df_rolling_simple_mean,
        df_rolling_simple_volatility,
    )


# Get rolling window parameters
print(
    f"\nGiven your data has **{len(DF_Adj_Close)}** periods, enter a time_frame and a time_step."
)
time_frame = get_integer_input("Enter time_frame (window size):")
time_step = get_integer_input("Enter time_step (periods to step):")

# Calculate rolling metrics
try:
    (
        df_log_retrun_roll,
        df_log_volatility_roll,
        df_simple_mean_roll,
        df_simple_volatility_roll,
    ) = calculate_rolling_metrics(DF_Adj_Close, time_frame, time_step)

    print("\n**Rolling Log Returns:**")
    print(df_log_retrun_roll.head())
    print("\n**Rolling Log volatility:**")
    print(df_log_volatility_roll.head())
    print("\n**Rolling Simple Returns:**")
    print(df_simple_mean_roll.head())
    print("\n**Rolling Simple Volatility:**")
    print(df_simple_volatility_roll.head())

except ValueError as e:
    print(f"\n🛑 Error in rolling calculation: {e}")
    exit()


## 📈 6. Visualization
# Plotting the results
def plot_metrics(df: pd.DataFrame, title: str):
    """Generates and displays a plot for the given DataFrame."""
    df.plot(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(f"Rolling Step (Time Frame: {time_frame}, Step: {time_step})")
    plt.ylabel("Value")
    plt.legend(title="Ticker")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


print("\n🖼️ Generating plots for rolling metrics...")
plot_metrics(df_log_retrun_roll, "Rolling Logarithmic Return Over Period")
plot_metrics(df_log_volatility_roll, "Rolling Log Volatility (Standard Deviation)")
plot_metrics(df_simple_mean_roll, "Rolling Simple Mean Return")
plot_metrics(
    df_simple_volatility_roll, "Rolling Simple Volatility (Standard Deviation)"
)
