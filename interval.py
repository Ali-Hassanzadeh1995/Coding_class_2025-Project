import numpy as np
from datetime import date, datetime
from typing import Tuple
import sys

# ==============================================================================
# 🎯 Module: interval.py
# Purpose: Determines the list of valid data intervals (e.g., '1d', '1mo')
#          based on the date range, prompts the user for a selection, and
#          returns the corresponding annualization coefficient.
# ==============================================================================


def set_interval(S_date: str, E_date: str) -> Tuple[str, int]:
    """
    Calculates the duration between the start and end dates, filters the list of
    available Yahoo Finance intervals based on that duration, prompts the user
    for an interval, and returns the chosen interval and its annualization coefficient.

    Args:
        S_date (str): Start date string (YYYY-MM-DD).
        E_date (str): End date string (YYYY-MM-DD).

    Returns:
        Tuple[str, int]: The selected interval string and the annualization coefficient.
    """
    # Convert date strings to numpy datetime objects for easy arithmetic difference (in days).
    S_date_dt = np.datetime64(S_date)
    E_date_dt = np.datetime64(E_date)

    # Calculate the number of days between the start and end dates.
    diff_days = (E_date_dt - S_date_dt).astype(int)

    # Dictionary of all valid yfinance intervals and their limitations.
    VALID_INTERVALS = {
        "1m": "1 minute (Max 7 days)",
        "2m": "2 minute (Max 60 days)",
        "5m": "5 minute (Max 60 days)",
        "15m": "15 minute (Max 60 days)",
        "30m": "30 minute (Max 60 days)",
        "60m": "60 minute (Max 60 days)",
        "90m": "90 minute (Max 60 days)",
        "1d": "1 day (Full history)",
        "5d": "5 day (Full history)",
        "1wk": "1 week (Full history)",
        "1mo": "1 month (Full history)",
        "3mo": "3 month (Full history)",
    }

    # Filter VALID_INTERVALS based on the duration (diff_days) because intraday intervals
    # have strict lookback limits set by the data provider (e.g., Yahoo Finance).
    if diff_days <= 7:
        # Only 1-minute data is available for very short ranges (Max 7 days).
        valid_keys = ["1m"]
    elif 7 < diff_days <= 60:
        # Intraday data (2m to 90m) is available, along with daily and longer intervals.
        valid_keys = [k for k in VALID_INTERVALS if k not in ["1m"]]
    else:  # diff_days > 60
        # Only daily, weekly, and monthly data are available for longer ranges (Full history).
        valid_keys = [
            k
            for k in VALID_INTERVALS
            if k not in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
        ]

    filtered_intervals = {k: VALID_INTERVALS[k] for k in valid_keys}

    # --- User Input and Validation ---
    print(f"\n📅 Given the selected dates, acceptable intervals are:", flush=True)
    for key, valu in filtered_intervals.items():
        print(f"  - **{key}**: {valu}", flush=True)

    counter = 0
    while counter < 4:
        interval = input("\nEnter your suitable interval: ").strip().lower()
        if interval in filtered_intervals:
            print(
                f"\n✅ Entered interval is **{interval}** ({filtered_intervals[interval]}).",
                flush=True,
            )
            break
        else:
            print(
                "\n❌ Entered interval is not acceptable! Please choose from the list.",
                flush=True,
            )
            counter += 1
            if counter == 4:
                sys.exit()

    # --- Annualization Coefficient Assignment ---
    # The coefficient is the estimated number of periods in one year.
    # This is critical for annualizing metrics like volatility and Sharpe ratio.
    # E.g., 252 trading days/year, 52 weeks/year, 12 months/year.
    # The intraday values (1m-90m) are approximations based on 6.5 trading hours/day * 252 days.
    INTERVALS_coefficient = {
        "1m": 98280,  # 6.5 hrs * 60 min * 252 days
        "2m": 49140,
        "5m": 19656,
        "15m": 6552,
        "30m": 3276,
        "60m": 1638,
        "90m": 1092,
        "1d": 252,  # Standard trading days per year
        "5d": 52,  # Trading weeks per year
        "1wk": 52,  # Calendar weeks per year
        "1mo": 12,  # Months per year
        "3mo": 4,  # Quarters per year
    }

    coefficient = INTERVALS_coefficient[interval]

    # Return the chosen interval key (e.g., '1d') and its corresponding coefficient (e.g., 252).
    return interval, coefficient


if __name__ == "__main__":
    # --- Example Execution Block (Requires external modules) ---
    try:
        from input_number_symbols import get_integer_input, get_valid_symbols
        from date_checker import get_min_valid_date, get_valid_date_input

        N_stocks = get_integer_input("Please enter the number of stocks: ")

        Set_stocks = get_valid_symbols(N_stocks)

        if not Set_stocks:
            print("Execution aborted due to symbol entry error.")
            exit()

        min_common_date = get_min_valid_date(Set_stocks)
        today_date = date.today()
        print(f"Today's date is: {today_date}")

        # Get Start Date
        S_date = get_valid_date_input(
            f"Enter Start Date (YYYY-MM-DD, min: {min_common_date}): ",
            min_date=min_common_date,
            max_date=today_date,
        )

        # Get End Date
        E_date = get_valid_date_input(
            f"Enter End Date (YYYY-MM-DD, e.g., 2025-11-07): ",
            min_date=datetime.strptime(S_date, "%Y-%m-%d").date(),
            max_date=today_date,
        )

        print(f"\n**Data Range:** {S_date} to {E_date}")

        # Call set_interval
        interval, coefficient = set_interval(S_date, E_date)
        print(f"\nSelected interval: {interval}, Coefficient: {coefficient}")

    except ImportError:
        print(
            "Error: Required utility modules ('input_number_symbols.py', 'date_checker.py') not found."
        )
        # Provide a basic test case if modules are missing
        print("Running isolated test for set_interval...")
        test_interval, test_coeff = set_interval("2025-01-01", "2025-11-01")
        print(f"Test Result: Interval={test_interval}, Coefficient={test_coeff}")
