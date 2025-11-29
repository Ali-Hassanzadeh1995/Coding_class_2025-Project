import yfinance as yf
from datetime import date, datetime
import re
from typing import Set
import sys

# ==============================================================================
# 🎯 Module: date_checker.py
# Purpose: Ensures all user-provided dates are in the correct format, within
#          valid financial history range, and follow chronological rules (Start <= End).
# ==============================================================================


# --- Helper Function for Strict Format Validation ---
def is_valid_date_format(date_str: str) -> bool:
    """
    Validates if a string is in the STRICT 'YYYY-MM-DD' format (e.g., 2025-01-05).
    It requires two digits for month and two digits for day, enforcing leading zeros.

    Returns:
        bool: True if the format is correct and the date is a valid calendar date.
    """
    # Regex to ensure exactly YYYY-MM-DD pattern (4 digits, hyphen, 2 digits, hyphen, 2 digits)
    date_regex = r"^\d{4}-\d{2}-\d{2}$"

    # 1. Check if the string adheres to the YYYY-MM-DD pattern
    if not re.match(date_regex, date_str):
        return False

    # 2. Check if the string represents a REAL calendar date (e.g., catches '2025-13-40' or '2025-02-30')
    try:
        # datetime.strptime attempts to parse the string into a date object.
        # This will raise ValueError for invalid dates.
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        # Invalid date detected (e.g., too many days in month, month > 12).
        return False


# --- Core Date Checker Functions ---


def get_min_valid_date(stock_symbols: Set[str]) -> date:
    """
    Fetches the historical data start date (period="max") for each symbol using yfinance.
    It returns the LATEST (maximum) of these individual start dates, as this LATEST date
    is the common minimum start date for all selected stocks.

    Args:
        stock_symbols (set): A set of validated stock ticker symbols.

    Returns:
        date: The latest possible start date common to all stocks.

    Raises:
        ValueError: If no valid history could be fetched for any stock.
    """
    print("\n🔍 Checking minimum valid history date for all selected stocks...")
    min_dates = []

    # Iterate through each symbol to find its maximum history start point
    for symbol in stock_symbols:
        try:
            symbol_temp = yf.Ticker(symbol)
            # Request maximum historical data to find the absolute earliest date.
            hist = symbol_temp.history(period="max", auto_adjust=True)

            if hist.empty:
                print(f"⚠️ Could not fetch data for {symbol}. Skipping...", flush=True)
                sys.exit()

            # Extract the earliest date from the DataFrame index (index.min()) and convert to a Python date object.
            start_date_ts = hist.index.min().to_pydatetime().date()
            min_dates.append(start_date_ts)
            print(f"   -> Lowest valid date for {symbol} is {start_date_ts}")

        except Exception as e:
            print(f"⚠️ Error fetching date for {symbol}: {e}", flush=True)
            sys.exit()

    if not min_dates:
        raise ValueError("Could not find a valid start date for any of the stocks.")
        sys.exit()

    # The latest date in the list of earliest dates is the date when all stocks are available.
    common_min_date = max(min_dates)

    print(
        f"\n✅ Given your symbol(s) list, the **minimum common valid start date** is: **{common_min_date}**",
        flush=True,
    )
    return common_min_date


def get_valid_date_input(
    prompt: str, min_date: date = None, max_date: date = None
) -> str:
    """
    Prompts the user for a date, validates its format (YYYY-MM-DD), and checks it against
    optional minimum (oldest allowed date) and maximum (latest allowed date) boundaries.

    Args:
        prompt (str): The input message displayed to the user.
        min_date (date, optional): The earliest acceptable date.
        max_date (date, optional): The latest acceptable date.

    Returns:
        str: The validated date string in 'YYYY-MM-DD' format.
    """
    counter = 0
    while counter < 3:
        date_str = input(prompt).strip()

        # 1. Format and Calendar Validation
        if not is_valid_date_format(date_str):
            print(
                "❌ Entered date is not in the required **YYYY-MM-DD** format or is not a real date! "
                "Please ensure you use **two digits** for month and day (e.g., 2025-01-05). "
                "Please try again."
            )
            counter += 1
            if counter == 3:
                sys.exit()
            continue

        # Convert the validated string to a date object for easy comparison
        input_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # 2. Range Validation (Minimum Date Check)
        # This is used for the Start Date check (must be after all stock histories begin)
        # and for the End Date check (must be after the Start Date).
        if min_date and input_date < min_date:
            print(
                f"❌ Entered date is too early! It must be greater than or equal to **{min_date}**.",
                flush=True,
            )
            counter += 1
            if counter == 3:
                sys.exit()
            continue

        # 3. Range Validation (Maximum Date Check)
        # This is typically used to prevent the user from entering a future date.
        if max_date and input_date > max_date:
            print(
                f"❌ Entered date is in the future! It must be less than or equal to **{max_date}** (Today).",
                flush=True,
            )
            counter += 1
            if counter == 3:
                sys.exit()
            continue
        # If all checks pass
        return date_str


if __name__ == "__main__":
    # --- Example Execution Block (Requires external input_number_symbols module) ---
    from input_number_symbols import get_integer_input, get_valid_symbols

    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Get the set of stock symbols (random test or user-entered validated)
    Set_stocks: Set[str] = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        sys.exit()

    # Find the earliest common start date
    min_common_date = get_min_valid_date(Set_stocks)
    today_date = date.today()
    print(f"\nToday's date is: {today_date}")

    # --- Start Date Input ---
    S_date = get_valid_date_input(
        f"Enter Start Date (YYYY-MM-DD, min: {min_common_date}): ",
        min_date=min_common_date,
        max_date=today_date,
    )

    # --- End Date Input ---
    S_date_dt = datetime.strptime(S_date, "%Y-%m-%d").date()
    E_date = get_valid_date_input(
        f"Enter End Date (YYYY-MM-DD, min: {S_date}): ",
        min_date=S_date_dt,  # End date must be on or after the selected Start Date
        max_date=today_date,
    )

    print(f"\n**Final Valid Data Range:** {S_date} to {E_date}")
