import yfinance as yf
from datetime import date, datetime
import re  # <-- Import the regex module

# Assuming get_integer_input and get_valid_symbols are available from input_number_symbols


# --- Helper Function for Strict Format Validation ---
def is_valid_date_format(date_str: str) -> bool:
    """
    Validates if a string is in the STRICT 'YYYY-MM-DD' format (e.g., 2025-01-05).
    Requires two digits for month and two digits for day.
    """
    # Regex to ensure exactly YYYY-MM-DD format (must have leading zeros)
    date_regex = r"^\d{4}-\d{2}-\d{2}$"

    # 1. Check if the string adheres to the YYYY-MM-DD pattern
    if not re.match(date_regex, date_str):
        return False

    # 2. Check if the string represents a REAL calendar date (e.g., no 2025-13-40)
    try:
        # datetime.strptime attempts to parse the string into a date object
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        # This catches invalid dates like '2025-02-30'
        return False


# --- End of Helper Function ---


def get_min_valid_date(stock_symbols: set) -> date:
    """
    Fetches the historical data start date for each symbol and returns the
    LATEST (maximum) of these start dates, as it's the minimum common date.
    """
    print("\n🔍 Checking minimum valid history date for all selected stocks...")
    min_dates = []

    # Iterate through each symbol to find its maximum history
    for symbol in stock_symbols:
        try:
            symbol_temp = yf.Ticker(symbol)
            # Fetch minimal data to get the index start date
            hist = symbol_temp.history(period="max", auto_adjust=True)

            if hist.empty:
                print(f"⚠️ Could not fetch data for {symbol}. Skipping...")
                continue

            # The start date of the history is the minimum index value
            start_date_ts = hist.index.min().to_pydatetime().date()
            min_dates.append(start_date_ts)
            print(f"   -> Lowest valid date for {symbol} is {start_date_ts}")

        except Exception as e:
            print(f"⚠️ Error fetching date for {symbol}: {e}")
            continue

    # The maximum (latest) of the individual min dates is the common minimum start date
    if not min_dates:
        raise ValueError("Could not find a valid start date for any of the stocks.")

    common_min_date = max(min_dates)

    print(
        f"\n✅ Given your symbol(s) list, the **minimum common valid start date** is: **{common_min_date}**"
    )
    return common_min_date


def get_valid_date_input(
    prompt: str, min_date: date = None, max_date: date = None
) -> str:
    """
    Prompts the user for a date, validates its strict format, and checks it against
    optional minimum and maximum date boundaries.
    """
    while True:
        date_str = input(prompt).strip()

        # 1. Format Validation (Uses the new, strict function)
        if not is_valid_date_format(date_str):
            print(
                "❌ Entered date is not in the required **YYYY-MM-DD** format! "
                "Please ensure you use **two digits** for month and day (e.g., 2025-01-05). "
                "Please try again."
            )
            continue

        # Convert valid string to a date object for comparison
        input_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # 2. Range Validation (Min Date Check - for Start Date)
        if min_date and input_date < min_date:
            print(
                f"❌ Entered date is too early! Start date must be greater than or equal to **{min_date}**."
            )
            continue

        # 3. Range Validation (Max Date Check - for End Date)
        if max_date and input_date > max_date:
            print(
                f"❌ Entered date is in the future! End date must be less than or equal to **{max_date}** (Today)."
            )
            continue

        # If all checks pass
        return date_str


if __name__ == "__main__":
    from input_number_symbols import get_integer_input, get_valid_symbols

    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Get the set of stock symbols (either random test or user-entered validated)
    Set_stocks = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        exit()

    min_common_date = get_min_valid_date(Set_stocks)
    today_date = date.today()
    print(f"Today's date is: {today_date}")

    # Get and validate the Start Date
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
