import numpy as np


def set_interval(S_date, E_date):
    S_date = np.datetime64(S_date)
    E_date = np.datetime64(E_date)

    diff_days = E_date - S_date
    diff_days = diff_days.astype(int)

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
    if 7 < diff_days <= 60:
        VALID_INTERVALS = dict(list(VALID_INTERVALS.items())[1:])
    elif diff_days > 60:
        VALID_INTERVALS = dict(list(VALID_INTERVALS.items())[7:])
    print(f"\n📅 Given the selected dates, acceptable intervals are:")
    for key, valu in VALID_INTERVALS.items():
        print(f"  - **{key}**: {valu}")

    # Get and validate the interval input
    while True:
        interval = input("\nEnter your suitable interval:").strip().lower()
        if interval in VALID_INTERVALS:
            print(
                f"\n✅ Entered interval is **{interval}** ({VALID_INTERVALS[interval]})."
            )
            break
        else:
            print(
                "\n❌ Entered interval is not acceptable! Please choose from the list."
            )
    return interval


from input_number_symbols import get_integer_input, get_valid_symbols
from date_checker import get_min_valid_date, get_valid_date_input
from datetime import date, datetime

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
set_interval(S_date, E_date)
