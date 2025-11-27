import numpy as np
from datetime import date, datetime


def set_interval(S_date, E_date):
    S_date = np.datetime64(S_date)
    E_date = np.datetime64(E_date)

    diff_days = (E_date - S_date).astype(int)

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

    # Filter VALID_INTERVALS safely based on diff_days
    if diff_days <= 7:
        valid_keys = ["1m"]
    elif 7 < diff_days <= 60:
        valid_keys = [k for k in VALID_INTERVALS if k not in ["1m"]]
    else:  # diff_days > 60
        valid_keys = [
            k
            for k in VALID_INTERVALS
            if k not in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
        ]

    filtered_intervals = {k: VALID_INTERVALS[k] for k in valid_keys}

    print(f"\n📅 Given the selected dates, acceptable intervals are:", flush=True)
    for key, valu in filtered_intervals.items():
        print(f"  - **{key}**: {valu}", flush=True)

    # Get and validate the interval input
    while True:
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

    coefficient = INTERVALS_coefficient[interval]

    # ✅ Return as tuple
    return interval, coefficient


if __name__ == "__main__":
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
        f"Enter Start Date (YYYY-MM-DD, e.g., 2025-01-01): ",
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
