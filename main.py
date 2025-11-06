"""
Stock Data Input and Validation Script
--------------------------------------
Collects a list of stock tickers from the user, validates them against the S&P 500 list,
and retrieves date ranges for historical data collection.

Author: Ali Hassanzadeh
"""

import pandas as pd
import yfinance as yf
from datetime import date
from SP500_Symbol_checker import SP500_Symbol_checker


class StockDataInput:
    """Handles user input, symbol validation, and date selection for stock data."""

    def __init__(self):
        self.num_stocks = 0
        self.stock_set = set()
        self.today = date.today()
        self.start_date = None
        self.end_date = None
        self.min_valid_start = None

    # ---------- Utility Methods ---------- #
    def get_integer_input(self, prompt):
        """Ensure integer input from user."""
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("❌ Invalid input! Please enter an integer.")

    def validate_date_format(self, date_str):
        """Check if date_str is in valid YYYY-MM-DD format."""
        if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
            return False
        try:
            y, m, d = map(int, date_str.split("-"))
            return (1 <= m <= 12) and (1 <= d <= 31)
        except ValueError:
            return False

    def get_date_input(self, prompt, min_limit=None, max_limit=None):
        """Prompt user for a date and ensure it’s within the valid range."""
        while True:
            date_str = input(prompt).strip()
            if not self.validate_date_format(date_str):
                print("❌ Invalid date format! Use YYYY-MM-DD.")
                continue

            y, m, d = map(int, date_str.split("-"))
            date_list = [y, m, d]

            if min_limit and date_list < min_limit:
                print(
                    f"⚠️ Date must be >= {min_limit[0]}-{min_limit[1]:02d}-{min_limit[2]:02d}."
                )
                continue
            if max_limit and date_list > max_limit:
                print(
                    f"⚠️ Date must be <= {max_limit[0]}-{max_limit[1]:02d}-{max_limit[2]:02d}."
                )
                continue

            return date_str  # valid date

    # ---------- Main Steps ---------- #
    def get_number_of_stocks(self):
        """Ask for number of stocks."""
        self.num_stocks = self.get_integer_input("Enter the number of stocks: ")

    def get_symbols(self):
        """Ask user for stock symbols or generate random ones."""
        choice = (
            input(
                f"If you want to test the program, type 'test' to generate {self.num_stocks} random stocks. Otherwise, type 'N': "
            )
            .strip()
            .lower()
        )

        if choice == "test":
            self.stock_set = SP500_Symbol_checker(
                None, self.num_stocks
            ).Symbols_random_gen()
            print(f"✅ Generated random stocks: {self.stock_set}")
        else:
            print("\nEnter the stock symbols you want to analyze.")
            for i in range(self.num_stocks):
                for attempt in range(3):
                    symbol = input(f"Enter symbol {i+1}: ").upper()
                    if SP500_Symbol_checker(symbol, 0).Symbols_check():
                        self.stock_set.add(symbol)
                        break
                    else:
                        print(f"❌ {symbol} is not a valid S&P 500 symbol!")
                else:
                    print("⚠️ Skipping after 3 invalid attempts.")
            print(f"\n✅ Final symbol set: {self.stock_set}")

    def find_minimum_valid_start(self):
        """Find the latest earliest date among selected tickers."""
        print("\nFetching earliest available dates...")
        min_dates = []

        for ticker in self.stock_set:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="max")
                if hist.empty:
                    print(f"⚠️ No data found for {ticker}. Skipping.")
                    continue
                first_date = hist.index.min().strftime("%Y-%m-%d")
                y, m, d = map(int, first_date.split("-"))
                min_dates.append([y, m, d])
                print(f"📅 Earliest valid date for {ticker}: {first_date}")
            except Exception as e:
                print(f"⚠️ Error retrieving data for {ticker}: {e}")

        if not min_dates:
            raise ValueError("No valid data found for any ticker.")

        self.min_valid_start = max(min_dates)
        print(
            f"\n📊 The latest earliest valid date across tickers is "
            f"{self.min_valid_start[0]}-{self.min_valid_start[1]:02d}-{self.min_valid_start[2]:02d}."
        )

    def get_date_range(self):
        """Ask user for start and end dates within valid range."""
        print(f"\nNote: Date format is YYYY-MM-DD (e.g. {self.today})")

        self.start_date = self.get_date_input(
            "Enter Start Date: ", min_limit=self.min_valid_start
        )
        self.end_date = self.get_date_input(
            "Enter End Date: ",
            min_limit=[
                self.min_valid_start[0],
                self.min_valid_start[1],
                self.min_valid_start[2],
            ],
            max_limit=[self.today.year, self.today.month, self.today.day],
        )

        print(f"\n✅ Final Date Range: {self.start_date} → {self.end_date}")

    # ---------- Run Full Input Process ---------- #
    def run(self):
        """Run the entire user input sequence."""
        print("=== STOCK DATA INPUT ===")
        self.get_number_of_stocks()
        self.get_symbols()
        self.find_minimum_valid_start()
        self.get_date_range()
        print("\n✅ All inputs validated successfully! Ready for data analysis.")


# ---------- Main Program ---------- #
if __name__ == "__main__":
    stock_input = StockDataInput()
    stock_input.run()
