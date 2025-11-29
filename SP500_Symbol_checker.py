from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
from typing import Set, Union

# ==============================================================================
# 🎯 Improvement Plan Context:
# This class implements a robust web scraper to get the S&P 500 list from Wikipedia.
# Key improvements include:
# 1. Dynamic Table Selection: Uses regex to reliably find the correct table by searching
#    for the 'symbol' keyword, fixing the 'find_all("table")[0]' fragility.
# 2. Resiliency: Includes error handling for HTTP requests (timeouts, status codes)
#     and file I/O (missing or corrupt CSV).
#
# Future work (not implemented here):
# * Address Weakness 2 (Hardcoded Source) by passing the URL as an argument.
# * Address Weakness 3 (Access Issues) by implementing header rotation or delays.
# ==============================================================================


class SP500_Symbol_checker:
    """
    A utility class for fetching, checking, and generating S&P 500 stock symbols.
    It manages the persistence of the symbol list in 'Symbols.csv' for efficiency.
    """

    def __init__(self, Symbol: str = None, Number: int = 0) -> None:
        """
        Initializes the symbol checker with optional parameters.
        Symbol: The specific ticker to check (e.g., "AAPL").
        Number: The quantity of random symbols to generate.
        """
        self.Symbol = Symbol
        self.Number = Number

    # ==========================================================================
    # ⚙️ Core Scraper Function
    # ==========================================================================

    def Symbols_df_maker(self):
        """
        Scrapes the S&P 500 list from Wikipedia.
        It identifies the correct table, extracts headers and data rows, and saves the result to 'Symbols.csv'.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Define a basic User-Agent header to identify the request and prevent immediate blocking.
        headers = {"User-Agent": "AliWikiBot/1.0 (https://github.com/ali)"}

        print(f"Attempting to scrape data from: {url}")
        try:
            # Send the request with a timeout for robustness.
            page = requests.get(url, headers=headers, timeout=10)
            # Raise an exception for bad HTTP status codes (4xx or 5xx), including 403 Forbidden.
            page.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return

        soup = BeautifulSoup(page.text, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            print("No tables found on the page.")
            return

        # Dynamic Table Selection: Iterate through tables to find the one containing the tickers.
        counter = 0
        while counter < len(tables):
            # Use regex to search for the case-insensitive string "symbol" within the table's HTML structure.
            if re.search("symbol", str(tables[counter]), re.IGNORECASE):
                break
            counter += 1

        if counter == len(tables):
            print("Could not find the data table containing the 'symbol' column.")
            return

        table = tables[counter]

        # Data Extraction: Extract headers (<th> tags).
        headers = [th.text.strip() for th in table.find_all("th")]
        df = pd.DataFrame(columns=headers)

        # Data Extraction: Extract rows (<tr> tags).
        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip the header row (index 0)
            # Extract data from both data cells (<td>) and header cells (<th>)
            # to handle variations in Wikipedia table formatting.
            cells = [
                d.text.strip()
                for d in row.find_all(["td", "th"])
                if d.text.strip() != ""
            ]

            # Validation: Only append the row if the number of cells matches the number of headers.
            if len(cells) == len(df.columns):
                # Efficiently append the new row to the DataFrame.
                df.loc[len(df)] = cells

        # Save the result. index=False prevents writing the DataFrame index to the CSV.
        df.to_csv("Symbols.csv", index=False)
        print("S&P 500 symbols successfully scraped and saved to Symbols.csv.")

    # ==========================================================================
    # 📚 Helper Function (Internal Use)
    # ==========================================================================

    def _read_symbol_data(self) -> Union[pd.DataFrame, bool]:
        """
        Internal helper function to handle reading 'Symbols.csv', with fallback logic.
        Calls Symbols_df_maker() if the file is missing or detects a corrupted structure.
        Returns the DataFrame on success, or False on unrecoverable failure.
        """
        try:
            df = pd.read_csv("Symbols.csv")

            # Check for the critical 'Symbol' column. If missing, the CSV is corrupt or malformed.
            if "Symbol" not in df.columns:
                print("CSV file missing 'Symbol' column. Re-scraping data...")
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")
            return df

        except FileNotFoundError:
            print("Symbols.csv not found. Scraping data...")
            self.Symbols_df_maker()
            # Attempt to read again after scraping
            try:
                df = pd.read_csv("Symbols.csv")
                return df
            except Exception as e:
                print(f"Error after re-scraping: {e}")
                return False

        except Exception as e:
            print(f"Unexpected error while reading CSV: {e}")
            return False

    # ==========================================================================
    # 📈 Public Interface Functions
    # ==========================================================================

    def Symbols_check(self) -> bool:
        """
        Checks if the provided symbol (self.Symbol) exists in the S&P 500 list.
        Returns True if valid, False otherwise.
        """
        df = self._read_symbol_data()
        if isinstance(df, bool) and not df:
            return False

        # Convert the 'Symbol' column to a set for fast O(1) lookup time.
        set_symbols = set(df["Symbol"])

        # Check the symbol, converting it to uppercase for case-insensitivity.
        return self.Symbol.upper() in set_symbols

    def Symbols_random_gen(self) -> Union[Set[str], bool]:
        """
        Generates a random set of S&P 500 stock symbols of size self.Number.
        Returns a Python set of random symbols or False on failure.
        """
        df = self._read_symbol_data()
        if isinstance(df, bool) and not df:
            return False

        # Convert the 'Symbol' column to a list for selection.
        list_symbols = list(df["Symbol"])

        # Use numpy's choice function to randomly select the required number of unique symbols.
        # Number should be less than 500!
        if self.Number > len(list_symbols):
            print(
                f"Cannot select {self.Number} unique symbols from {len(list_symbols)} symbols (number of symbols must be less than total symbols)."
            )
        else:
            random_stocks = np.random.choice(list_symbols, self.Number, replace=False)

        # Convert the numpy array of chosen symbols into a Python set of strings.
        Set_stocks = {str(i) for i in random_stocks}

        return Set_stocks


# Example usage:
# checker = SP500_Symbol_checker(Symbol="GOOG", Number=3)
# # checker.Symbols_df_maker() # Only call this if you need to force a scrape
# print(f"Is GOOG in S&P 500? {checker.Symbols_check()}")
# print(f"Random Tickers: {checker.Symbols_random_gen()}")
