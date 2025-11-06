"""
🎯 Improvement Plan: Generic Web Table Scraper (Currently focused on S&P 500)

This code currently extracts the S&P 500 list from a hardcoded Wikipedia URL.
While functional, it is **fragile** and **not reusable** for other sources like NASDAQ.

**Key Weaknesses to Address:**
1.  **Fragile Table Selection:** Uses `find_all("table")[0]`, which breaks if the target table index changes.
2.  **Hardcoded Source:** The URL is fixed, preventing use for other data sources.
3.  **Access Issues:** The 403 Forbidden error suggests need for **advanced header handling** (e.g., rotation/delays) to prevent blocking.

"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


class SP500_Symbol_checker:
    # Changed class methods to be instance methods where appropriate
    # The __init__ is currently not used but kept for class structure.
    def __init__(self, Symbol, Number) -> None:
        self.Symbol = Symbol
        self.Number = Number

    def Symbols_check(
        self,
    ):
        # Corrected to use the instance's Symbol and call Symbols_df_maker with the class name or self
        try:
            df = pd.read_csv("Symbols.csv")
            # Ensure the 'Symbol' column exists before trying to access it
            if "Symbol" not in df.columns:
                print("CSV file is missing the 'Symbol' column. Re-scraping data.")
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")

            set_symbols = set(df["Symbol"])
        except FileNotFoundError:
            # If the file doesn't exist, create it then try again
            print("Symbols.csv not found. Scraping data...")
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = set(df["Symbol"])
        except Exception as e:
            print(f"An unexpected error occurred while reading CSV: {e}")
            return False

        # Access the Symbol stored in the instance and convert it to uppercase
        if self.Symbol.upper() in set_symbols:
            return True
        else:
            return False

    def Symbols_random_gen(self):
        try:
            df = pd.read_csv("Symbols.csv")
            # Ensure the 'Symbol' column exists before trying to access it
            if "Symbol" not in df.columns:
                print("CSV file is missing the 'Symbol' column. Re-scraping data.")
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")
            set_symbols = list(df["Symbol"])
        except FileNotFoundError:
            # If the file doesn't exist, create it then try again
            print("Symbols.csv not found. Scraping data...")
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = list(df["Symbol"])
        except Exception as e:
            print(f"An unexpected error occurred while reading CSV: {e}")
            return False
        set_stocks = set(np.random.choice(set_symbols, 10))
        Set_stocks = set()
        for i in set_stocks:
            Set_stocks.add(str(i))
        return Set_stocks

    # Changed to an instance method (added self) for consistency and potential future use of self attributes
    def Symbols_df_maker(self):
        # URL is hardcoded, which is the point to be improved in future refactoring.
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Headers for ethical scraping and to help avoid 403 errors.
        headers = {"User-Agent": "AliWikiBot/1.0 (https://github.com/ali)"}

        try:
            page = requests.get(url, headers=headers)
            page.raise_for_status()  # Check for HTTP errors like 403
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return  # Exit if request fails

        soup = BeautifulSoup(page.text, "html.parser")  # Explicitly set parser

        # FRAGILE POINT: Assumes the target table is always the first one (index 0).
        tables = soup.find_all("table")
        if not tables:
            print("No tables found on the page.")
            return

        table = tables[0]

        word_titles = table.find_all("th")
        word_titles_list = [title.text.strip() for title in word_titles]
        df = pd.DataFrame(columns=word_titles_list)

        row_datas = table.find_all("tr")

        # Iterate over rows, skipping the header row (index 0)
        for data in row_datas[1:]:
            # Use 'td' (table data) elements for robustness, though 'd.text' in tr works for this page
            row_data = [
                d.text.strip()
                for d in data.find_all(["td", "th"])
                if d.text.strip() != ""
            ]

            # Simple check to ensure the row has the right number of columns before insertion
            if len(row_data) == len(df.columns):
                df.loc[len(df)] = row_data
            # else:
            #     # Optional: Add logging/printing for rows that are skipped due to mismatch

        # Save the DataFrame to a CSV file.
        df.to_csv("Symbols.csv", index=False)
        print("S&P 500 symbols successfully scraped and saved to Symbols.csv")


# Example of how to use the corrected class:
# scraper = SP500_Symbol_checker("AAPL")
# scraper.Symbols_df_maker()  # Scrapes the data (only needed once or to refresh)
# is_sp500 = scraper.Symbols_check()
# print(f"Is AAPL in S&P 500? {is_sp500}")
