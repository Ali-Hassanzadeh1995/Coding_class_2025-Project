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
import re


class SP500_Symbol_checker:
    """
    A utility class for handling S&P 500 stock symbols.
    """

    def __init__(self, Symbol, Number) -> None:
        self.Symbol = Symbol
        self.Number = Number

    def Symbols_check(self):
        """
        Check if the given symbol exists in the S&P 500 list.
        Returns True if valid, False otherwise.
        """
        try:
            df = pd.read_csv("Symbols.csv")

            # Ensure the 'Symbol' column exists
            if "Symbol" not in df.columns:
                print("CSV file missing 'Symbol' column. Re-scraping data...")
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")

            set_symbols = set(df["Symbol"])

        except FileNotFoundError:
            print("Symbols.csv not found. Scraping data...")
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = set(df["Symbol"])

        except Exception as e:
            print(f"Unexpected error while reading CSV: {e}")
            return False

        return self.Symbol.upper() in set_symbols

    def Symbols_random_gen(self):
        """
        Generate a random set of S&P 500 stock symbols.
        Returns a Python set of random symbols.
        """
        try:
            df = pd.read_csv("Symbols.csv")

            # Ensure the 'Symbol' column exists
            if "Symbol" not in df.columns:
                print("CSV file missing 'Symbol' column. Re-scraping data...")
                self.Symbols_df_maker()
                df = pd.read_csv("Symbols.csv")

            set_symbols = list(df["Symbol"])

        except FileNotFoundError:
            print("Symbols.csv not found. Scraping data...")
            self.Symbols_df_maker()
            df = pd.read_csv("Symbols.csv")
            set_symbols = list(df["Symbol"])

        except Exception as e:
            print(f"Unexpected error while reading CSV: {e}")
            return False

        # Generate random symbols
        random_stocks = np.random.choice(set_symbols, self.Number)
        Set_stocks = {str(i) for i in random_stocks}

        return Set_stocks

    def Symbols_df_maker(self):
        """
        Scrape the S&P 500 list from Wikipedia and save it as Symbols.csv.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "AliWikiBot/1.0 (https://github.com/ali)"}

        try:
            page = requests.get(url, headers=headers)
            page.raise_for_status()  # Check for HTTP errors

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return

        soup = BeautifulSoup(page.text, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            print("No tables found on the page.")
            return
        counter = 0
        while True:
            if re.search("symbol", str(tables[counter]), re.IGNORECASE):
                break
            else:
                counter += 1

        table = tables[counter]
        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        df = pd.DataFrame(columns=headers)

        # Extract rows
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = [
                d.text.strip()
                for d in row.find_all(["td", "th"])
                if d.text.strip() != ""
            ]
            if len(cells) == len(df.columns):
                df.loc[len(df)] = cells

        # Save to CSV
        df.to_csv("Symbols.csv", index=False)
        print("S&P 500 symbols successfully scraped and saved to Symbols.csv.")


# Example usage:
# scraper = SP500_Symbol_checker("AAPL", 0)
# scraper.Symbols_df_maker()
# print(scraper.Symbols_check())
# print(scraper.Symbols_random_gen())
