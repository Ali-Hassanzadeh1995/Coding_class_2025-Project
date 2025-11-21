from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import time  # For simulated delays/robustness


class WebTableScraper:
    """
    Generic utility class for scraping a specific table from a given URL.
    This replaces the fragile, hardcoded Symbols_df_maker.
    """

    def __init__(
        self, url, table_class_name, symbol_column_name, output_filename="Symbols.csv"
    ):
        self.url = url
        self.table_class_name = table_class_name
        self.symbol_column_name = symbol_column_name
        self.output_filename = output_filename

        # Simple list of user agents to rotate (Weakness 3 - Basic mitigation)
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        ]

    def _get_random_header(self):
        """Rotates User-Agent for basic bot detection avoidance."""
        # Using a simple deterministic selection for this example
        ua = self.user_agents[int(time.time() % len(self.user_agents))]
        return {"User-Agent": ua}

    def scrape_and_save_data(self):
        """
        Scrapes the target table using pandas' read_html for robustness
        and saves it to CSV.
        """
        headers = self._get_random_header()

        print(f"Scraping data from: {self.url}...")

        try:
            # Pandas read_html is much better at parsing standard HTML tables
            # We explicitly target the table by its CSS class for robustness (Weakness 1 fix)

            # Note: We must pass the headers to the request, not to pd.read_html
            page = requests.get(self.url, headers=headers)
            page.raise_for_status()  # Check for HTTP errors (e.g., 403)

            # pd.read_html returns a list of all DataFrames found on the page
            dfs = pd.read_html(page.text, match=self.symbol_column_name)

            if not dfs:
                # Fallback: Search by CSS class if the column name match fails (less common)
                dfs = pd.read_html(page.text, attrs={"class": self.table_class_name})

            if not dfs:
                print("Error: Could not find the table on the page.")
                return

            # The target table is usually the first one found by the match criteria
            df = dfs[0]

            # Clean column names (strip whitespace and periods)
            df.columns = df.columns.str.strip().str.replace(".", "", regex=False)

            # Ensure the required symbol column exists
            if self.symbol_column_name not in df.columns:
                print(
                    f"Error: Table found but missing required column: '{self.symbol_column_name}'"
                )
                print(f"Available columns: {list(df.columns)}")
                return

            # Save the DataFrame to CSV
            # IMPORTANT: The DataFrame is now correctly loaded, fixing the "No columns" error.
            df.to_csv(self.output_filename, index=False)
            print(f"Data successfully scraped and saved to {self.output_filename}.")

        except requests.exceptions.HTTPError as e:
            print(
                f"HTTP Error {e.response.status_code}: Access Denied/Forbidden. Status: {e}. Try a delay or rotating proxies."
            )
        except Exception as e:
            print(f"An unexpected error occurred during scraping: {e}")


class SP500_Symbol_checker:
    """
    A utility class for handling S&P 500 stock symbols.
    Now uses the generic WebTableScraper.
    """

    # Configuration for S&P 500 Wikipedia table (Weakness 2 fix)
    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # The symbol column header on Wikipedia
    SP500_SYMBOL_COL = "Symbol"
    # CSS class for the content table on Wikipedia
    SP500_TABLE_CLASS = "wikitable"
    CSV_FILENAME = "Symbols.csv"

    def __init__(self, Symbol, Number) -> None:
        self.Symbol = Symbol
        self.Number = Number
        # Initialize the generic scraper
        self.scraper = WebTableScraper(
            url=self.SP500_URL,
            table_class_name=self.SP500_TABLE_CLASS,
            symbol_column_name=self.SP500_SYMBOL_COL,
            output_filename=self.CSV_FILENAME,
        )

    def Symbols_check(self):
        """Check if the given symbol exists in the S&P 500 list."""
        try:
            df = pd.read_csv(self.CSV_FILENAME)
            if self.SP500_SYMBOL_COL not in df.columns:
                print("CSV file missing 'Symbol' column. Re-scraping data...")
                self.Symbols_df_maker()
                df = pd.read_csv(self.CSV_FILENAME)

            set_symbols = set(df[self.SP500_SYMBOL_COL].astype(str).str.upper())

        except FileNotFoundError:
            print(f"{self.CSV_FILENAME} not found. Scraping data...")
            self.Symbols_df_maker()
            try:
                df = pd.read_csv(self.CSV_FILENAME)
                set_symbols = set(df[self.SP500_SYMBOL_COL].astype(str).str.upper())
            except Exception as e:
                # Critical fail if scraping failed to produce a valid file
                print(f"Error after scraping attempt: {e}")
                return False

        except Exception as e:
            # This is where your original error was being caught!
            print(f"Unexpected error while reading CSV: {e}")
            return False

        return self.Symbol.upper() in set_symbols

    def Symbols_random_gen(self):
        """Generate a random set of S&P 500 stock symbols."""
        try:
            df = pd.read_csv(self.CSV_FILENAME)
            # ... (rest of the logic is similar to Symbols_check)
            if self.SP500_SYMBOL_COL not in df.columns:
                print("CSV file missing 'Symbol' column. Re-scraping data...")
                self.Symbols_df_maker()
                df = pd.read_csv(self.CSV_FILENAME)

            set_symbols = list(df[self.SP500_SYMBOL_COL].astype(str))

        except FileNotFoundError:
            print(f"{self.CSV_FILENAME} not found. Scraping data...")
            self.Symbols_df_maker()
            try:
                df = pd.read_csv(self.CSV_FILENAME)
                set_symbols = list(df[self.SP500_SYMBOL_COL].astype(str))
            except Exception as e:
                print(f"Error after scraping attempt: {e}")
                return False

        except Exception as e:
            print(f"Unexpected error while reading CSV: {e}")
            return False

        # Handle case where the CSV might be empty despite success message
        if not set_symbols:
            print("Symbol list is empty after reading CSV.")
            return set()

        # Generate random symbols
        random_stocks = np.random.choice(set_symbols, self.Number, replace=False)
        return {str(i) for i in random_stocks}

    def Symbols_df_maker(self):
        """Triggers the generic scraper to scrape and save the S&P 500 list."""
        self.scraper.scrape_and_save_data()


# Example usage:
# Create an instance that will check for 'AAPL' and generate 5 random symbols
# The scraper will use the configuration defined in SP500_Symbol_checker
checker = SP500_Symbol_checker("AAPL", 5)

# Run the scraping process (now using the robust logic)
checker.Symbols_df_maker()

# Now the CSV should be valid, and the check/gen methods should work
print(f"\nIs AAPL in S&P 500? {checker.Symbols_check()}")
print(f"Random 5 symbols: {checker.Symbols_random_gen()}")
