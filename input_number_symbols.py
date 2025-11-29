import sys
from typing import Set, Union
from SP500_Symbol_checker import SP500_Symbol_checker

# ==============================================================================
# 🎯 Module: input_number_symbols.py
# Purpose: Handles all user input related to the quantity and symbols of stocks,
# ensuring the inputs are valid integers and recognized S&P 500 tickers.
# ==============================================================================


def get_integer_input(prompt: str) -> int:
    """
    Safely gets an integer input from the user, handling non-integer entries.
    It loops indefinitely until a valid integer is provided.
    """
    while True:
        user_input = input(prompt)
        try:
            # Attempt to convert the input string to an integer.
            return int(user_input)
        except ValueError:
            # Handle the case where input is not a whole number.
            print("❌ The entry is not a valid integer! Please try again.", flush=True)


def get_valid_symbols(num_stocks: int) -> Union[Set[str], Set]:
    """
    Prompts the user for stock symbols or generates a random set for testing.
    All user-entered symbols are validated against the S&P 500 list
    using the SP500_Symbol_checker class.

    Args:
        num_stocks (int): The required number of stock symbols.

    Returns:
        set: A set of validated, unique, uppercase stock tickers.
             Returns an empty set on failure or maximum retries.
    """
    test_input = input(
        f"If you want to test the program, enter 'Y' to use a random list of {num_stocks} stocks. If not, enter 'N': "
    )

    if test_input.upper() == "Y":
        # Mode 1: Random Generation for Testing
        print(f"\n🍪 Generating {num_stocks} random S&P 500 symbols...", flush=True)
        # Initializes the checker with (Symbol=None, Number=num_stocks)
        return SP500_Symbol_checker(None, num_stocks).Symbols_random_gen()
    else:
        # Mode 2: User-Entered Validation
        print("\nPlease enter the symbols of stocks you want to consider.", flush=True)
        set_stocks = set()

        # Loop for collecting the required number of unique symbols
        for i in range(num_stocks):
            # Loop for retry logic (3 attempts per symbol)
            for attempt in range(1, 4):
                temp_symbol = input(f"Please enter the {i + 1}th symbol:").strip()

                # Initialize the checker for validation (Symbol=temp_symbol, Number=0)
                checker = SP500_Symbol_checker(temp_symbol, 0)

                # Check if the symbol is valid (e.g., exists in S&P 500 CSV)
                if checker.Symbols_check():
                    # Check for duplicates
                    if temp_symbol.upper() not in set_stocks:
                        set_stocks.add(temp_symbol.upper())
                        break  # Symbol is valid and unique, move to the next stock
                    else:
                        print(
                            f"❌ '{temp_symbol.upper()}' is a repeated symbol! Enter new one.",
                            flush=True,
                        )

                else:
                    # Symbol is invalid (not found in the S&P 500 list)
                    print(
                        f"❌ '{temp_symbol.upper()}' isn't a valid symbol! Try again.",
                        flush=True,
                    )

                # Check if max retries reached
                if attempt == 3:
                    print(
                        "🛑 Maximum retries reached. Please check your symbols and run again.",
                        flush=True,
                    )
                    return (
                        set()
                    )  # Return empty set to signal failure in the main program

            # If the retry loop was exited due to failure, the main loop should also stop.
            if not set_stocks and num_stocks > 0:
                sys.exit()

        return set_stocks


if __name__ == "__main__":
    # --- Example Execution Block ---
    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Get the set of stock symbols (either random test or user-entered validated)
    Set_stocks = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        sys.exit()

    # Convert set to a list for yfinance download
    List_stocks = list(Set_stocks)
    print(f"\n**Selected Stocks:** {List_stocks}\n")
