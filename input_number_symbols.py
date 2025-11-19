import yfinance as yf
from SP500_Symbol_checker import SP500_Symbol_checker


# --- UTILITY FUNCTIONS FOR INPUT/VALIDATION ---


def get_integer_input(prompt: str) -> int:
    """
    Safely gets an integer input from the user, handling non-integer entries.
    """
    while True:
        user_input = input(prompt)
        try:
            return int(user_input)
        except ValueError:
            print("❌ The entry is not a valid integer! Please try again.")


def get_valid_symbols(num_stocks: int) -> set:
    """
    Prompts the user for symbols or generates a random set for testing,
    and validates the entered symbols using SP500_Symbol_checker.
    """
    test_input = input(
        f"If you want to test the program, enter 'Y' to use a random list of {num_stocks} stocks. If not, enter 'N': "
    )

    if test_input.upper() == "Y":
        # Generate a random set of symbols for testing purposes
        print(f"\n🍪 Generating {num_stocks} random S&P 500 symbols...")
        return SP500_Symbol_checker(None, num_stocks).Symbols_random_gen()
    else:
        print("\nPlease enter the symbols of stocks you want to consider.")
        set_stocks = set()
        for i in range(num_stocks):
            # Inner loop for retries on invalid symbol entry
            for attempt in range(1, 4):
                temp_symbol = input(f"Please enter the {i + 1}th symbol:").strip()

                # Check if the symbol is valid using the custom checker class
                if SP500_Symbol_checker(temp_symbol, 0).Symbols_check():
                    set_stocks.add(temp_symbol.upper())
                    break  # Move to the next symbol
                else:
                    print(
                        f"❌ '{temp_symbol.upper()}' isn't a valid symbol! Try again."
                    )

                if attempt == 3:
                    print(
                        "🛑 Maximum retries reached. Please check your symbols and run again."
                    )
                    return (
                        set()
                    )  # Return empty set to halt the program (or raise an error)

            # If the set is empty (due to an error), break the outer loop
            if not set_stocks and num_stocks > 0:
                break

        return set_stocks


if __name__ == "__main__":
    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Get the set of stock symbols (either random test or user-entered validated)
    Set_stocks = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("Execution aborted due to symbol entry error.")
        exit()

    # Convert set to a list for yfinance download
    List_stocks = list(Set_stocks)
    print(f"\n**Selected Stocks:** {List_stocks}\n")
