# ==============================================================================
# 📚 1. Standard Library Imports
# ==============================================================================
from datetime import date, datetime
import sys
from typing import List, Set, Tuple

# ==============================================================================
# 📚 2. Third-Party Library Imports
# ==============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from IPython.display import display

# ==============================================================================
# 📚 3. Helper Module Imports
# ==============================================================================
# Import necessary utility functions for robust user input, date validation,
# and interval/annualization coefficient determination.
try:
    from input_number_symbols import get_integer_input, get_valid_symbols
    from date_checker import get_min_valid_date, get_valid_date_input
    from interval import set_interval
except ImportError as e:
    print(f"🔴⚠️🔴 Critical Import Error: {e}")
    print(
        "Please ensure 'input_number_symbols.py', 'date_checker.py', and 'interval.py' are in the directory."
    )
    sys.exit()

# ==============================================================================
# ⚙️ 4. Global Constants (Risk-Free Rate)
# ==============================================================================
# This is the assumed annual risk-free rate, used as the benchmark return
# in the Sharpe Ratio calculation (e.g., U.S. T-bill rate).
RISK_FREE_RATE_PERCENT = (
    4.25  # e.g., 4.25 (Used for printing and internal percentage calculation)
)
RISK_FREE_RATE_DECIMAL = (
    4.25 / 100.0
)  # e.g., 0.0425 (Used for optimization in decimal form)


# ==============================================================================
# 🧮 5. Calculation Functions
# ==============================================================================


def calculate_rolling_metrics_optimized(
    df: pd.DataFrame, time_frame: int, time_step: int, coefficient: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates rolling Mean, Volatility, and Annualized Sharpe Ratio for individual assets (Mode 1).

    Args:
        df (pd.DataFrame): DataFrame of Adjusted Close prices.
        time_frame (int): The window size (number of periods) for the rolling calculation.
        time_step (int): The step size between calculations.
        coefficient (int): The annualization factor (e.g., 252 for daily data).
    """
    if time_frame > len(df):
        raise ValueError("Time frame is larger than the total number of data points.")

    # Convert Adjusted Close prices to simple periodic returns in percentage form.
    df_simple_returns = df.pct_change() * 100
    df_simple_returns = df_simple_returns.dropna()

    # Calculate rolling mean and standard deviation (volatility).
    df_rolling_mean = df_simple_returns.rolling(window=time_frame).mean()
    df_rolling_vol = df_simple_returns.rolling(window=time_frame).std()

    # Calculate the Risk-Free Rate (Rf) for the calculation period (e.g., daily Rf if data is daily).
    Rf_per_period = (1 + RISK_FREE_RATE_PERCENT / 100) ** (1 / coefficient) - 1
    Rf_per_period_perc = Rf_per_period * 100

    # Calculate the Annualized Sharpe Ratio:
    # Sharpe = (Mean_Return_Period - Rf_Period) / Volatility_Period * sqrt(coefficient)
    df_rolling_sharpe = (
        (df_rolling_mean - Rf_per_period_perc) / df_rolling_vol * np.sqrt(coefficient)
    )

    # Select only the stepped values to reduce plot clutter and computation time.
    start_index = time_frame - 1
    df_mean_stepped = df_rolling_mean.iloc[start_index::time_step].dropna(how="all")
    df_vol_stepped = df_rolling_vol.iloc[start_index::time_step].dropna(how="all")
    df_sharpe_stepped = df_rolling_sharpe.iloc[start_index::time_step].dropna(how="all")

    return (df_mean_stepped, df_vol_stepped, df_sharpe_stepped)


def calculate_sharpe_ratio(
    weights,
    mean_returns_decimal,
    cov_matrix_decimal,
    risk_free_rate_annual,
    coefficient,
):
    """
    Calculates the Annualized Sharpe Ratio for a portfolio (Mode 2 Logic).
    Sharpe Ratio (S) = (Rp - Rf) / STDp. Higher is better.
    """
    # 1. Annualized Portfolio Return (Rp): Sum of weighted mean periodic returns * coefficient
    annual_return = np.sum(mean_returns_decimal * weights) * coefficient

    # 2. Annualized Portfolio Volatility (σp): sqrt(w' * Cov * w) * sqrt(coefficient)
    annual_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix_decimal, weights))
    ) * np.sqrt(coefficient)

    if annual_volatility == 0:
        print(
            f"There is a problem! covariance matrix is a zero matrix!? or you put weight vector zerooo??!",
            flush=True,
        )
        return 0.0

    # 3. Sharpe Ratio Calculation: (Annual Return - Annual Risk-Free Rate) / Annual Volatility
    sharpe_ratio = (annual_return - risk_free_rate_annual) / annual_volatility
    return sharpe_ratio


def negative_sharpe_ratio(
    weights,
    mean_returns_decimal,
    cov_matrix_decimal,
    risk_free_rate_annual,
    coefficient,
):
    """
    Objective function for the minimization process (Markowitz Optimization).
    Minimizing the negative Sharpe Ratio is mathematically equivalent to
    MAXIMIZING the positive Sharpe Ratio, which is the goal of optimization.
    """
    if not np.isclose(np.sum(weights), 1.0):
        pass  # Note: The constraint handles this, but a check remains.
    sharpe = calculate_sharpe_ratio(
        weights,
        mean_returns_decimal,
        cov_matrix_decimal,
        risk_free_rate_annual,
        coefficient,
    )
    return -sharpe


# ==============================================================================
# 📈 6. Visualization Function
# ==============================================================================


def plot_metrics(
    df: pd.DataFrame,
    title: str,
    is_rolling: bool,
    kind: str = "line",
    time_frame: int = None,
    time_step: int = None,
    value_label: str = None,
):
    """Generates visualizations for time series or bar charts of financial metrics."""
    try:
        plt.figure(figsize=(12, 6))
        full_title = title
        x_label = "Date"
        color = None

        if is_rolling:
            full_title += f"\n(Window Size: {time_frame}, Step: {time_step})"
            x_label = "End Date of Rolling Window"

        if kind == "bar":
            # Assign colors for visual distinction in comparison charts
            if "Volatility" in title:
                color = "red"
            elif "Sharpe" in title:
                color = "blue"
            elif "Mean" in title:
                color = "green"

            ax = df.plot(
                kind="bar",
                figsize=(12, 6),
                color=color,
                alpha=0.8,
                edgecolor="black",
                legend=False,
            )
            plt.axhline(0, color="black", linewidth=0.8)  # Add zero line for reference
            x_label = "Symbols"
            plt.xticks(rotation=0)

            # Add value annotations to the top of each bar
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
        else:
            plt.plot(df)
            x_label = "Date"

        y_label = value_label if value_label else "Value"

        plt.title(full_title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        if kind == "line":
            # Place legend outside the plot area for clarity
            plt.legend(
                df.columns, title="Symbols", bbox_to_anchor=(1.05, 1), loc="upper left"
            )

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not generate plot for '{title}'. Error: {e}")


# ==============================================================================
# 🏁 7. Main Execution Block
# ==============================================================================


def execute_analysis():
    """Main function to run the financial analysis toolkit."""
    print("=========================================")
    print(" 💸 FINANCIAL ANALYSIS TOOLKIT V1.0 ")
    print("=========================================\n")

    # ---------------------------------------------------------
    # PART A: SELECT MODE
    # ---------------------------------------------------------
    print("Please select your Analysis Mode:")
    print("📚 [1] Individual Stock Analysis (Rolling Mean, Volatility, Sharpe)")
    print("📖 [2] Portfolio Optimization (Weights, Covariance, Markowitz)")

    counter = 0
    while counter < 3:
        mode = input("Enter 1 or 2: ").strip()
        if mode not in ["1", "2"]:
            counter += 1
            if counter == 3:
                print("🛑 Invalid mode selection. Exit!")
                sys.exit()
            print("🔂 Invalid mode selection. Try again!")
            continue
        break
    # ---------------------------------------------------------
    # PART B: Input Collection & Data Download
    # ---------------------------------------------------------
    print("\n--- Data Configuration ---")
    N_stocks = get_integer_input("Please enter the number of stocks: ")

    # Validation hint for Portfolio Mode
    if mode == "2" and N_stocks < 2:
        print("⚠️ Note: Portfolio optimization typically requires 2 or more stocks.")
        sys.exit()

    # Get valid symbols using the helper module logic (user input or random generation).
    Set_stocks: Set[str] = get_valid_symbols(N_stocks)

    if not Set_stocks:
        print("🔴 Execution aborted due to symbol entry error.")
        sys.exit()

    List_stocks: List[str] = list(Set_stocks)
    print(f"\n**Selected Stocks:** {List_stocks}\n")

    # Date Collection (Uses helper module for validation and min date check)
    min_common_date = get_min_valid_date(Set_stocks)
    today_date = date.today()
    S_date = get_valid_date_input(
        f"Enter Start Date (YYYY-MM-DD, min: {min_common_date}): ",
        min_date=min_common_date,
        max_date=today_date,
    )
    S_date_dt = datetime.strptime(S_date, "%Y-%m-%d").date()
    E_date = get_valid_date_input(
        f"Enter End Date (YYYY-MM-DD, min: {S_date}): ",
        min_date=S_date_dt,
        max_date=today_date,
    )
    # Determine the appropriate data interval and annualization coefficient.
    interval, coefficient = set_interval(S_date, E_date)
    print(f"\n**Data Range:** {S_date} to {E_date}, Interval: {interval}")

    # Data Download from Yahoo Finance
    print("\n⬇️ Downloading data from Yahoo Finance...")
    try:
        data = yf.download(
            List_stocks, start=S_date, end=E_date, interval=interval, auto_adjust=False
        )
        if data.empty:
            raise ValueError("No data returned.")
    except Exception as e:
        print(f"🛑 Error during data download: {e}", file=sys.stderr)
        sys.exit()

    # Extract Adjusted Close prices and drop any rows with missing data.
    DF_Adj_Close: pd.DataFrame = data["Adj Close"].copy()
    DF_Adj_Close.dropna(inplace=True)

    if DF_Adj_Close.empty or len(DF_Adj_Close) < 2:
        print("🛑 Insufficient valid data. Aborting.")
        sys.exit()

    # Initial Plot of Adjusted Close Prices
    display(DF_Adj_Close.head())
    plot_metrics(
        DF_Adj_Close,
        "Adjusted Close Price Time Series",
        is_rolling=False,
        kind="line",
        value_label="Price (Dollars)",
    )

    # ---------------------------------------------------------
    # PART C: Execute Selected Mode Logic
    # ---------------------------------------------------------

    # ==============================================================================
    # MODE 1: Individual Stock Analysis
    # ==============================================================================
    if mode == "1":
        print("\n📚 STARTING INDIVIDUAL STOCK ANALYSIS")

        # Calculate simple returns for the full period.
        DF_simple_return: pd.DataFrame = DF_Adj_Close.pct_change() * 100
        DF_simple_return = DF_simple_return.dropna()

        print("\n📊 Simple Returns (Sample):")
        display(DF_simple_return.head())

        option = (
            str(
                input(
                    "\nDo you want Full-Period results (Y) or Rolling Time Frame (N)? (Y/N): "
                )
            )
            .strip()
            .upper()
        )

        if option == "Y":
            # --- Full Period Analysis ---
            df_mean_full = DF_simple_return.mean().to_frame("Mean Return")
            df_vol_full = DF_simple_return.std().to_frame("Volatility")

            # Calculate the Risk-Free Rate (Rf) per period.
            Rf_per_period = (1 + RISK_FREE_RATE_PERCENT / 100) ** (1 / coefficient) - 1
            Rf_per_period_perc = Rf_per_period * 100

            # Calculate the Annualized Sharpe Ratio for the full period.
            df_sharpe_full = (
                (df_mean_full["Mean Return"] - Rf_per_period_perc)
                / df_vol_full["Volatility"]
                * np.sqrt(coefficient)
            )
            df_sharpe_full = df_sharpe_full.to_frame("Sharpe Ratio")

            df_results = pd.concat([df_mean_full, df_vol_full, df_sharpe_full], axis=1)
            print("\n📊 Full Period Metrics:")
            display(df_results)

            # Plotting Full Period Results
            plot_metrics(DF_simple_return, "Simple Return Time Series", False, "line")
            plot_metrics(df_vol_full, "Full Period Volatility Comparison", False, "bar")
            plot_metrics(
                df_sharpe_full, "Full Period Sharpe Ratio Comparison", False, "bar"
            )

        else:
            # --- Rolling Period Analysis ---
            print(f"\nGiven your data has **{len(DF_Adj_Close)}** periods.")

            # --- Get and Validate time_frame ---
            counter = 0
            while counter < 3:
                # Get input for time_frame
                time_frame_input = input("Enter time_frame (window size, e.g., 20): ")

                # 1. Check for empty input
                if not time_frame_input:
                    counter += 1
                    print("Input cannot be empty.")
                    continue

                # 2. Check for integer conversion
                try:
                    time_frame = int(time_frame_input)
                except ValueError as e:
                    # Handle non-integer input or validation errors
                    counter += 1
                    print(f"Invalid input: {e}. Please re-enter time_frame.")
                    continue

                # 3. Check logical constraint (must be positive)
                if time_frame <= 0:
                    counter += 1
                    print("time_frame must be a positive integer.")
                    continue

                # 4. Check logical constraint (less than total data length)
                if time_frame >= len(DF_Adj_Close):
                    counter += 1
                    print(
                        f"time_frame ({time_frame}) must be less than the total periods in the data ({len(DF_Adj_Close)})."
                    )
                    continue

                if counter == 3:
                    sys.exit()
                # If all checks pass, break the loop
                break

            # --- Get and Validate time_step ---
            counter = 0
            while counter < 3:
                # Get input for time_step
                time_step_input = input("Enter time_step (periods to step, e.g., 5): ")

                # 1. Check for empty input
                if not time_step_input:
                    counter += 1
                    print("Input cannot be empty.")
                    continue

                try:
                    # 2. Check for integer conversion
                    time_step = int(time_step_input)
                except ValueError as e:
                    # Handle non-integer input or validation errors
                    counter += 1
                    print(f"Invalid input: {e}. Please re-enter time_step.")
                    continue

                # 3. Check logical constraint (must be positive)
                if time_step <= 0:
                    counter += 1
                    print("time_step must be a positive integer.")

                # 4. Check logical constraint (less than time_frame)
                if time_step >= time_frame:
                    counter += 1
                    print(
                        f"time_step ({time_step}) must be less than time_frame ({time_frame})."
                    )
                    continue

                if counter == 3:
                    sys.exit()

                # If all checks pass, break the loop
                break

            try:
                # Call the rolling metric function to compute stepped results.
                (df_mean_roll, df_vol_roll, df_sharpe_roll) = (
                    calculate_rolling_metrics_optimized(
                        DF_Adj_Close, time_frame, time_step, coefficient
                    )
                )
                print("\n💹 Generating plots for rolling metrics...")

                # Plotting Rolling Results (time series)
                plot_metrics(
                    df_mean_roll,
                    "Rolling Simple Mean Return",
                    True,
                    "line",
                    time_frame,
                    time_step,
                    "Percent per period",
                )
                plot_metrics(
                    df_vol_roll,
                    "Rolling Volatility",
                    True,
                    "line",
                    time_frame,
                    time_step,
                    "Percent per period",
                )
                plot_metrics(
                    df_sharpe_roll,
                    "Rolling Sharpe Ratio",
                    True,
                    "line",
                    time_frame,
                    time_step,
                )

            except ValueError as e:
                print(f"\n🛑 Error in rolling calculation: {e}")
                sys.exit()

    # ==============================================================================
    # MODE 2: Portfolio Analysis
    # ==============================================================================
    elif mode == "2":
        print("\n📖 STARTING PORTFOLIO OPTIMIZATION")
        print(
            "\n⚠️ Note: Default the weight for each stock is considered one over number of stocks!!"
        )
        # 1. Weights Input
        option = (
            input("Do you want to enter a custom weight vector? (Y/N): ")
            .strip()
            .upper()
        )

        total_amount_input = input(
            f"💲 Enter the total amount of investing (default: 1000 dollars): "
        )
        try:
            total_amount = float(total_amount_input)
        except ValueError:
            total_amount = 1000.0
            print(f"Invalid amount entered. Defaulting to {total_amount:.2f} euros.")

        weight = np.zeros(N_stocks)
        if option == "Y":
            print("\n**Entering Custom Weights** (Must be non-negative)")
            counter = 0
            while not np.any(weight) and counter < 3:
                for i in range(N_stocks):
                    counter_1 = 0
                    while counter_1 < 3:
                        weight_input = input(
                            f"Enter the weight of {List_stocks[i]} (e.g., 0.1): "
                        )
                        try:
                            w = float(weight_input)
                            if w < 0:
                                print("Weight must be non-negative.")
                                continue
                            weight[i] = w
                            break
                        except ValueError:
                            print("Invalid number. Try again.")
                            counter_1 += 1
                    if counter_1 == 3:
                        sys.exit()
                if np.any(weight):
                    counter += 1
            if counter == 3:
                print("🛑 Total weight is zero. Aborting.")
                sys.exit()
        else:
            # Equal weight assignment
            weight = np.array([1.0 / N_stocks] * N_stocks)
            print("\n**Using Equal Weights.**")

        # Normalize weights to ensure they sum exactly to 1.0 (100% investment).
        sum_weight = np.sum(weight)
        normal_w = weight / sum_weight
        print(f"Normalized weights: {dict(zip(List_stocks, normal_w))}")

        # 2. Portfolio Value Calculation
        DF_simple_return: pd.DataFrame = DF_Adj_Close.pct_change() * 100
        DF_simple_return = DF_simple_return.dropna()

        # Calculate the daily portfolio return based on the weighted sum of individual asset returns.
        portfolio_daily_returns_perc: pd.Series = DF_simple_return.dot(normal_w)

        DF_Prt_value_daily = pd.DataFrame(
            index=portfolio_daily_returns_perc.index,
            columns=["The daily value of the portfolio"],
        )
        # Calculate the cumulative portfolio value over time.
        current_portfolio_value = total_amount
        for i in portfolio_daily_returns_perc.index:
            current_portfolio_value = current_portfolio_value * (
                1 + portfolio_daily_returns_perc.loc[i] / 100.0
            )
            DF_Prt_value_daily.loc[i] = current_portfolio_value

        print("\n💰 Daily Portfolio Value:")
        display(DF_Prt_value_daily)
        plot_metrics(
            DF_Prt_value_daily,
            "Portfolio Value Time Series",
            False,
            "line",
            value_label="Value (Currency)",
        )

        # 3. Metrics Calculation for Current Portfolio
        df_cov = DF_simple_return.cov()
        cov_matrix = df_cov.to_numpy()
        # Convert covariance from percentage squared to decimal squared (dividing by 100*100 = 10000).
        cov_matrix_decimal = cov_matrix / 10000.0

        # Calculate Annualized Volatility (Standard Deviation)
        volatility_annual_decimal = np.sqrt(
            np.dot(normal_w.T, np.dot(cov_matrix_decimal, normal_w))
        ) * np.sqrt(coefficient)
        volatility_annual_perc = volatility_annual_decimal * 100.0
        print(
            f"\n📉 Current Annualized Portfolio Volatility: {volatility_annual_perc:.4f}%"
        )

        # Calculate Annualized Return and Sharpe Ratio
        mean_returns_decimal = DF_simple_return.mean() / 100.0
        portfolio_return_annual_decimal = (
            np.sum(mean_returns_decimal * normal_w) * coefficient
        )

        sharpe_ratio_full = (
            portfolio_return_annual_decimal - RISK_FREE_RATE_DECIMAL
        ) / volatility_annual_decimal

        df_sharpe_full = pd.DataFrame(
            {"Portfolio": sharpe_ratio_full}, index=["Sharpe Ratio"]
        ).T
        print("\n⚖️ Full Period Portfolio Annualized Sharpe Ratio:")
        display(df_sharpe_full)
        plot_metrics(
            df_sharpe_full,
            "Current Portfolio Sharpe Ratio",
            False,
            "bar",
            value_label="Sharpe Ratio",
        )

        # 4. Markowitz Optimization
        print("\n🏦 Starting Markowitz Optimization (Maximize Sharpe Ratio)...")
        # Bounds: No short selling or leverage (weights between 0 and 1).
        bounds = tuple((0, 1) for _ in range(N_stocks))
        # Constraint: All weights must sum to 1 (full investment).
        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        initial_weights = normal_w

        # Use Sequential Least Squares Programming (SLSQP) to find the weights
        # that minimize the negative Sharpe Ratio (i.e., maximize the positive Sharpe Ratio).
        optimal_results = minimize(
            negative_sharpe_ratio,
            initial_weights,
            args=(
                mean_returns_decimal,
                cov_matrix_decimal,
                RISK_FREE_RATE_DECIMAL,
                coefficient,
            ),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if optimal_results.success:
            optimal_weights = optimal_results.x
            optimal_sharpe = -optimal_results.fun

            print("\n✅ Optimization Successful!")
            print(
                f"Optimal Annualized Sharpe Ratio (Tangency Portfolio): {optimal_sharpe:.4f}"
            )
            print("\nOptimal Weights (Decimal):")
            optimal_weights_dict = dict(zip(List_stocks, optimal_weights))
            for symbol, weight in optimal_weights_dict.items():
                print(f"  {symbol}: {weight:.4f} ({weight*100:.2f}%)")

            # Plotting Optimal Weights

            # Create a DataFrame from the optimal weights dictionary for plotting.
            df_optimal_weights = pd.DataFrame.from_dict(
                optimal_weights_dict, orient="index", columns=["Weight"]
            ).sort_values(
                by="Weight", ascending=False
            )  # Sort by weight value for better visualization
            df_optimal_weights["Weight"] = df_optimal_weights["Weight"] * 100
            plot_metrics(
                df_optimal_weights,
                "Optimal Portfolio Weights (Max Sharpe Ratio)",
                is_rolling=False,
                kind="bar",
                value_label="Weight (Decimal)",
            )

            # Calculate metrics for the optimal portfolio
            optimal_return_annual = (
                np.sum(mean_returns_decimal * optimal_weights) * coefficient
            )
            optimal_volatility_annual = np.sqrt(
                np.dot(optimal_weights.T, np.dot(cov_matrix_decimal, optimal_weights))
            ) * np.sqrt(coefficient)

            print(f"\nOptimal Annualized Return: {optimal_return_annual*100:.4f}%")
            print(
                f"Optimal Annualized Volatility: {optimal_volatility_annual*100:.4f}%"
            )
        else:
            print(f"\n❌ Optimization Failed. Status: {optimal_results.message}")


if __name__ == "__main__":
    execute_analysis()
