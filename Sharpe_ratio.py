import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sharpe_ratio_results(
    sharpe_df: pd.DataFrame, start_date: str, end_date: str, risk_free_rate: float
) -> None:
    """
    Takes a pre-calculated DataFrame of Sharpe Ratios, displays it, and
    generates a colored bar plot for visual analysis.

    Args:
        sharpe_df (pd.DataFrame): DataFrame with Ticker as index and
                                  a column named 'Sharpe Ratio'.
        start_date (str): Start date of the historical data (YYYY-MM-DD).
        end_date (str): End date of the historical data (YYYY-MM-DD).
        risk_free_rate (float): Annual risk-free rate used in the calculation.
    """

    # Ensure the DataFrame has the correct column and is sorted for display
    if "Sharpe Ratio" not in sharpe_df.columns:
        raise ValueError("DataFrame must contain a column named 'Sharpe Ratio'.")

    sharpe_df = sharpe_df.sort_values(by="Sharpe Ratio", ascending=False)

    print("\n" + "=" * 50)
    print("      🏆 SHARPE RATIO RESULTS 🏆")
    print("=" * 50)

    # Display the result DataFrame
    print(sharpe_df.to_markdown(floatfmt=".4f"))

    # --- Plotting the Results ---

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bar chart
    # Use green for positive ratios and red for negative ratios
    colors = np.where(sharpe_df["Sharpe Ratio"] >= 0, "green", "red")
    sharpe_df["Sharpe Ratio"].plot(kind="bar", ax=ax, color=colors)

    # Add a line at y=0 for reference
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Set titles and labels
    ax.set_title(
        f"Sharpe Ratios ({start_date} - {end_date})\nRisk-Free Rate: {risk_free_rate * 100:.2f}%",
        fontsize=14,
    )
    ax.set_xlabel("Stock Ticker", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=0)

    # Display the plot
    plt.tight_layout()
    plt.show()
    #


# --- Example of creating the DataFrame (similar to your initial code) ---

# Mock Data setup (replace with your actual data generation)
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "IBM", "TSLA"]
data_start = "2022-01-01"
data_end = "2023-01-01"
rfr_rate = 0.045
data_for_df = {
    "AAPL": [0.55],
    "MSFT": [0.82],
    "NVDA": [1.35],
    "GOOGL": [0.45],
    "IBM": [-0.10],
    "TSLA": [0.25],
}
# Creating the mock DataFrame
my_sharpe_ratio_df = pd.DataFrame.from_dict(
    data_for_df, orient="index", columns=["Sharpe Ratio"]
)

# --- Calling the new function ---
if __name__ == "__main__":
    plot_sharpe_ratio_results(
        sharpe_df=my_sharpe_ratio_df,
        start_date=data_start,
        end_date=data_end,
        risk_free_rate=rfr_rate,
    )
