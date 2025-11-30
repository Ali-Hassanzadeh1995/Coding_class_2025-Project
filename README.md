# 📈 Quantitative Stock Investment Analysis Tool

A Python-based project designed for financial analysis, offering detailed evaluation for individual stock performance and multi-asset portfolio optimization based on **Modern Portfolio Theory (MPT)**.

## ✨ Features

* **Flexible Data Acquisition:** Retrieves historical stock data from **Yahoo Finance** based on user-defined dates and intervals (`1d`, `1wk`, `1mo`, etc.).
* **Symbol Validation:** Robustly checks user-inputted stock tickers against a dynamically scraped list of **S&P 500 constituents** (`SP500_Symbol_checker.py`).
* **Automated Time/Interval Setup:** Automatically determines the correct **annualization coefficient** (e.g., 252 for daily data) based on the user-selected interval, ensuring metric accuracy (`interval.py`).
* **Fundamental Metrics:** Calculates and visualizes essential financial metrics (returns, volatility, Sharpe Ratio).
* **Portfolio Optimization (Markowitz):** Implements the **Sequential Least Squares Programming (SLSQP)** method to determine the **Optimal Weight Vector ($\mathbf{W}$)** that maximizes the **Sharpe Ratio**—known as the **Tangency Portfolio**.

---

## 💻 Program Logic and Flow Chart (Corrected)

The core program logic is bifurcated based on the user's selected **Analysis Mode** in the initial step.

### **General Flow**

1.  **Select Mode:** User chooses `[1] Individual Stock Analysis` or `[2] Portfolio Optimization`.
2.  **Input N:** User enters the desired **Number of Stocks (N)**. (Note: Mode 2 requires $N \geq 2$ for portfolio construction).
3.  **Data Collection:**
    * **Symbols:** Validated S&P 500 Ticker(s) are provided.
    * **Dates:** Valid Start and End Dates are entered and validated against the common minimum history date for all selected stocks.
    * **Interval/Coefficient:** The data interval (e.g., '1d', '1mo') and the corresponding **annualization coefficient** (e.g., 252 for daily) are determined.
4.  **Data Download:** Historical Adjusted Close Prices are fetched using `yfinance.download()`.
5.  **Execute Mode Logic:**

### **Mode 1: Individual Stock Analysis (Number of stocks $\geq$ 1)**

| Calculation | Output |
| :--- | :--- |
| **Full-Period Metrics** | Single-value Mean Return, Volatility ($\sigma$), and Annualized Sharpe Ratio for each stock, displayed as a **Bar Chart** for comparison. |
| **Rolling Time Frame Metrics** | Calculates and plots the **time-series trend** of the Rolling Mean, Volatility, and Annualized Sharpe Ratio over a user-defined window and step. |

### **Mode 2: Portfolio Optimization (Number of stocks $\geq$ 2)**

| Calculation | Output |
| :--- | :--- |
| **Current Portfolio Metrics** | Calculates the Annualized Volatility and Sharpe Ratio for the **Initial Portfolio** (equal-weighted or custom weights). Plots the Portfolio Value over time. |
| **Optimization (MPT)** | Uses the $\mathbf{W}$ that maximizes the Sharpe Ratio: |
| **Optimal Weight Vector ($\mathbf{W}$)** | The specific percentage allocation for each stock in the Tangency Portfolio. |
| **Optimal Sharpe Ratio** | The maximum risk-adjusted return achievable with the given assets. |
| **Optimal Return & Volatility** | The expected annual return and risk level of the Optimal Portfolio. |

---

## 🧮 Formulae & Financial Context

### Key Financial Definitions

| Metric | Formula / Concept | Purpose | Context/Threshold |
| :--- | :--- | :--- | :--- |
| **Simple Return** | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Measures the percentage change in value from one period to the next. | Used directly for portfolio value calculation. |
| **Sharpe Ratio** | $SR = \frac{E[R_p] - R_f}{\sigma_p}$ | Measures the **excess return** (above the risk-free rate, $R_f$) per unit of **total risk** ($\sigma_p$). | **Threshold:** Generally, an $SR \geq 1$ is considered good, $\geq 2$ is very good, and $\geq 3$ is excellent. |
| **Annualization Coefficient ($C$)** | $C = 252$ (Daily), $52$ (Weekly), or $12$ (Monthly). | A constant used to convert periodic metrics into an annualized figure for proper comparison. | Volatility is annualized by $\times \sqrt{C}$, and mean returns by $\times C$. |
| **Annualized Volatility** | $\sigma_p^{Annual} = \sigma_p^{Period} \times \sqrt{C}$ | The standard deviation of returns projected over a full year, the measure for investment risk. | A higher $\sigma$ means a higher probability of price fluctuation (risk). |
| **Optimal Weight Vector ($\mathbf{W}$)** | $\mathbf{W}$ that maximizes the $SR$ (Minimize $-SR$) subject to $\sum w_i = 1$. | The asset allocation vector for the **Tangency Portfolio** on the Efficient Frontier. | The sum of all optimal weights must be $1.0$ (or $100\%$ total investment). |

---

## 🔗 Streamlit App Link

You can view and interact with the live version of this application here:

[https://ali-project-coding-2025.streamlit.app/](https://ali-project-coding-2025.streamlit.app/)
