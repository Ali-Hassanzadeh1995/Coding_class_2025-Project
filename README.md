# 📈 Quantitative Stock Investment Analysis Tool

A Python-based project designed for financial analysis, offering detailed evaluation for single stock performance and multi-asset portfolio optimization based on Modern Portfolio Theory (MPT).

## ✨ Features

* **Flexible Data Acquisition:** Retrieves historical data based on user-defined dates and intervals.
* **Fundamental Metrics:** Calculates and visualizes essential financial metrics (returns, volatility).
* **Portfolio Optimization:** Implements the Markowitz Mean-Variance optimization to determine optimal asset allocation for maximum risk-adjusted return (Sharpe Ratio).

---

## 💻 Program Logic and Flow Chart

The program flow is structured around the number of assets the user wishes to analyze.

```text
Stocks invest 
	|
	|____ 1. Initial Input: Number of Stocks (N) 
	|
	|____ 2. Common Inputs (Apply to both N=1 and N>1)
	|			|
	|			|____ Stock Ticker Symbol(s) (Input: List of N symbols, e.g., ['AAPL', 'MSFT'])
	|			|____ start-date (e.g., '2020-01-01')
	|			|____ end-date (e.g., '2024-12-31')
	|			|____ interval (e.g., '1d' for Daily, '1wk' for Weekly)
	|			|____ time-frame (optional, e.g., for rolling calculations)
	|			|____ time-step (optional, e.g., for window size)
	|
	|---
	|
	|____ 3. First Option: Single Stock Analysis (N == 1)
	|			|
	|			|____ Core Output: 
	|			|		* simple-return (Data-Frame & Chart)
	|			|		* log-return (Data-Frame & Chart)
	|			|		* volatility (Data-Frame & Chart)
	|			|		* log volatility (Data-Frame & Chart)
	|
	|---
	|	
	|____ 4. Second Option: Portfolio Optimization (N > 1)
				|
				|____ Core Output (for the whole portfolio): 
				|		* simple-return (Data-Frame & Chart)
				|		* log-return (Data-Frame & Chart)
				|		* volatility (Data-Frame & Chart)
				|		* log volatility (Data-Frame & Chart)
				|
				|____ Optimization Output (Markowitz / MPT):
				|		* **Sharpe-ratio** (A single risk-adjusted return value for the optimal portfolio)
				|		* **Optimal Weight Vector (W)**: A list/vector of N weights, $\mathbf{w} = (w_1, w_2, ..., w_N)$, where $\sum w_i = 1$, representing the 	optimal capital allocation for the maximum Sharpe Ratio.
```
---
## 🧮 Formulae

### Key Financial Definitions

| Metric | Formula / Concept | Purpose |
| :--- | :--- | :--- |
| **Simple Return** | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Measures the percentage change in value from one period to the next. |
| **Log Return** | $r_t = \ln(\frac{P_t}{P_{t-1}})$ | Used in academic finance for easier calculation of multi-period returns. |
| **Volatility** | Standard Deviation ($\sigma$) of Returns | Measures the dispersion of returns, used as a proxy for asset risk. |
| **Log Volatility** | Standard Deviation ($\sigma$) of Log Returns | Measures how much log returns fluctuate; mathematically cleaner risk measure used inquantitative finance. |
| **Sharpe Ratio** | $SR = \frac{E[R_p] - R_f}{\sigma_p}$ | Measures the excess return (above the risk-free rate, $R_f$) per unit of risk ($\sigma_p$). |
| **Optimal Weight Vector** | $\mathbf{W}$ that maximizes the Sharpe Ratio | The percentage allocation for each stock that yields the best risk-adjusted return ($\sum w_i = 1$). |



