# Markowitz portfolio implementation (efficient frontier, min-variance, max-Sharpe)
# Requirements: yfinance, numpy, pandas, matplotlib, scipy
# pip install yfinance numpy pandas matplotlib scipy

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# -------------------------
# Utilities
# -------------------------
def get_price_data(tickers, start, end, adjust_close=True, progress=False):
    """Download adjusted close prices for tickers using yfinance."""
    data = yf.download(list(tickers), start=start, end=end, progress=progress)
    if adjust_close:
        # If multi-ticker, yfinance returns 'Adj Close' as column level
        if "Adj Close" in data.columns:
            prices = data["Close"].copy()
        else:
            prices = data["Close"].copy()  # fallback
    else:
        prices = data["Close"].copy()
    # Ensure DataFrame even for a single ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.columns = [str(c) for c in prices.columns]
    return prices


def annualize_returns_and_cov(prices, trading_days=252):
    """Return annualized expected returns (arithmetic mean) and covariance."""
    # daily returns
    daily_ret = prices.pct_change().dropna(how="all")
    mu_daily = daily_ret.mean()
    cov_daily = daily_ret.cov()
    mu_annual = mu_daily * trading_days
    cov_annual = cov_daily * trading_days
    return mu_annual, cov_annual, daily_ret


def portfolio_performance(weights, mu, cov):
    """Return portfolio expected return and volatility given weights."""
    w = np.array(weights)
    port_return = np.dot(w, mu)
    port_vol = np.sqrt(w.T @ cov @ w)
    return port_return, port_vol


# -------------------------
# Optimization objective functions
# -------------------------
def minimize_volatility(weights, mu, cov):
    """Objective: portfolio volatility (to minimize)."""
    return portfolio_performance(weights, mu, cov)[1]


def negative_sharpe(weights, mu, cov, risk_free_rate=0.0):
    """Objective: negative Sharpe ratio (to minimize)."""
    ret, vol = portfolio_performance(weights, mu, cov)
    # Prevent division by zero
    if vol == 0:
        return 1e6
    return -(ret - risk_free_rate) / vol


# -------------------------
# Solvers
# -------------------------
def solve_min_variance(mu, cov, allow_short=False):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    res = minimize(
        minimize_volatility,
        x0,
        args=(mu, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x, portfolio_performance(res.x, mu, cov), res


def solve_max_sharpe(mu, cov, risk_free_rate=0.0, allow_short=False):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    res = minimize(
        negative_sharpe,
        x0,
        args=(mu, cov, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x, portfolio_performance(res.x, mu, cov), res


def efficient_frontier(mu, cov, points=50, allow_short=False):
    """Construct points on the efficient frontier by targeting returns."""
    n = len(mu)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    results = []
    # target returns range between min individual asset return and max
    ret_min = min(mu)
    ret_max = max(mu)
    target_returns = np.linspace(ret_min, ret_max, points)
    for r_target in target_returns:
        x0 = np.repeat(1 / n, n)
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x, r_target=r_target: np.dot(x, mu) - r_target,
            },
        ]
        res = minimize(
            minimize_volatility,
            x0,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            w = res.x
            ret, vol = portfolio_performance(w, mu, cov)
            results.append((ret, vol, w))
    # Return arrays
    rets = np.array([r for r, v, w in results])
    vols = np.array([v for r, v, w in results])
    weights = np.array([w for r, v, w in results])
    return rets, vols, weights


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example tickers (you can change)
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "IBM"]
    start = "2020-01-01"
    end = "2025-01-01"
    risk_free_rate = 0.045  # annual

    # 1) Get prices and compute mu, cov
    prices = get_price_data(tickers, start, end, progress=False)
    mu, cov, daily_ret = annualize_returns_and_cov(prices, trading_days=252)

    # 2) Minimum-variance portfolio (long-only)
    w_minvar, (ret_minvar, vol_minvar), res_minvar = solve_min_variance(
        mu, cov, allow_short=False
    )

    # 3) Maximum Sharpe (tangent) portfolio (long-only)
    w_tan, (ret_tan, vol_tan), res_tan = solve_max_sharpe(
        mu, cov, risk_free_rate, allow_short=False
    )

    # 4) Efficient frontier (long-only)
    ef_rets, ef_vols, ef_weights = efficient_frontier(
        mu, cov, points=60, allow_short=False
    )

    # Display results
    df_weights = pd.DataFrame(
        {
            "Ticker": list(mu.index),
            "Expected Return (ann)": mu.values,
            "MinVar Weight": w_minvar,
            "Tangent Weight": w_tan,
        }
    ).set_index("Ticker")

    print("\nExpected annual returns (mu):")
    print(mu.round(4))
    print("\nMin-variance portfolio:")
    print(f"Return: {ret_minvar:.4f}, Volatility: {vol_minvar:.4f}")
    print(df_weights[["Expected Return (ann)", "MinVar Weight", "Tangent Weight"]])

    # Compute Sharpe of tangent
    sharpe_tan = (ret_tan - risk_free_rate) / vol_tan
    print(
        f"\nTangent portfolio -> Return: {ret_tan:.4f}, Vol: {vol_tan:.4f}, Sharpe: {sharpe_tan:.4f}"
    )

    # Plot efficient frontier + min-var + tangent
    plt.figure(figsize=(10, 6))
    plt.plot(ef_vols, ef_rets, "b--", lw=2, label="Efficient frontier")
    plt.scatter(
        vol_minvar,
        ret_minvar,
        marker="*",
        s=200,
        label="Minimum-variance",
        color="green",
    )
    plt.scatter(
        vol_tan, ret_tan, marker="*", s=200, label="Max-Sharpe (Tangent)", color="red"
    )
    # plot individual assets
    indiv_vol = np.sqrt(np.diag(cov))
    plt.scatter(indiv_vol, mu, marker="o", s=50, label="Individual assets")
    for i, t in enumerate(mu.index):
        plt.annotate(
            t, (indiv_vol[i], mu[i]), xytext=(6, 0), textcoords="offset points"
        )
    plt.title("Efficient Frontier (annualized)")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.legend()
    plt.grid(True)
    plt.show()
