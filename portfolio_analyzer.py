"""
Portfolio Analysis Core Module
Handles: optimization, Monte Carlo, stress tests, backtests
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Result of portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    stock_tickers: List[str]


def get_returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns from price data"""
    return np.log(prices / prices.shift(1)).dropna()


def optimize_portfolio(
    returns: pd.DataFrame,
    risk_level: str  # 'conservative', 'moderate', 'aggressive'
) -> PortfolioResult:
    """
    Optimize portfolio for given risk level using mean-variance optimization.
    Conservative: minimum variance
    Moderate: max Sharpe with moderate risk
    Aggressive: maximum Sharpe ratio
    """
    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized

    def neg_sharpe(weights):
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - 0.02) / port_vol if port_vol > 0 else 0

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(weights):
        return np.sum(mean_returns * weights)

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)

    if risk_level == 'conservative':
        result = minimize(portfolio_volatility, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
    else:
        result = minimize(neg_sharpe, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)

    weights = result.x
    ret = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    sharpe = (ret - 0.02) / vol if vol > 0 else 0

    # For moderate, we might want to scale between conservative and aggressive
    if risk_level == 'moderate':
        # Get conservative weights
        res_cons = minimize(portfolio_volatility, init_guess, method='SLSQP',
                           bounds=bounds, constraints=constraints)
        cons_weights = res_cons.x
        # Blend 50% conservative, 50% aggressive
        weights = 0.5 * weights + 0.5 * cons_weights
        weights = weights / weights.sum()
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        sharpe = (ret - 0.02) / vol if vol > 0 else 0

    return PortfolioResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        stock_tickers=list(returns.columns)
    )


def monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: np.ndarray,
    principal: float,
    years: int = 3,
    n_simulations: int = 1000
) -> Tuple[np.ndarray, Dict]:
    """
    Monte Carlo simulation of portfolio value over time.
    Returns: (simulation paths, stats dict)
    """
    mean_ret = returns.mean().values
    cov = returns.cov().values
    n_days = int(years * 252)
    daily_mean = np.dot(weights, mean_ret)
    port_var = np.dot(weights.T, np.dot(cov, weights))
    daily_vol = np.sqrt(port_var)

    paths = np.zeros((n_simulations, n_days + 1))
    paths[:, 0] = principal
    np.random.seed(42)

    for i in range(n_simulations):
        daily_returns = np.random.normal(daily_mean, daily_vol, n_days)
        for t in range(1, n_days + 1):
            paths[i, t] = paths[i, t-1] * (1 + daily_returns[t-1])

    final_values = paths[:, -1]
    stats = {
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'std': np.std(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_25': np.percentile(final_values, 25),
        'percentile_75': np.percentile(final_values, 75),
        'percentile_95': np.percentile(final_values, 95),
        'min': np.min(final_values),
        'max': np.max(final_values),
    }
    return paths, stats


# Historical crisis periods (approximate start dates and typical drawdowns)
CRISIS_PERIODS = {
    '2008金融海嘯': ('2008-09-01', '2009-03-01'),
    '2011歐債危機': ('2011-07-01', '2011-10-01'),
    '2015中國股災': ('2015-08-01', '2015-09-01'),
    '2018貿易戰': ('2018-10-01', '2018-12-01'),
    '2020疫情崩盤': ('2020-02-01', '2020-03-01'),
    '2022升息風暴': ('2022-01-01', '2022-10-01'),
}


def stress_test(
    prices: pd.DataFrame,
    weights: np.ndarray,
    principal: float
) -> Dict[str, Dict]:
    """
    Stress test: simulate portfolio performance during historical crises.
    """
    returns = get_returns_matrix(prices)
    results = {}

    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        try:
            period_returns = returns.loc[start:end]
            if len(period_returns) < 2:
                continue
            port_returns = period_returns.values @ weights
            cumulative = np.prod(1 + port_returns) - 1
            final_value = principal * (1 + cumulative)
            loss = principal - final_value
            results[crisis_name] = {
                'start': start,
                'end': end,
                'return_pct': cumulative * 100,
                'final_value': final_value,
                'loss': loss,
                'drawdown_pct': -cumulative * 100 if cumulative < 0 else 0
            }
        except (KeyError, IndexError):
            continue

    return results


def backtest(
    prices: pd.DataFrame,
    weights: np.ndarray,
    principal: float,
    years_ago: int = 5
) -> Dict:
    """
    Backtest: if bought N years ago with given allocation, what would value be now?
    Uses available data (may be less than years_ago if history is shorter).
    """
    returns = get_returns_matrix(prices)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years_ago)
    period_returns = returns[returns.index >= cutoff]

    if len(period_returns) < 2:
        return {'error': '歷史資料不足，請增加歷史年數', 'final_value': principal}

    port_returns = period_returns.values @ weights
    cumulative_return = np.prod(1 + port_returns) - 1
    final_value = principal * (1 + cumulative_return)
    actual_years = (period_returns.index[-1] - period_returns.index[0]).days / 365.25

    return {
        'start_date': str(period_returns.index[0].date()),
        'end_date': str(period_returns.index[-1].date()),
        'years': round(actual_years, 1),
        'cumulative_return_pct': cumulative_return * 100,
        'final_value': final_value,
        'profit': final_value - principal,
        'cagr': (final_value / principal) ** (1 / max(actual_years, 0.1)) - 1 if actual_years > 0 else 0
    }


def project_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    years: int
) -> Dict[str, float]:
    """
    Project expected returns for 1-3 years (simple compound growth).
    """
    mean_ret = returns.mean().values
    port_daily_return = np.dot(weights, mean_ret)
    port_annual_return = port_daily_return * 252

    projections = {}
    for y in [1, 2, 3]:
        projected_return = (1 + port_annual_return) ** y - 1
        projections[f'{y}年'] = projected_return * 100

    return projections
