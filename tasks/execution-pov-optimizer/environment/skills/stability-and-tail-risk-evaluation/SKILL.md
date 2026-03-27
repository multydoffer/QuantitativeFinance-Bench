# Stability and Tail-Risk Evaluation

Use this skill when you need to judge whether a strategy is robust, not just profitable on average.

## 1) Return Series First

Compute stability metrics from a consistent daily return series derived from an equity curve.

Do not mix mark conventions across days.

## 2) Core Metrics

Typical baseline set:

- annualized return and volatility,
- Sharpe and Sortino,
- maximum drawdown,
- rolling-window Sharpe floor (e.g., lower quantile of rolling Sharpe).

These capture central tendency and path dependence.

## 3) Tail Metrics

Useful downside measures:

- daily CVaR at chosen alpha,
- bootstrap probability of horizon loss,
- bootstrap horizon CVaR.

Use deterministic seed for reproducible benchmarking.

## 4) Bootstrap Design Notes

Block bootstrap is preferable to IID resampling when returns have serial structure.

Choose and document:

- number of paths,
- horizon length,
- block length,
- tail alpha.

## 5) Pass/Fail Policy Design

A robust gate combines several criteria (example):

- Sharpe above minimum,
- drawdown not worse than floor,
- downside probability below threshold,
- optional ratio constraints (e.g., Omega).

Avoid relying on a single metric.
