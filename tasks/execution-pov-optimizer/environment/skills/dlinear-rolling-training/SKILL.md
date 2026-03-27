# Rolling DLinear for Return Forecasting

Use this skill when you need a small, transparent time-series forecaster that is retrained repeatedly in a rolling simulation.

## 1) Minimal DLinear Construction

Given a lookback return window `x`:

- compute trend using a causal moving average,
- seasonal component is `x - trend`,
- concatenate `[seasonal, trend]` as linear features.

Train a linear map with ridge regularization and use it for one-step-ahead prediction.

## 2) Rolling Training Protocol

At each day `d`:

- training data uses returns up to day `d-1` only,
- prediction targets day `d`,
- no peeking at future returns.

If history is insufficient for the configured lookback, use a safe fallback prediction (commonly 0.0).

## 3) Regularization and Stability

Ridge is useful for tiny samples and collinearity in trend/seasonal features.

Practical checks:

- matrix inversion is numerically stable,
- training MSE is finite,
- coefficient norm does not explode.

## 4) Integration with Cross-Sectional Alpha

A common pattern is additive blend:

`alpha_eff = alpha_base_eff + blend * dlinear_pred`

This lets the forecaster act as a tactical overlay while preserving structural alpha priors.

## 5) Auditability Requirements

Log enough metadata to reproduce predictions:

- day index and asset id,
- prediction value,
- train sample count,
- train loss (e.g., MSE),
- coefficient snapshot (or checkpoint file).
