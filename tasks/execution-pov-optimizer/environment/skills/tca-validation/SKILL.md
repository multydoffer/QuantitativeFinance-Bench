# Transaction Cost Analysis (TCA) Validation Checklist

Use this checklist to validate execution outputs and avoid sign/units mistakes in implementation shortfall style metrics.

## 1) Define a Clear Benchmark Price

Pick one benchmark (arrival, decision, or VWAP) and apply it consistently. For arrival-based shortfall:

- buy cost contribution should increase when fill > arrival,
- sell cost contribution should increase when fill < arrival.

If both sides use one formula, test signs carefully.

For arrival-benchmark implementation shortfall:

- buy cost term: `qty * (fill - arrival)` with `qty > 0`,
- sell cost term: `abs(qty) * (arrival - fill)` with `qty < 0`.

## 2) Keep Units Explicit

- shares: integer (or lot multiple),
- prices: currency per share,
- notional/cost: currency,
- bps metric:
\[
\text{bps} = 10^4 \times \frac{\text{cost}}{\text{reference notional}}
\]

Validate denominator is nonzero; return 0 when no turnover.

Keep a tiny unit table in code comments or docs:

- `shares`,
- `currency/share`,
- `currency`,
- `bps`.

## 3) Reconciliation Invariants

For each asset:

- `post_shares = start_shares + sum(fills)`,
- `leftover = target - post_shares`,
- fill aggregation in CSV equals any summary map field.

At portfolio level:

- end cash plus marked positions equals final portfolio value,
- reported turnover equals sum of absolute benchmark notional over fills.

For multi-day simulations, verify each day independently and cumulatively.

## 4) Stress Checks Worth Running

- scale all volumes down to force partial fills,
- increase spread/impact coefficients to confirm bps rises,
- flip order direction and ensure cost formulas remain sign-correct.
- apply asymmetric liquidity shocks by asset to expose hidden cross-asset assumptions.

## 5) Reporting Hygiene

- round only at final reporting boundaries, not inside state recursion,
- include enough fill-level columns (slice, qty, mid, participation, fill price) for reproducibility.

## 6) Linking Execution to Multi-Day Performance

Single-day TCA answers “did we pay too much today?”. To argue **stability** of a strategy, chain end-of-day portfolio values into a curve and compute risk-adjusted metrics (Sharpe, Sortino, drawdown, rolling Sharpe quantiles) on **daily returns** derived from that curve. Always state whether the curve is mark-to-market at the same benchmark (e.g. last close) each day.

## 7) Typical Audit Artifacts

- fill-level table with benchmark and realized price,
- per-day ledger (start equity, costs, end equity),
- stability curve and metric snapshot generated from the same run.

