# Portfolio Construction Under Risk Constraints

Use this skill when converting forecasts into tradable target weights under leverage and concentration constraints.

## 1) Baseline Optimizer Pattern

A standard risk-aware form is:

`w_raw = (1/lambda) * inv(Sigma) * alpha`

Then enforce portfolio-level and asset-level constraints.

## 2) Constraint Application Order

Recommended order:

1. clip each asset to position bounds,
2. compute gross leverage,
3. if gross exceeds target, scale down proportionally.

Avoid scaling up when gross is below target unless strategy design explicitly requires it.

## 3) From Weights to Shares

Use current equity and reference prices to map to shares.

Key implementation details:

- preserve asset ordering consistently,
- apply lot rounding only at share stage,
- compute residual as `target - current`.

## 4) Multi-Day Consistency

In rolling simulations:

- today’s post-trade shares become tomorrow’s current shares,
- any financing effects (borrow, funding) must affect cash before next-day optimization.

## 5) Common Pitfalls

- mismatched covariance row/column order vs alpha order,
- computing equity with stale prices,
- mixing pre-trade and post-trade exposures when checking leverage.
