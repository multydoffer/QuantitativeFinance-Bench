# Task: Multi-Day Adaptive Portfolio Construction + Execution Engine

Build a full rolling strategy engine, not a single-day calculator.

For each day in a scenario path, your solver must:
1) update expected returns and risk from that day’s state,
2) compute constrained target weights,
3) convert to target shares from current equity,
4) execute intraday with POV + market impact,
5) carry positions/cash to next day,
6) evaluate multi-day stability and tail risk at the end.

## Inputs

- `/app/data/alphas.json`: base expected returns by asset.
- `/app/data/covariance.csv`: base covariance matrix.
- `/app/data/portfolio.json`: initial cash and shares.
- `/app/data/intraday_bars.csv`: base intraday template (`slice_idx,asset,close,volume`).
- `/app/data/market_scenarios.json`:
  - array `days`, each with
    - `day_idx`
    - `asset_returns` map asset->daily close-to-close return
    - `liquidity_scale` map asset->multiplier for template volume
    - `alpha_bias` map asset->multiplier for base alpha
    - `vol_scale` scalar multiplier on covariance level.
- `/app/data/params.json` includes:
  - strategy/execution: `risk_aversion`, `max_abs_weight`, `gross_target`, `gross_limit_post_trade`, `max_participation`, `lot_size`, `half_spread_bps`, `impact_coeff`, `commission_bps`
  - DLinear: `dlinear_lookback`, `dlinear_ma_window`, `dlinear_ridge`, `dlinear_alpha_blend`
  - financing: `borrow_short_bps_annual`
  - stability: `risk_free_annual`, `rolling_window`, `stability_min_sharpe`, `stability_max_drawdown_floor`, `bootstrap_paths`, `bootstrap_horizon_days`, `bootstrap_block_days`, `tail_cvar_alpha`, `tail_loss_prob_limit`, `min_omega_ratio`.

## Rolling Day Loop (core chain)

Let day index be `d`.

### A) Price and Alpha State
- Maintain per-asset reference close. Start from template `slice 0 close`.
- Daily arrival price for day `d`:
  \[
  P^{arr}_{d,i} = P^{ref}_{d-1,i} (1 + r_{d,i})
  \]
  where \(r_{d,i}\) is scenario `asset_returns`.
- Base alpha:
  \[
  a^{base\_eff}_{d,i} = a^{base}_i \cdot alpha\_bias_{d,i}
  \]
- Fit a tiny DLinear-style predictor on each asset's historical realized daily returns up to `d-1`:
  - trend = causal moving average over `dlinear_ma_window`
  - seasonal = return window - trend
  - feature = concat(seasonal, trend)
  - **train** a ridge linear projection (penalty `dlinear_ridge`) on rolling samples, then predict next return \(\hat r^{DL}_{d,i}\)
- Final alpha used by optimizer:
  \[
  a_{d,i} = a^{base\_eff}_{d,i} + dlinear\_alpha\_blend \cdot \hat r^{DL}_{d,i}
  \]
- Effective covariance:
  \[
  \Sigma_d = \Sigma^{base} \cdot vol\_scale_d^2
  \]

### B) Target Portfolio
\[
w^{raw}_d = \frac{1}{\lambda}\Sigma_d^{-1}a_d
\]
then clip to `[-max_abs_weight, max_abs_weight]`; if gross above `gross_target`, scale down proportionally.

Current equity:
\[
E_d = cash_d + \sum_i q_{d,i}^{cur} P^{arr}_{d,i}
\]
Target shares:
\[
q_{d,i}^{tgt} = lot \cdot round\left(\frac{w_{d,i}E_d}{P^{arr}_{d,i} \cdot lot}\right)
\]
Residual order \(res_{d,i} = q_{d,i}^{tgt} - q_{d,i}^{cur}\).

### C) Intraday Execution (same logic, per day)
- Use intraday template prices scaled from arrival:
  \[
  mid_{d,i,t} = P^{arr}_{d,i} \cdot \frac{mid^{template}_{i,t}}{mid^{template}_{i,0}}
  \]
- Use scenario liquidity scale:
  \[
  vol_{d,i,t} = vol^{template}_{i,t} \cdot liquidity\_scale_{d,i}
  \]
- POV child sizing for each asset/slice:
  - cap:
    \[
    cap_{d,i,t} = lot\_size \cdot \left\lfloor \frac{vol_{d,i,t}\cdot max\_participation}{lot\_size}\right\rfloor
    \]
  - let `rem` be residual before slice `t`
  - let `remaining_volume` be sum of `vol_{d,i,u}` for `u >= t`
  - proposed child:
    \[
    child^{raw}_{d,i,t} = rem \cdot \frac{vol_{d,i,t}}{remaining\_volume}
    \]
  - round to nearest lot, then:
    - if rounded child is 0 while `abs(rem) >= lot_size` and `cap >= lot_size`, force one lot in `sign(rem)` direction
    - clip by both `cap` and `abs(rem)`
    - final child must be a lot multiple
- Fill price includes spread + impact + commission.
- Deduct cash by signed fill notional.

### D) End-of-Day Carry
- Post shares after fills become next day current shares.
- End-of-day mark uses last intraday mid for each asset.
- Apply daily short borrow cost:
  \[
  borrow\_cost_d = \sum_i \max(-q_{d,i}^{post},0)\cdot P^{close}_{d,i}\cdot \frac{borrow\_short\_bps\_annual}{10000\cdot252}
  \]
  subtract from cash.
- Portfolio value:
  \[
  V_d = cash_d^{post} + \sum_i q_{d,i}^{post} P^{close}_{d,i}
  \]
- Append \(V_d\) to equity curve.

## Final Stability / Tail Metrics

From daily returns of the generated equity curve:
- `ann_return`, `ann_vol`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `rolling_sharpe_floor_20d`, `omega_ratio`.
- `daily_cvar_alpha`: mean worst `ceil(alpha*n)` daily returns.
- Deterministic block bootstrap (seed=42):
  - horizon `bootstrap_horizon_days`, block `bootstrap_block_days`, paths `bootstrap_paths`
  - `bootstrap_prob_20d_loss`: probability horizon return < 0
  - `bootstrap_cvar_20d`: tail mean at `tail_cvar_alpha`.

`stability_pass` iff:
- `sharpe_ratio >= stability_min_sharpe`
- `max_drawdown >= stability_max_drawdown_floor`
- `omega_ratio >= min_omega_ratio`
- `bootstrap_prob_20d_loss <= tail_loss_prob_limit`

Clarification:

- The output key name must remain exactly `rolling_sharpe_floor_20d`.
- Its calculation uses the configurable `rolling_window` value from `params.json`; the `20d` suffix is part of the fixed output schema name, not a hardcoded 20-day override.

## Required Outputs

1) `/app/output/solver.py` (must support `--input-dir`, `--output-dir`, and copy itself).
- When `--output-dir` differs from the running script directory, `solver.py` written there must be byte-for-byte identical to the executed script.
- When `--output-dir` is the same directory, skip self-copy onto itself safely.

2) `/app/output/results.json`
- keep original execution metrics for the **final day**:
  - `weights`, `target_shares`, `filled_shares`, `leftover_shares`, `turnover_notional`, `implementation_shortfall`, `total_cost_bps_vs_arrival`, `gross_leverage_post`, `portfolio_value_end`
- plus:
  - `stability_metrics` with keys exactly:
    - `daily_return_today`
    - `ann_return`
    - `ann_vol`
    - `sharpe_ratio`
    - `sortino_ratio`
    - `omega_ratio`
    - `max_drawdown`
    - `daily_cvar_alpha`
    - `rolling_sharpe_floor_20d`
    - `bootstrap_prob_20d_loss`
    - `bootstrap_cvar_20d`
    - `stability_pass`
  - `strategy_pnl_total`: `equity_curve[-1] - equity_curve[0]`
  - `days_simulated`
  - `dlinear_last_day_mean_abs_pred`
  - `participation_breaches`: count of fill rows in `fills.csv` across all simulated days where `participation > max_participation + 1e-12`

Clarification:

- In `results.json`, everything listed under "keep original execution metrics for the final day" is final-day-only.
- `participation_breaches` is explicitly an all-days aggregate over the full simulation, not a final-day-only field.

3) `/app/output/fills.csv`
- columns: `day_idx,slice_idx,asset,qty,price,mid,volume,participation`
- `volume` must be emitted as a whole-number numeric field with 0 decimal places

4) `/app/output/daily_ledger.csv`
- columns: `day_idx,start_equity,end_equity,borrow_cost,turnover_notional,implementation_shortfall,gross_leverage_post`.

5) `/app/output/stability_curve.csv`
- columns: `day_index,equity`
- `day_index` is 0-based contiguous index (`0..days_simulated-1`), independent of scenario `day_idx` labels.
- `equity` is a numeric value rounded to 2 decimals.

6) `/app/output/dlinear_signals.csv`
- columns: `day_idx,asset,pred_return,alpha_eff,train_samples,train_mse`

7) `/app/output/dlinear_weights.json`
- one checkpoint per day/asset with trained coefficient vector:
```json
{
  "checkpoints": [
    {
      "day_idx": <int>,
      "asset": "<str>",
      "coef": [<float>, ...],
      "lookback": <int>
    }
  ]
}
```

8) `/app/output/summary.json`
- `no_participation_breach`
- `shares_reconcile`
- `leverage_within_limit`
- `stability_pass`
- `all_checks_passed`

Summary semantics:

- `no_participation_breach` iff every fill row has `participation <= max_participation + 1e-12`.
- `shares_reconcile` iff for each asset on final day:
  - `filled_shares` equals sum of `fills.csv` quantities for that asset on final day,
  - `leftover_shares = target_shares - (start_of_final_day_shares + filled_shares)`.
- `leverage_within_limit` iff `gross_leverage_post <= gross_limit_post_trade + 1e-12`.
- `stability_pass` must equal `results.json -> stability_metrics -> stability_pass`.
- `all_checks_passed` iff all four prior flags are true.

## Determinism and Rounding

- Use deterministic bootstrap seed `42`.
- `results.json`:
  - weights and all numeric fields inside `stability_metrics` (except boolean `stability_pass`): numeric values rounded to 6 decimals unless otherwise specified
  - `total_cost_bps_vs_arrival`, `gross_leverage_post`: 6 decimals
  - money fields: numeric values rounded to 2 decimals
  - `target_shares`, `filled_shares`, `leftover_shares`, `participation_breaches`, `days_simulated`: integers
  - `dlinear_last_day_mean_abs_pred`: 8 decimals
- `fills.csv`:
  - `price`, `mid`: 6 decimals
  - `volume`: 0 decimals
  - `participation`: 8 decimals
- `daily_ledger.csv`:
  - money fields: 2 decimals
  - `gross_leverage_post`: 6 decimals
