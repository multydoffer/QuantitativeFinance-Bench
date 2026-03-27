# Scenario-Driven Strategy Simulation Ledger

Use this skill when implementing a day-by-day simulation where market state changes over time and all accounting must reconcile.

## 1) Daily Loop Skeleton

For each day:

1. derive day state (returns, liquidity, vol scale, etc.),
2. build targets from current portfolio state,
3. execute orders and update cash/shares,
4. apply financing and carry costs,
5. mark to end-of-day prices and record ledger row.

## 2) Ledger Fields to Persist

At minimum persist:

- start equity,
- turnover notional,
- execution shortfall/cost,
- financing cost (if any),
- end equity,
- post-trade leverage.

These fields should support both performance analysis and reconciliation.

## 3) Reconciliation Invariants

- end equity equals cash plus marked positions,
- next day start state equals prior day end state,
- cumulative PnL from ledger matches equity-curve delta.

## 4) Data Discipline

- use deterministic ordering of assets and slices,
- avoid hidden state outside explicit portfolio variables,
- avoid non-deterministic randomness unless seeded and documented.

## 5) Debug Strategy

When results look wrong:

- replay a single day with verbose logs,
- verify one asset end-to-end before scaling to all assets,
- compare simulated fills against cap/participation constraints.
