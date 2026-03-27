# Execution Slicing Under Liquidity Constraints

Use this guide when you need to transform a parent order into time-sliced child orders with participation and lot-size constraints.

## 1) State Variables You Should Track

For each asset:

- parent residual shares (signed),
- remaining market volume in future slices,
- per-slice participation cap in shares,
- executed quantity cumulative sum.

Recompute residual after every slice; do not pre-commit static children without updating state.

Recommended minimal state record per slice:

- `residual_before`,
- `proposed_child_raw`,
- `child_after_rounding`,
- `child_after_cap`,
- `residual_after`.

## 2) Robust Child Quantity Construction

A practical sequence:

1. compute unconstrained child from residual times a schedule ratio,
2. round to lot size,
3. apply sign-consistent cap and residual clipping,
4. enforce a minimum one-lot child when residual is meaningful but rounding collapses to zero and liquidity allows it.

This avoids ending the schedule with large unfilled tails caused only by repeated rounding-to-zero.

Implementation tip:

- clip in this order: lot rounding -> cap -> residual magnitude -> final lot rounding.
- always preserve sign after each clip.

## 3) Cap and Participation Safety

Participation is normally:

\[
\text{participation} = |q| / \max(V, 1)
\]

Even if code uses integer arithmetic for cap shares, always log and check realized participation explicitly. This catches accidental unit mistakes (shares vs lots) and sign bugs.

Numerical safety:

- use `max(volume, 1)` in denominator to avoid zero-division,
- still treat near-zero volume slices as effectively non-tradable.

## 4) Last-Slice Behavior

A common policy is "try to complete with available cap":

- on final slice, the proposed child can be residual itself,
- still clip to cap and lot constraints.

Do not silently assume full completion is guaranteed. Report leftover residuals explicitly.

If you support multi-day carry:

- persist leftover residual with sign and lot integrity,
- avoid resetting to zero at day boundary unless explicitly modeled.

## 5) Common Failure Modes

- Mixing signed and absolute residual in cap clipping.
- Rounding before scaling by schedule ratio.
- Using total-day volume instead of remaining volume from current slice onward.
- Forgetting to update cash/shares with sign-consistent fill quantities.

## 6) Quick Validation Checklist

- Every child quantity is an exact lot multiple.
- `abs(child_qty) <= per_slice_cap`.
- Residual sign only flips when crossing through zero.
- Sum of children plus leftover equals parent order exactly.

