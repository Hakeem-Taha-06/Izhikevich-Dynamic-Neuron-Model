# Role 4: Explicit Solver Developer (RK4) — Documentation

## What You Must Document Here

### 1. Method Description
- Explain the classical 4th-Order Runge-Kutta algorithm step-by-step (k1, k2, k3, k4 stages).
- State the order of accuracy: O(h⁴) per step, O(h⁴) global error.

### 2. Complexity Analysis Table
| Metric | Value | Explanation |
|--------|-------|-------------|
| Time Complexity  | O(?) | Number of function evaluations per step × total steps |
| Space Complexity | O(?) | Storage for state arrays and intermediate k-vectors |

### 3. Stability Breakdown Report
This is **critical** — you must experimentally find the exact time-step `h` where RK4 diverges.

| Time-Step `dt` (ms) | Result | Notes |
|----------------------|--------|-------|
| 0.01                 | ✅ Stable | Matches baseline closely |
| 0.05                 | ✅ Stable | ... |
| ...                  | ...    | ... |
| ???                  | ❌ NaN/Diverge | **Breaking point** |

Report the **exact threshold** `dt` value where the method first produces NaN or divergent voltages.

### 4. Accuracy vs. Baseline
- Compare your RK4 trajectory against the LSODA baseline at `dt = 0.01`.
- Report the RMSE for V, m, h, n (you will get official numbers from Role 9, but do a self-check here).

---

> **Deliver to:** Role 1 (Team Leader) — Complexity analysis notes and the stability breaking point report.  
> **Deliver to:** Roles 9 & 10 — The working `solve_explicit_rk4()` function.
