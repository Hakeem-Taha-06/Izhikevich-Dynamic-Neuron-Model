# Role 5: Implicit Solver Developer (Backward Euler) — Documentation

## What You Must Document Here

### 1. Method Description
- Explain the Backward Euler (implicit) integration formula.
- Describe **why** it requires solving a nonlinear system at each time step.
- State that `scipy.optimize.fsolve` is used for the internal root-finding loop.

### 2. Complexity Analysis Table
| Metric | Value | Explanation |
|--------|-------|-------------|
| Time Complexity  | O(?) | Cost per step includes the iterative root-finding (fsolve calls) |
| Space Complexity | O(?) | Storage for state arrays and Jacobian approximations |

### 3. Unconditional Stability Proof
- Explain what "unconditionally stable" means for this method (A-stable).
- Show evidence: run the solver with the **same large `dt`** that crashes RK4 (from Role 4's report) and confirm it still produces valid results.

| Time-Step `dt` (ms) | RK4 Result | Backward Euler Result |
|----------------------|------------|----------------------|
| (RK4 breaking dt)    | ❌ NaN     | ✅ Stable (describe output) |

### 4. Trade-off Discussion
- Discuss the accuracy vs. stability trade-off compared to explicit methods.
- Note: Backward Euler is only 1st-order accurate (O(h)), so it may produce a smoother but less precise trajectory.

---

> **Deliver to:** Role 1 (Team Leader) — Complexity analysis and the stability comparison notes.  
> **Deliver to:** Roles 9 & 10 — The working `solve_implicit_backward_euler()` function.
