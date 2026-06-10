# Role 9: The Evaluator (Trajectory Accuracy & Speed) — Documentation

## What You Must Document Here

### 1. Ground Truth Generation
- Describe how the baseline LSODA solver (Role 3) is called to produce the reference dataset.
- Document any randomized initial condition variations used and the random seed.
- The CSV is saved to `data/ground_truth.csv` with headers: `Time,Voltage,m,h,n`.

### 2. RMSE Metrics Table
Compute and fill in the Root Mean Squared Error for each solver against the baseline:

| Method | RMSE_V (mV) | RMSE_m | RMSE_h | RMSE_n | Execution Time (ms) |
|--------|-------------|--------|--------|--------|---------------------|
| LSODA (Baseline) | — | — | — | — | ? |
| RK4 (Role 4)     | ? | ? | ? | ? | ? |
| Backward Euler (Role 5) | ? | ? | ? | ? | ? |
| PINN (Role 8)    | ? | ? | ? | ? | ? |

### 3. Trajectory Comparison Plots
- Save a figure comparing all four solvers' Voltage `V(t)` trajectories on one plot.
- Save additional subplots for `m(t)`, `h(t)`, `n(t)` if applicable.
- All figures go to `outputs/figures/`.

### 4. Execution Time Benchmarking
- Describe how timing is measured (e.g., `time.perf_counter()` around each solver call).
- Note any caveats (e.g., PINN training time vs. inference time).

---

> **Deliver to:** Role 1 (Team Leader) — Final RMSE table and comparison plots for the IEEE paper.  
> **Deliver to:** Role 8 — The ground truth CSV (`data/ground_truth.csv`) for PINN training.
