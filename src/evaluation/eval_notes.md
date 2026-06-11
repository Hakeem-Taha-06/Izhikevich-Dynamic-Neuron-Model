# Role 11: Evaluation Notes — Izhikevich Neuron Model

**File:** `src/evaluation/evaluator.py`  
**Role Owner:** Role 11 — Master Evaluator & Analyst  
**Model:** Izhikevich (2007) Generalized Biophysical Neuron  
**Deadline:** Saturday, June 13th 2026, 7:00 AM

---

## 1. Evaluation Objectives

Per ROLES.md, Role 11 must:

1. **Efficiency Analysis** — Compute RMSE (vs Ground Truth) and Wall-Clock time for every method and present as a master table.
2. **Biological Pattern Testing** — Override `config.py` parameters to trigger different firing regimes (e.g., Regular Spiking → Chattering), rerun all solvers, and generate Phase Portraits and Time-Series plots.

---

## 2. Inputs

| Source | Role | Format |
|---|---|---|
| Ground Truth (Radau segmented) | Role 4 | `np.ndarray (N, 3): [Time, v, u]` |
| RK4 trajectory | Role 5 | `np.ndarray (N, 3): [Time, v, u]` |
| Backward Euler trajectory | Role 6 | `np.ndarray (N, 3): [Time, v, u]` |
| Adams-Bashforth 2 trajectory | Role 7 | `np.ndarray (N, 3): [Time, v, u]` |
| ML (PINN) trajectory | Role 10 | `np.ndarray (N, 3): [Time, v, u]` |

All inputs conform to the global interface contract: `(N, 3)` ordered `[Time, v, u]`.

---

## 3. Methodology

### 3.1 RMSE Calculation

For each method, interpolate its output onto the ground truth time axis (to handle any minor grid differences), then:

$$\text{RMSE}_v = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(v_i^{method} - v_i^{GT})^2}$$

$$\text{RMSE}_u = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(u_i^{method} - u_i^{GT})^2}$$

### 3.2 Wall-Clock Time

Each solver is run `n_runs = 3` times; the average wall-clock time is reported. For the ground truth, the single generation time is used directly.

### 3.3 Spike Alignment Note

Because each solver handles the discrete reset independently, spike times may differ by one or two `dt` steps between methods. RMSE is computed on the interpolated voltage, so small spike-timing offsets create a predictable RMSE spike. This is expected and is part of the comparison.

---

## 4. Biological Pattern Testing Parameters

| Regime | C_m | k | a | b | c | d |
|---|---|---|---|---|---|---|
| **Regular Spiking** (default) | 100 | 0.7 | 0.03 | -2.0 | -50 | 100 |
| **Chattering (CH)** | 50 | 1.5 | 0.03 | 1.0 | -40 | 150 |

*Source: Izhikevich (2007), Table 8.1*

When switching regimes, all module-level constants in the solver modules must be updated to match the new `config.py` values. The `evaluator.py` handles this by temporarily patching the config object.

---

## 5. Output Deliverables

| Deliverable | Path | Format |
|---|---|---|
| Time-series plot (Regular Spiking) | `outputs/figures/time_series_regular_spiking.png` | PNG, 150 DPI |
| Phase portrait (Regular Spiking) | `outputs/figures/phase_portrait_regular_spiking.png` | PNG, 150 DPI |
| Time-series plot (Chattering) | `outputs/figures/time_series_chattering.png` | PNG, 150 DPI |
| Phase portrait (Chattering) | `outputs/figures/phase_portrait_chattering.png` | PNG, 150 DPI |
| Master table | printed to stdout and returned as `dict` | — |

---

## 6. References

1. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press. Table 8.1.
2. Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error over the root mean square error. *Climate Research*, 30, 79–82.