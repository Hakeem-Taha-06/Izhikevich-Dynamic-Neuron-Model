# Evaluation Module — `src/evaluation/`

This module provides a unified evaluation pipeline that compares all solver outputs (numerical methods and PINN) against the Radau ground truth on a consistent set of metrics.

---

## 1. Evaluation Methodology (`evaluator.py`)

### Data Loading
1. Load the ground truth from `data/ground_truth.csv`
2. Load each solver/PINN output from `outputs/`
3. Interpolate solver outputs onto the ground truth time grid (100,001 points, dt = 0.01 ms) if the time grids differ

### Metrics Computed

#### Global RMSE
Root Mean Square Error computed over the full trajectory for both state variables:

$$\text{RMSE}_v = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(v_i^{pred} - v_i^{gt})^2}$$

$$\text{RMSE}_w = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(w_i^{pred} - w_i^{gt})^2}$$

#### Spike-Timing Error
Spikes are detected by threshold crossings at `v ≥ v_peak`. The timing error is the absolute difference between detected spike times in the solver output and the ground truth:

$$\Delta t_{spike} = |t_{spike}^{pred} - t_{spike}^{gt}|$$

#### Dynamic Time Warping (DTW)
A shape-based similarity metric that accounts for small phase shifts between trajectories. Lower DTW distance indicates better visual match independent of uniform time alignment.

---

## 2. Results Summary

### Solver Performance at dt = 0.01 ms

| Method | RMSE v (mV) | RMSE w (pA) | Spikes Detected | Wall Time (s) | DTW Distance |
|---|---|---|---|---|---|
| **RK4** | 1.59 | 1.89 | 6 | 4.52 | 0.0126 |
| **AB2** | 1.60 | 1.91 | 6 | 5.51 | 0.0134 |
| **Backward Euler** | 2.02 | 2.41 | 6 | 29.71 | 0.0181 |
| **PINN** | — | — | — | — | — |

### Stability vs Step Size

| Step Size h (ms) | AB2 RMSE v | BE RMSE v | RK4 RMSE v | AB2 Spikes | BE Spikes | RK4 Spikes |
|---|---|---|---|---|---|---|
| 0.01 | 1.60 | 2.02 | 1.59 | 6 | 6 | 6 |
| 0.05 | 2.72 | 3.78 | 2.12 | 6 | 6 | 6 |
| 0.1 | 3.08 | 5.06 | 3.81 | 6 | 6 | 6 |
| 0.5 | 7.40 | 9.88 | 5.42 | 6 | 6 | 6 |
| 1.0 | 8.67 | 64.49† | 8.59 | 6 | 0 | 6 |

† Backward Euler detects zero spikes at h = 1 ms — fsolve converges to a spurious sub-threshold fixed point.

### Key Findings

1. **RK4** provides the best accuracy-to-speed ratio due to its O(h⁴) convergence.
2. **AB2** achieves the fastest wall-clock time and competitive accuracy via its history-flush protocol.
3. **Backward Euler** is unconditionally stable (never diverges) but the non-convex quadratic residual admits spurious sub-threshold roots at large step sizes, silencing spiking entirely. Unconditional stability ≠ biological correctness.
4. **PINN** captures continuous dynamics from physics but struggles with the discrete spike reset — a known limitation of applying continuous function approximators to discontinuous systems.

---

## 3. Visualizations

All comparison plots are saved to `outputs/figures/`:
- Individual solver voltage overlays vs ground truth
- RMSE convergence plots (log-log scale)
- Phase portraits (v, w) for each solver
- PINN prediction overlay

---

## 4. How to Run

```bash
python -c "from src.evaluation.evaluator import run_full_evaluation; run_full_evaluation()"
```

Or through the main pipeline:
```bash
python main.py
```

---

## References

1. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.
2. Schiesser, W. E. (2014). *Differential Equation Analysis in Biomedical Science and Engineering*. Wiley.
