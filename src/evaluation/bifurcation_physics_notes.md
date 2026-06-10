# Role 10: Physics Coherency Analyst (The Crash Tester) — Documentation

## What You Must Document Here

### 1. Bifurcation Analysis Description
- Explain what a bifurcation threshold is in the Hodgkin-Huxley model context:
  the critical external current `I_ext` at which the neuron transitions from a resting (quiescent) state to repetitive spiking (limit cycle oscillations).

### 2. Sweep Configuration
| Parameter | Value |
|-----------|-------|
| `I_ext` range | 0.0 → 15.0 uA/cm² |
| Number of sweep steps | 50 (or your chosen value) |
| Simulation duration per step | 100 ms |
| Spike detection method | ? (e.g., peak counting, voltage threshold crossing) |

### 3. Bifurcation Threshold Results Table
| Method | Bifurcation Threshold `I_ext` (uA/cm²) | Notes |
|--------|----------------------------------------|-------|
| LSODA (Baseline) | ? | Reference value |
| RK4 (Role 4)     | ? | Should match baseline |
| Backward Euler (Role 5) | ? | May differ slightly due to 1st-order accuracy |
| PINN (Role 8)    | ? | Deviation indicates physics loss quality |

### 4. Bifurcation Diagram
- Generate a plot: X-axis = `I_ext`, Y-axis = steady-state max/min of `V(t)`.
- Each solver should be a separate color/line on the same diagram.
- Save to `outputs/figures/bifurcation_diagram.png`.

### 5. Physical Validity Conclusions
- Do all solvers agree on the bifurcation threshold?
- If not, explain the source of deviation (numerical accuracy, PINN approximation error, etc.).

---

> **Deliver to:** Role 1 (Team Leader) — The bifurcation diagram and threshold comparison table for the IEEE paper.
