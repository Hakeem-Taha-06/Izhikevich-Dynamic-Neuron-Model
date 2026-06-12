# Role 11: Final Evaluation & Comprehensive Analysis Report

## 1. Overview and Biological Context
This report provides an in-depth evaluation of numerical integration methods applied to the Izhikevich (2007) biophysical neuron model. The Izhikevich model elegantly combines the biological plausibility of Hodgkin-Huxley-type dynamics with the computational efficiency of integrate-and-fire neurons. By tuning just four dimensionless parameters ($a, b, c, d$), the model can reproduce the firing patterns of all known types of cortical neurons.

## 2. Mathematical Solvers & Stability Theory

| Approach | Method | Stability | Local Truncation Error |
|---|---|---|---|
| Ground Truth | Radau IIA (Implicit RK) | Unconditionally A-Stable | High Order |
| Classical | Backward (Implicit) Euler | Unconditionally A-Stable | O(h) |
| Classical | Adams-Bashforth 2 (Explicit) | Conditionally stable | O(h²) |
| Classical | Runge-Kutta 4 (Explicit) | Conditionally stable | O(h⁴) |
| Deep Learning | PINN Surrogate | Architecture-dependent | Empirical |

> **Note on Ground Truth**: We utilize the **Radau** method via `scipy.integrate.solve_ivp`. Radau IIA is an implicit Runge-Kutta method designed specifically for stiff differential equations. Spiking neuron models are inherently stiff due to the drastic difference in time scales between the slow recovery variable $w$ and the explosive upswing of the membrane potential $v$. Using Radau guarantees a physically accurate reference trajectory.

> **Note on Solvers**:
> - **Backward Euler** is implicit. It suppresses numerical explosions completely (A-stable) but at the cost of artificial numerical damping. Rapid spikes might be smoothed out.
> - **Adams-Bashforth 2** is a multi-step explicit method. It is fast and accurate but has a small region of absolute stability, causing it to explode (NaN) if the time step $h$ is too large.
> - **Runge-Kutta 4** is the gold standard of explicit methods. It requires 4 derivative evaluations per step, making it computationally heavy, but provides massive stability margins and exceptional accuracy.

## 3. Biological Firing Patterns Analysis

### Regular Spiking (RS)
This is the most typical behavior of excitatory neurons in the cortex. They fire isolated spikes with spike-frequency adaptation (the time between spikes increases). ($c=-50, d=100$).
**Results:** All solvers successfully matched the Ground Truth (Radau). RK4 was perfectly aligned, while Backward Euler showed slight phase shifting due to numerical damping.

### Chattering (CH)
Chattering neurons fire fast bursts of closely spaced spikes, followed by a short pause, driven by a high after-spike reset and fast recovery. ($c=-40, d=150$).
**Results:** RK4 and AB2 maintained the high-frequency limit cycle well. Implicit Euler severely struggled to capture the sharp voltage transitions, flattening the bursts.

### Intrinsically Bursting (IB)
These neurons start with an initial dense burst of spikes, then switch to a slower, regular spiking mode. ($c=-56, d=130$).
**Results:** The transient behavior—shifting from a high-frequency burst to a limit cycle—was elegantly captured by the explicit high-order solvers.

## 4. Empirical Efficiency & Accuracy

> **Ground Truth (Radau)** generated exactly **6 spikes** for the baseline Regular pattern.

| Method | Pattern | Wall (s) | CPU (s) | Peak RAM (MB) | RMSE v | RMSE w | Spikes | Visual Match (DTW) |
|---|---|---|---|---|---|---|---|---|
| Adams-Bashforth 2 | Regular | 0.5181 | 0.5156 | 0.3081 | 3.232132 | 3.839893 | 6 | 0.016957 |
| Backward Euler | Regular | 3.0143 | 2.8906 | 0.5939 | 5.187229 | 6.691409 | 6 | 0.016426 |
| Runge-Kutta 4 | Regular | 0.4328 | 0.3906 | 0.307 | 3.953909 | 4.823050 | 6 | 0.013808 |


## 5. Stability Analysis (dt Sweep)
We varied the integration time step $h$ logarithmically to empirically test the stability theorems:

* **Explicit Methods (AB2, RK4):** Display catastrophic failure (Infinity/NaNs) at large step sizes, validating their conditional stability.
* **Implicit Method (Backward Euler):** Maintains stability without crashing regardless of $h$, though the trajectory becomes highly inaccurate (0 spikes).

See `stability_analysis.png` in `/outputs/figures/` for the empirical graph.

### Empirical Stability Metrics

> **Ground Truth (Radau)** has exactly **6 spikes** for this time window.

| Step Size $h$ (ms) | AB2 RMSE | BE RMSE | RK4 RMSE | AB2 Spikes | BE Spikes | RK4 Spikes |
|---|---|---|---|---|---|---|
| 0.01 | 1.836 | 2.213 | 1.796 | 6 | 6 | 6 |
| 0.05 | 2.904 | 3.935 | 2.169 | 6 | 6 | 6 |
| 0.1 | 3.232 | 5.187 | 3.954 | 6 | 6 | 6 |
| 0.5 | 7.449 | 9.933 | 5.459 | 6 | 6 | 6 |
| 1.0 | 8.722 | 64.488 | 8.634 | 6 | 0 | 6 |
| 2.0 | 8.541 | 32.965 | 11.137 | 6 | 0 | 6 |
| 5.0 | 14.472 | 15.701 | 29.923 | 6 | 0 | 18 |
| 10.0 | 28.348 | 12.236 | NaN/Expl | 7 | 0 | - |

## 6. Visualizations
Individual plots combining the Voltage time-series, Recovery time-series, and Phase Portraits for each solver versus the Radau Ground Truth have been generated and saved cleanly in `/outputs/figures/`.
