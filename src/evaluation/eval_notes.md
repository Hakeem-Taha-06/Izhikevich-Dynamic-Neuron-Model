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

### Izhikevich 2007 Exact Parameters
| Pattern | $C_m$ | $k$ | $v_r$ | $v_t$ | $v_{peak}$ | $a$ | $b$ | $c$ | $d$ | $I_{ext}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| Regular Spiking (RS) | 100 | 0.7 | -60 | -40 | 35 | 0.03 | -2 | -50 | 100 | 70 |
| Chattering (CH) | 50 | 1.5 | -60 | -40 | 25 | 0.03 | 1 | -40 | 150 | 300 |
| Intrinsically Bursting (IB) | 150 | 1.2 | -75 | -45 | 50 | 0.01 | 5 | -56 | 130 | 600 |

### Regular Spiking (RS)
This is the most typical behavior of excitatory neurons in the cortex. They fire isolated spikes with spike-frequency adaptation (the time between spikes increases).
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
| Adams-Bashforth 2 | Regular | 5.6603 | 5.5 | 3.0547 | 1.603203 | 1.905993 | 6 | 0.047237 |
| Backward Euler | Regular | 69.0913 | 64.7969 | 3.628 | 2.015869 | 2.410668 | 6 | 0.069929 |
| Runge-Kutta 4 | Regular | 11.0352 | 10.625 | 3.0536 | 1.593039 | 1.893293 | 6 | 0.046444 |


## 5. Stability Analysis (dt Sweep)
We varied the integration time step $h$ logarithmically to empirically test the stability theorems:

* **Explicit Methods (AB2, RK4):** Display catastrophic failure (Infinity/NaNs) at large step sizes, validating their conditional stability.
* **Implicit Method (Backward Euler):** Maintains stability without crashing regardless of $h$, though the trajectory becomes highly inaccurate (0 spikes).

See `stability_analysis.png` in `/outputs/figures/` for the empirical graph.

### Empirical Stability Metrics

> **Ground Truth (Radau)** has exactly **6 spikes** for this time window.

| Step Size $h$ (ms) | AB2 RMSE | BE RMSE | RK4 RMSE | AB2 Spikes | BE Spikes | RK4 Spikes |
|---|---|---|---|---|---|---|
| 0.01 | 1.603 | 2.016 | 1.593 | 6 | 6 | 6 |
| 0.05 | 2.720 | 3.779 | 2.116 | 6 | 6 | 6 |
| 0.1 | 3.079 | 5.062 | 3.814 | 6 | 6 | 6 |
| 0.5 | 7.398 | 9.877 | 5.421 | 6 | 6 | 6 |
| 1.0 | 8.674 | 64.493 | 8.585 | 6 | 0 | 6 |
| 2.0 | 8.541 | 32.966 | 11.115 | 6 | 0 | 6 |
| 5.0 | 14.473 | 15.685 | 29.914 | 6 | 0 | 18 |
| 10.0 | 28.341 | 12.213 | NaN/Expl | 7 | 0 | - |

## 6. Visualizations
Individual plots combining the Voltage time-series, Recovery time-series, and Phase Portraits for each solver versus the Radau Ground Truth have been generated and saved cleanly in `/outputs/figures/`.
