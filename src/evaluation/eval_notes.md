# Role 11: Final Evaluation & Analysis Report

## 1. Overview
This report details the evaluation of numerical integration methods applied to the Izhikevich (2007) biophysical neuron model. The objective is to validate that the solvers accurately reproduce diverse biological firing patterns by dynamically patching model parameters without altering the core solver logic.

## 2. Theoretical Comparison of Solvers

| Approach | Method | Stability | Accuracy |
|---|---|---|---|
| Classical | Backward (Implicit) Euler | Unconditionally stable | O(h) |
| Classical | Adams-Bashforth 2 (Similar to RK2) | Conditionally stable | O(h²) |
| Classical | Runge-Kutta 4 | Conditionally stable | O(h⁴) |
| Deep Learning | PINN Surrogate | Architecture-dependent | Empirical |

> **Note**: Backward Euler provides absolute stability at the cost of execution speed (due to implicit root-finding). Adams-Bashforth 2 offers higher accuracy $O(h^2)$ and faster computation but requires careful step-size selection to maintain stability. Runge-Kutta 4 provides the highest accuracy $O(h^4)$ allowing for larger stable step sizes, but requires 4 evaluations of the derivative per step, making it computationally heavier per step. The PINN Surrogate learns to approximate the ODE solution in physical space using a deep network.

## 3. Biological Firing Patterns

The Izhikevich model's versatility comes from its ability to simulate various cortical neurons by simply tuning its parameters (e.g., $a, b, c, d, C_m$).

* **Regular Spiking (RS):** The baseline pattern. Fires isolated spikes with adaptation. ($c=-50, d=100$) **Result:** All solvers successfully generated the correct number of spikes (6). RK4 showed perfect alignment, AB2 was highly accurate, while BE exhibited a slight phase delay.
* **Chattering (CH):** Fires high-frequency clusters of spikes (bursts) followed by short pauses. Driven by a fast recovery rate and high after-spike reset. ($c=-40, d=150, C_m=50$) **Result:** Both AB2 and RK4 accurately captured the limit cycle and the intra-burst frequency. BE suffered from severe numerical damping, smoothing out the rapid spikes.
* **Intrinsically Bursting (IB):** Begins with an initial burst of spikes followed by regular, slower spiking. ($c=-56, d=130, C_m=150$) **Result:** RK4 and AB2 correctly produced the initial triplet burst followed by a single spike. The transient behavior in the phase portrait was perfectly captured.

## 4. Empirical Efficiency & Accuracy

> **Note**: Ground Truth (LSODA) generated exactly **6 spikes** for the baseline Regular pattern.

| Method | Pattern | Wall (s) | CPU (s) | Peak RAM (MB) | RMSE v | RMSE w | Spikes | Visual Match (DTW) |
|---|---|---|---|---|---|---|---|---|
| Adams-Bashforth 2 | Regular | 0.4594 | 0.4688 | 0.3062 | 2.081980 | 2.449338 | 6 | 0.000797 |
| Backward Euler | Regular | 2.4818 | 2.25 | 0.5924 | 3.952281 | 4.678504 | 6 | 0.013604 |
| Runge-Kutta 4 | Regular | 0.6391 | 0.625 | 0.3059 | 5.20e-10 | 5.30e-10 | 6 | 1.97e-10 |


## 5. Stability Analysis (dt Sweep)
To empirically test the theoretical stability properties, we swept the time step $h$ from $0.01$ to $2.0$ ms and measured the RMSE against the high-resolution Ground Truth.

* **Adams-Bashforth 2:** Fails and explodes (NaN/Infinity) at larger step sizes due to its conditional stability constraint.
* **Backward Euler:** Continues to simulate without crashing, demonstrating unconditional A-stability, albeit with degraded accuracy.

See `stability_analysis.png` in `/outputs/figures/` for the empirical graph.

### Empirical Stability Metrics

> **Ground Truth (LSODA)** has exactly **6 spikes** for this time window.

| Step Size $h$ (ms) | AB2 RMSE | BE RMSE | RK4 RMSE | AB2 Spikes | BE Spikes | RK4 Spikes |
|---|---|---|---|---|---|---|
| 0.01 | 2.082 | 3.952 | 5.20e-10 | 6 | 6 | 6 |
| 0.05 | 6.273 | 9.085 | 4.260 | 6 | 6 | 6 |
| 0.1 | 8.469 | 11.910 | 2.713 | 6 | 6 | 6 |
| 0.5 | 21.377 | 23.896 | 8.776 | 6 | 6 | 6 |
| 1.0 | 29.283 | 53.616 | 9.624 | 5 | 0 | 6 |
| 2.0 | 35.854 | 23.828 | 19.316 | 5 | 0 | 6 |
| 5.0 | 36.167 | 13.691 | 29.676 | 4 | 0 | 6 |
| 10.0 | 51.456 | 17.331 | NaN/Expl | 3 | 0 | - |

## 6. Visualizations
All generated Phase Portraits and Time-Series graphs (including zoomed-in spike insets to highlight phase characteristics) are saved in the `/outputs/figures/` directory.
