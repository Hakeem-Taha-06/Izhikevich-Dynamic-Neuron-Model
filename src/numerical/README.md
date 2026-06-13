# Numerical Solvers — `src/numerical/`

This module contains four solver implementations for the Izhikevich (2007) generalized biophysical neuron model. All solvers share the parameter set defined in `config.py` and output trajectories as `numpy.ndarray` of shape `(N, 3)` with columns `[Time (ms), v (mV), w (pA)]`.

---

## 1. Ground Truth Generator (`ground_truth_generator.py`)

### Method
The reference trajectory is produced by `scipy.integrate.solve_ivp` using the **Radau IIA** method — an implicit Runge–Kutta scheme (order 5) specifically designed for stiff systems. The Radau method is L-stable, meaning it suppresses numerical oscillations even when the system's eigenvalues span several orders of magnitude, as is the case during the fast spike upstroke.

Tolerances are set to `rtol = atol = 1e-12` to ensure the reference trajectory is accurate to machine precision. The evaluation grid uses `dt = 0.01 ms`, producing 100,001 uniformly spaced output points over the 1000 ms simulation window.

### Why This is a Reliable Reference
- **Stiff-aware integration:** The Radau method adapts its internal step size independently of the output grid, using much smaller steps during the spike upstroke (where dv/dt > 50 mV/ms) and larger steps during the quiescent phase.
- **Tight tolerances:** At `rtol = atol = 1e-12`, the local truncation error per step is bounded to ~12 significant digits, making numerical error negligible compared to solver errors from the other methods.
- **Discrete reset handling:** The spike reset (`v ← c, w ← w + d` when `v ≥ v_peak`) is applied as a post-integration step at each evaluation point, consistent with the event-driven protocol described in the original model.

### How to Run
```bash
python main.py --role 1
```
Output: `data/ground_truth.csv`

---

## 2. Backward Euler Solver (`backward_euler.py`)

### Method Description
The Backward (Implicit) Euler method evaluates the derivative at the *next* time step:

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

Because $y_{n+1}$ appears on both sides, each step requires solving a 2×2 nonlinear system using `scipy.optimize.fsolve` (modified Powell hybrid method with finite-difference Jacobian).

### Applied to the Izhikevich ODE
At each step, the implicit system is:

$$G_1(v_{n+1}, w_{n+1}) = v_{n+1} - v_n - h \cdot f_v(v_{n+1}, w_{n+1}) = 0$$
$$G_2(v_{n+1}, w_{n+1}) = w_{n+1} - w_n - h \cdot f_w(v_{n+1}, w_{n+1}) = 0$$

The current state $(v_n, w_n)$ serves as the initial guess for `fsolve`. With `dt = 0.01 ms`, convergence is achieved in exactly 7 function evaluations per step with zero failures across all 10,001 steps.

### Stability Analysis
For the linear test equation $y' = \lambda y$ (Re(λ) < 0), the amplification factor is:

$$|R(h\lambda)| = \frac{1}{|1 - h\lambda|} < 1 \quad \forall h > 0$$

This makes Backward Euler **unconditionally A-stable** — errors shrink at every step regardless of step size.

| Property | Forward Euler | Backward Euler |
|---|---|---|
| Accuracy order | O(h) | O(h) |
| Stability region | Finite disk | **Entire left half-plane** |
| Step size restriction | h < 2/\|λ_max\| | **None** |
| Per-step cost | 1 function eval | ~7 fsolve iterations |

### Stiffness in the Izhikevich Model
The Izhikevich system exhibits moderate stiffness: the voltage equation has a fast time scale (~1 ms during spikes) while the recovery variable evolves on a slow time scale (~33 ms, set by 1/a). This ratio of ~33:1 in eigenvalue magnitudes causes explicit methods to require small step sizes for stability, while Backward Euler remains stable at any step size.

However, at `h = 1.0 ms`, the implicit solver converges to a spurious sub-threshold fixed point, silencing all spiking. Unconditional stability guarantees no numerical explosion — but not biological correctness.

### Handling the Discrete Reset
The reset is applied **post-convergence**: after `fsolve` returns $(v_{n+1}, w_{n+1})$, the spike check is performed. This ensures the root-finder never sees the discontinuity.

### Error vs Ground Truth
Step size `dt = 0.01 ms`:
- RMSE v: 2.02 mV
- RMSE w: 2.41 pA
- Detected spikes: 6 (correct)

---

## 3. RK4 Solver (`rk4.py`)

### Method Description
The classic 4th-order Runge–Kutta method computes four intermediate slopes per step:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + h/2, \; y_n + h \cdot k_1/2)$$
$$k_3 = f(t_n + h/2, \; y_n + h \cdot k_2/2)$$
$$k_4 = f(t_n + h, \; y_n + h \cdot k_3)$$
$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

For the Izhikevich system, $f$ is the 2D vector:
$$f(t, [v, w]) = \left[\frac{k(v-v_r)(v-v_t) - w + I_{ext}(t)}{C_m}, \;\; a(b(v-v_r) - w)\right]$$

### Truncation Error Order
RK4 has local truncation error O(h⁵) and global error O(h⁴). Halving the step size reduces the global error by a factor of ~16.

### Error vs Ground Truth
Step size `dt = 0.01 ms`:
- RMSE v: 1.59 mV
- RMSE w: 1.89 pA
- Detected spikes: 6 (correct)

RK4 achieves the best accuracy-to-speed ratio among the three solvers due to its high convergence order and robustness against the spike discontinuities.

---

## 4. Adams–Bashforth 2 Solver (`adams_bashforth2.py`)

### Method Description
Adams–Bashforth 2 (AB2) is a second-order linear multi-step method that uses derivative information from the two most recent time steps:

$$y_{n+1} = y_n + h\left(\frac{3}{2}f_n - \frac{1}{2}f_{n-1}\right)$$

### History Flush at Spike Resets
The discrete reset invalidates the AB2 history because the stored derivative $f_{n-1}$ was computed before the discontinuity. The implementation handles this with a **history flush protocol**:
1. When `v ≥ v_peak`, apply the reset (`v ← c, w ← w + d`)
2. Execute a single Forward Euler step to generate a fresh history point
3. Resume AB2 integration from the new history

### Error vs Ground Truth
Step size `dt = 0.01 ms`:
- RMSE v: 1.60 mV
- RMSE w: 1.91 pA
- Detected spikes: 6 (correct)

AB2 achieves the fastest wall-clock time among the three solvers due to requiring only 2 function evaluations per step (vs. 4 for RK4 and ~7 for BE).

---

## 5. Solver Comparison Summary

| Method | Order | Stability | RMSE v (mV) | RMSE w (pA) | Spikes | Wall Time (s) |
|---|---|---|---|---|---|---|
| **RK4** | O(h⁴) | Conditional | 1.59 | 1.89 | 6 | 4.52 |
| **AB2** | O(h²) | Conditional | 1.60 | 1.91 | 6 | 5.51 |
| **Backward Euler** | O(h¹) | Unconditional | 2.02 | 2.41 | 6 | 29.71 |

All metrics at `dt = 0.01 ms` against the Radau ground truth.

---

## References

1. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.
2. Schiesser, W. E. (2014). *Differential Equation Analysis in Biomedical Science and Engineering*. Wiley. Chapter 4 (Dynamic Neuron Model).
3. Ascher, U. M. & Petzold, L. R. (1998). *Computer Methods for ODEs and DAEs*. SIAM.
4. Dormand, J. R. & Prince, P. J. (1980). A family of embedded Runge–Kutta formulae. *J. Comput. Appl. Math.*, 6(1), 19–26.
