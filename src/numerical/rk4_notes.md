# Role 5: RK4 — Method Documentation & Analysis

**File:** `src/numerical/rk4.py`  
**Role Owner:** Role 5 — Explicit Method Developer  
**Model:** Izhikevich (2007) Generalized Biophysical Neuron  
**Deadline:** Saturday, June 13th 2026, 7:00 AM

---

## 1. Governing Equations

$$C_m \frac{dv}{dt} = k(v - v_r)(v - v_t) - u + I_{ext} \tag{1}$$

$$\frac{du}{dt} = a\bigl[b(v - v_r) - u\bigr] \tag{2}$$

**Discrete after-spike reset:**

$$\text{if } v \ge v_{peak}: \quad v \leftarrow c, \quad u \leftarrow u + d \tag{3}$$

---

## 2. Method Description — Classical RK4

### 2.1 Core Idea

The 4th-Order Runge-Kutta method evaluates the derivative at **four stages** within each step and combines them with a weighted average:

| Stage | Evaluation point | Derivative |
|---|---|---|
| k1 | $t_n$, $y_n$ | slope at start |
| k2 | $t_n + h/2$, $y_n + h k_1/2$ | slope at midpoint (using k1) |
| k3 | $t_n + h/2$, $y_n + h k_2/2$ | slope at midpoint (using k2) |
| k4 | $t_n + h$, $y_n + h k_3$ | slope at endpoint |

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

### 2.2 Reset Handling

The discrete reset (equation 3) is applied **after** the full RK4 step completes. This means all four stages (k1–k4) operate on the pre-reset state, and the discontinuity is only introduced at the committed step boundary. This avoids corrupting the intermediate slope estimates.

---

## 3. Stability Analysis

### 3.1 Theoretical Background

For the test equation $y' = \lambda y$ (where $\lambda < 0$), RK4 has the amplification factor:

$$R(h\lambda) = 1 + h\lambda + \frac{(h\lambda)^2}{2} + \frac{(h\lambda)^3}{6} + \frac{(h\lambda)^4}{24}$$

This is the 4th-order Taylor expansion of $e^{h\lambda}$. Stability requires $|R(h\lambda)| \le 1$, which for real $\lambda < 0$ holds when:

$$h \le \frac{2.785}{|\lambda_{max}|}$$

### 3.2 Experimental Stability Limit

Using the automated `find_stability_limit()` scanner on the Izhikevich model with default parameters (`I_ext = 300 pA`, Regular Spiking):

| Parameter | Value |
|---|---|
| Stable up to dt | **7.0 ms** |
| **First NaN at dt** | **7.5 ms** |
| Search step | 0.5 ms |
| I_ext used | 300 pA |

**Interpretation:** The Izhikevich system near the spike upstroke has fast dynamics (large $|\lambda_{max}|$), which constrains RK4's stability. At dt = 7.5 ms, the step is too large to track the spike accurately and the solution diverges.

---

## 4. Algorithmic Complexity (Big-O)

Let:
- $N$ = number of time steps = $(T_{end} - T_{start}) / h$
- Each step: exactly **4 function evaluations** (k1–k4), each $O(1)$

| Operation | Cost per step | Total cost |
|---|---|---|
| k1 evaluation | $O(1)$ | $O(N)$ |
| k2 evaluation | $O(1)$ | $O(N)$ |
| k3 evaluation | $O(1)$ | $O(N)$ |
| k4 evaluation | $O(1)$ | $O(N)$ |
| Reset check | $O(1)$ | $O(N)$ |
| Array write | $O(1)$ | $O(N)$ |
| **Overall** | **O(1)** | **O(N)** |

**Space complexity:** $O(N)$ — trajectory stored in pre-allocated $(N, 3)$ array.

**Comparison vs Backward Euler:** RK4 costs 4 function evaluations/step (fixed), vs ~7 fsolve evaluations/step for Backward Euler. However, RK4 is conditionally stable (dt < 7.5 ms), while Backward Euler is unconditionally stable.

---

## 5. Output Interface Contract

| Field | Specification |
|---|---|
| Return type | `numpy.ndarray` |
| Shape | `(N, 3)` |
| Column 0 | Time (ms) — uniformly spaced by `dt` |
| Column 1 | v (mV) — membrane potential |
| Column 2 | u (pA) — recovery variable |
| Spike representation | Reset applied post-step; spike visible as step-down in v |

**Example (first 3 rows, default params):**

| Time (ms) | v (mV) | u (pA) |
|---|---|---|
| 0.00 | -60.00 | 0.00 |
| 0.01 | -59.97 | -0.000009 |
| 0.02 | -59.94 | -0.000036 |

---

## 6. Comparison with Other Methods in This Project

| Method | Role | Order | Stable? | Cost/step |
|---|---|---|---|---|
| Ground Truth (Radau + events) | Role 4 | ~5 | Stiff-stable | High |
| **RK4** | **Role 5** | **4** | **Conditionally (dt < 7.5 ms)** | **4 func evals** |
| Backward Euler | Role 6 | 1 | Unconditional | 1 nonlinear solve (~7 iters) |
| Adams-Bashforth 2 | Role 7 | 2 | Conditionally | 2 func evals + history |

**Trade-off:** RK4 gives the best accuracy among the explicit methods (4th order) at the cost of a step-size restriction. It is the natural baseline for comparing accuracy against the ground truth.

---

## 7. References

1. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.  
2. Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations* (3rd ed.). Wiley.  
3. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving ODEs I: Nonstiff Problems*. Springer.