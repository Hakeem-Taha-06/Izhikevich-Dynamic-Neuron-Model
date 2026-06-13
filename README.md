# Numerical and Physics-Informed Machine Learning Solutions for the Izhikevich Dynamic Neuron Model

This repository contains the codebase, numerical solvers, and Physics-Informed Neural Network (PINN) implementations for simulating the Izhikevich dynamic neuron model. The project rigorously benchmarks classical numerical integration schemes against modern scientific machine learning approaches.

---

## 📖 Overview

Dynamic neuron models are essential in computational neuroscience for simulating the electrical activity of biological neurons. The **Izhikevich model** provides an elegant mathematical formulation that balances biophysical realism (like the Hodgkin-Huxley model) with exceptional computational efficiency (like Integrate-and-Fire models).

This project explores the governing ODEs and the discrete after-spike reset conditions of the Izhikevich model. We implement and evaluate three numerical integration schemes:
1. **Runge-Kutta 4 (RK4)**
2. **Backward Euler (BE)**
3. **Adams-Bashforth 2 (AB2)**

These are benchmarked against a high-accuracy **Radau ground truth**. A Physics-Informed Neural Network (PINN) is also developed, exposing critical challenges imposed by the non-differentiable spike reset discontinuity.

---

## 🧮 Mathematical Modeling

The Izhikevich model is a two-variable ODE system governing the membrane potential $v(t)$ and a slow recovery current $w(t)$, driven by an external input $I(t)$:

$$ C_m \frac{dv}{dt} = k(v-v_r)(v-v_t) - w + I(t) $$
$$ \frac{dw}{dt} = a[b(v-v_r) - w] $$

### The Discrete Reset Mechanism
An instantaneous discrete reset is applied whenever the membrane potential reaches the spike peak ($v \geq 35$ mV):
$$ \text{if } v \geq 35 \text{ mV, then } v \leftarrow c \text{ and } w \leftarrow w + d $$

---

## ⚙️ Numerical Solvers

### 1. Runge-Kutta 4 (RK4)
A single-step explicit method achieving $O(h^4)$ accuracy. It provides the best accuracy-to-speed trade-off and is robust against spike discontinuities across a wide range of step sizes.

### 2. Adams-Bashforth 2 (AB2)
A multi-step explicit method with $O(h^2)$ accuracy. To handle the discrete spikes, we implemented a custom **History Flush** protocol that clears the derivative history at each spike. AB2 proved to be the fastest solver and achieved the lowest RMSE at moderate step sizes.

### 3. Backward Euler (BE)
An unconditionally stable implicit method. While it prevents divergence at large step sizes, its non-convex residual admits spurious sub-threshold fixed points at $h \geq 1.0$ ms, completely silencing the spiking behavior.

---

## 🧠 Physics-Informed Neural Network (PINN)

A PINN architecture consisting of 6 hidden layers × 128 neurons was developed to map normalized time to biological-scale outputs. The physics loss penalizes ODE residuals via auto-differentiation, employing a *curriculum spike-masking* protocol to avoid explosive gradients near the reset discontinuity.

### PINN Training Phases:
1. **Phase 1 (Adam)**: 8,000 epochs of initial convergence.
2. **Phase 2 (L-BFGS)**: 100 epochs of fine-tuning quasi-Newton refinement.

---

## 📊 Key Results

### Solver Performance Summary ($h=0.1$ ms)
| Method | Order | RMSE ($v$) | Time (s) |
|--------|-------|------------|----------|
| **AB2** | $O(h^2)$ | **2.78 mV** | **0.054 s** |
| **RK4** | $O(h^4)$ | 3.62 mV | 0.080 s |
| **BE**  | $O(h^1)$ | 4.97 mV | 0.468 s |

*Note: All three numerical methods detected 6 spikes at this step size. At larger step sizes ($h \geq 1.0$ ms), explicit methods explode, while BE maintains stability but fails biologically by detecting 0 spikes.*

### PINN Surrogate Performance
The composite training loss converged successfully, confirming that the curriculum spike-masking pipeline is mathematically sound. However, during inference:
- **Voltage RMSE**: 31.9 mV
- **Spike Count**: 0 spikes

Two correctable root causes for this failure were identified:
1. **Data Mismatch**: The training CSV was inadvertently generated with a constant $I=300$ pA over 100 ms instead of the $0 \to 70$ pA step over 1000 ms, preventing the model from learning the quiescent-to-spiking transition.
2. **Inference Mismatch**: A constant inference current was applied to all time steps, creating a train/inference distribution mismatch.

Once these data pipeline issues are corrected, the PINN architecture is expected to become a viable competitive continuous-time surrogate.

---

## 🛠️ Repository Structure

```text
├── data/                  # Ground truth CSVs generated via Radau integration
├── outputs/figures/       # Evaluation plots (Phase portraits, Time-series)
├── src/
│   ├── numerical/         # RK4, AB2, and BE solver implementations
│   ├── ml_model/          # PINN architecture, Fourier Features, and Training loop
│   └── evaluation/        # Master evaluator for RMSE, DTW, and Stability Analysis
├── config.py              # Centralized physical parameters and simulation settings
└── README.md              # Project documentation (This file)
```
