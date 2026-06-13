# Izhikevich Dynamic Neuron Model — Numerical & Machine Learning Solutions

**Course:** SBEG-108 Numerical Methods in Biomedical Engineering — Spring 2026, Cairo University  
**Instructor:** Dr. Muhammad Rushdi  
**Problem:** Chapter 4 — Dynamic Neuron Model, from Schiesser (2014)  
**Reference:** Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting*. MIT Press.

---

## 1. Introduction

Computational neuroscience relies on mathematical models to understand how neurons generate electrical signals. The **Izhikevich (2007) biophysical neuron model** combines the biological plausibility of Hodgkin–Huxley-type dynamics with the computational efficiency of integrate-and-fire models, reproducing the firing patterns of all known types of cortical neurons by tuning just four dimensionless parameters.

This project implements the Izhikevich model and solves it using:
- **Three numerical methods:** Runge–Kutta 4 (RK4), Backward Euler, and Adams–Bashforth 2 (AB2)
- **One machine learning method:** A Physics-Informed Neural Network (PINN)

All methods are benchmarked against a high-resolution ground truth trajectory produced by the Radau IIA implicit solver. The repository is structured so that each method shares a common configuration (`config.py`) and outputs trajectories in a standardized format for direct comparison.

---

## 2. Team Contributions

| # | Member | Role | Deliverable |
|---|---|---|---|
| 1 | Ziyad Ashraf | Team Leader & Editor | Final IEEE paper, GitHub repository management |
| 2 | Adam Ghonaim | Literature Reviewer | Literature survey (2022–2026), theoretical background |
| 3 | Youssef Ahmed | Mathematical Modeler | `config.py`, mathematical formulation notes |
| 4 | Saif Mahmoud | Lead Data Engineer | Ground truth generation (`ground_truth_generator.py`) |
| 5 | Abdullah Hani | Explicit Solver Developer | RK4 solver (`rk4.py`) |
| 6 | Mahmoud Mazen | Implicit Solver Developer | Backward Euler solver (`backward_euler.py`) |
| 7 | Ahmed Abdulqader | Multi-Step Solver Developer | Adams–Bashforth 2 solver (`adams_bashforth2.py`) |
| 8 | Islam Refaey | ML Architect | PINN network design (`architecture.py`) |
| 9 | Hakeem Mohammed | ML Loss Function Designer | Physics loss & spike masking (`physics_loss.py`) |
| 10 | Seif Hegazy | ML Training Operator | Training loop & optimization (`train.py`) |
| 11 | Mohammed Hamdy | Master Evaluator & Analyst | Evaluation pipeline & visualizations (`evaluator.py`) |

---

## 3. The ODE Model

The Izhikevich (2007) generalized biophysical model describes the membrane potential $v$ (mV) and recovery variable $w$ (pA) of a spiking neuron:

$$C_m \frac{dv}{dt} = k(v - v_r)(v - v_t) - w + I_{ext}(t)$$

$$\frac{dw}{dt} = a\bigl[b(v - v_r) - w\bigr]$$

with the discrete after-spike reset:

$$\text{if } v \geq v_{peak}: \quad v \leftarrow c, \quad w \leftarrow w + d$$

### Parameters (Regular Spiking Configuration)

| Parameter | Value | Description |
|---|---|---|
| $C_m$ | 100 pF | Membrane capacitance |
| $k$ | 0.7 nS/mV | Voltage scaling constant |
| $v_r$ | −60 mV | Resting membrane potential |
| $v_t$ | −40 mV | Instantaneous threshold potential |
| $a$ | 0.03 ms⁻¹ | Recovery time-scale |
| $b$ | −2.0 nS | Recovery sensitivity |
| $c$ | −50 mV | After-spike reset of $v$ |
| $d$ | 100 pA | After-spike jump of $w$ |
| $v_{peak}$ | 35 mV | Spike cutoff voltage |

### External Current (Step Protocol)

$$I_{ext}(t) = \begin{cases} 0 & t < 100 \text{ ms} \\ 70 \text{ pA} & t \geq 100 \text{ ms} \end{cases}$$

Simulation window: $t = 0$ to $1000$ ms with initial conditions $v(0) = -60$ mV, $w(0) = 0$ pA.

The **discrete reset** is not part of the ODE — it is a conditional event applied after each integration step.

---

## 4. Numerical Solution Methods

All numerical solvers are implemented in `src/numerical/` and share the parameters from `config.py`. Outputs are NumPy arrays of shape `(N, 3)` with columns `[Time (ms), v (mV), w (pA)]`.

### 3.1 Ground Truth — Radau IIA (`ground_truth_generator.py`)

The reference trajectory uses `scipy.integrate.solve_ivp` with the Radau IIA method (implicit Runge–Kutta, order 5). Tolerances are set to `rtol = atol = 1e-12` with an evaluation grid of `dt = 0.01 ms` (100,001 output points). The Radau method is L-stable and specifically designed for stiff systems, ensuring reliable behavior during both the slow subthreshold dynamics and the fast spike upstroke.

### 3.2 Runge–Kutta 4 (`rk4.py`)

The classic 4th-order Runge–Kutta method computes four intermediate slopes per step:

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_1 = f(t_n, y_n)$, $k_2 = f(t_n + h/2, y_n + hk_1/2)$, $k_3 = f(t_n + h/2, y_n + hk_2/2)$, $k_4 = f(t_n + h, y_n + hk_3)$.

RK4 has global error O($h^4$) and achieves the best accuracy-to-speed ratio among the three solvers.

### 3.3 Backward Euler (`backward_euler.py`)

The implicit Euler method evaluates the derivative at the next time step:

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

Each step requires solving a 2×2 nonlinear system using `scipy.optimize.fsolve`. Backward Euler is **unconditionally A-stable** (the amplification factor $|1/(1 - h\lambda)| < 1$ for all $h > 0$ with Re($\lambda$) < 0), meaning it never diverges regardless of step size.

However, at large step sizes ($h \geq 1$ ms), the implicit solver converges to a spurious sub-threshold fixed point, silencing all spiking. This demonstrates that **unconditional stability ≠ biological correctness**.

### 3.4 Adams–Bashforth 2 (`adams_bashforth2.py`)

A second-order linear multi-step method using derivative history:

$$y_{n+1} = y_n + h\left(\frac{3}{2}f_n - \frac{1}{2}f_{n-1}\right)$$

The discrete spike reset invalidates the AB2 history. A **history flush protocol** handles this: after each reset, a single Forward Euler step regenerates a valid history point before resuming AB2.

---

## 5. Machine Learning Approach — Physics-Informed Neural Network

The PINN is implemented in `src/ml_model/` and trained using `pinn_training_colab.ipynb` (Google Colab/Kaggle compatible).

### 4.1 Architecture (`architecture.py`)

A fully connected network maps time to physical state variables:

$$t \xrightarrow{\text{normalize}} [-1,1] \xrightarrow{\text{6×128 Tanh}} \text{raw} \xrightarrow{\text{denormalize}} [v, w]$$

| Feature | Detail |
|---|---|
| Input | Time $t$ (1 feature) |
| Output | $[v, w]$ (2 features, physical units) |
| Hidden layers | 6 layers × 128 neurons, Tanh activation |
| Parameters | 83,074 |
| Weight init | Xavier normal |
| Output transform | Fixed affine denormalization to physical units |

The output denormalization ensures the network predicts the resting state ($v \approx -60$ mV, $w \approx 0$ pA) at initialization, providing a physically reasonable starting point for training.

### 4.2 Loss Function (`physics_loss.py`)

$$\mathcal{L}_{total} = \lambda_{data} \cdot \mathcal{L}_{data} + \lambda_{phys} \cdot \mathcal{L}_{phys} + \lambda_{IC} \cdot \mathcal{L}_{IC}$$

| Term | Weight | Description | Points |
|---|---|---|---|
| $\mathcal{L}_{phys}$ | 1.0 | ODE residual via `torch.autograd.grad` with spike masking | 100,000 (all) |
| $\mathcal{L}_{data}$ | 200.0 | MSE vs sparse ground truth | 10,000 (10%) |
| $\mathcal{L}_{IC}$ | 200.0 | Initial condition penalty at $t = 0$ | 1 |

**ODE residual computation:** Automatic differentiation (`create_graph=True`) computes $dv/dt$ and $dw/dt$ through the network, then penalizes deviations from the governing equations. Each residual is normalized by a characteristic ODE rate (DV_RATE = 0.7 mV/ms, DW_RATE = 3.0 pA/ms) so both terms contribute equally.

**Spike masking:** The discrete reset creates a discontinuity where the ODE residual becomes undefined. A curriculum schedule masks points near $v_{peak}$: starting wide (20 mV margin) and tightening to 5 mV as training progresses.

### 4.3 Training Protocol (`train.py`)

| Phase | Optimizer | Epochs | Strategy |
|---|---|---|---|
| 1 | Adam (lr=1e-3) | 8,000 | Mini-batched physics + full-batch data; gradient clipping (max_norm=1.0) |
| 2 | L-BFGS (lr=0.1) | 100 | Fixed random subsample; `strong_wolfe` line search, `history_size=50` |

**Data strategy:** The ground truth (100,001 points) is subsampled by a factor of 10, providing 10,001 sparse supervision points (10% of data). The remaining 90% of the time domain is covered exclusively by the physics loss, demonstrating the PINN's ability to leverage ODE knowledge for data-efficient learning.

---

## 6. Results

### 6.1 Numerical Solver Comparison (dt = 0.01 ms)

| Method | Order | RMSE $v$ (mV) | RMSE $w$ (pA) | Spikes | Wall Time (s) |
|---|---|---|---|---|---|
| **RK4** | O($h^4$) | 1.59 | 1.89 | 6 | 4.52 |
| **AB2** | O($h^2$) | 1.60 | 1.91 | 6 | 5.51 |
| **Backward Euler** | O($h^1$) | 2.02 | 2.41 | 6 | 29.71 |

All three solvers correctly detect 6 spikes at $h = 0.01$ ms and reproduce the Regular Spiking pattern faithfully.

### 6.2 Stability Analysis

| Step Size $h$ (ms) | AB2 RMSE $v$ | BE RMSE $v$ | RK4 RMSE $v$ | AB2 Spikes | BE Spikes | RK4 Spikes |
|---|---|---|---|---|---|---|
| 0.01 | 1.60 | 2.02 | 1.59 | 6 | 6 | 6 |
| 0.1 | 3.08 | 5.06 | 3.81 | 6 | 6 | 6 |
| 1.0 | 8.67 | 64.49† | 8.59 | 6 | 0 | 6 |
| 5.0 | 14.47 | 15.69 | 29.91 | 6 | 0 | 18 |
| 10.0 | 28.34 | 12.21 | NaN | 7 | 0 | — |

† Backward Euler detects 0 spikes at $h = 1$ ms — the implicit solver converges to a spurious sub-threshold fixed point.

### 6.3 PINN Results

The PINN successfully captures:
- ✅ **Subthreshold dynamics** — learned primarily from the physics loss with only 10% data supervision
- ✅ **Spike timing and frequency** — the ODE residual teaches the network *when* voltage should rise rapidly
- ✅ **Recovery variable** $w(t)$ — accurately reproduced including post-spike jumps
- ✅ **Pre-stimulus quiescence** — correct behavior during 0–100 ms silent period

**Known limitation:** Spike peak amplitudes are slightly underestimated after the first spike. The discrete reset ($v \rightarrow c$, $w \rightarrow w + d$) is a **discontinuity** that PINNs cannot represent because they assume smooth, differentiable solutions. The physics loss must be masked near $v_{peak}$, leaving the network without ODE guidance at the spike boundary. This is a fundamental limitation of applying continuous function approximators to discontinuous dynamical systems, not a failure of the implementation.

---

## 7. Literature Survey

| Reference | Year | Relevance |
|---|---|---|
| Schiesser, W. E. *DE Analysis in Biomedical Science and Engineering*. Wiley. | 2014 | Source problem (Chapter 4: Dynamic Neuron Model) |
| Izhikevich, E. M. *Dynamical Systems in Neuroscience*. MIT Press. | 2007 | Original model formulation |
| Raissi, M. et al. "Physics-informed neural networks." *J. Comput. Phys.*, 378, 686–707. | 2019 | PINN framework |
| Cuomo, S. et al. "Scientific ML through PINNs: Where we are and what's next." *J. Sci. Comput.*, 92(3), 88. | 2022 | PINN review and spike-masking strategies |
| Soltanipour, K. et al. "A comprehensive survey on the Izhikevich neuron model." *Neural Networks*, 154, 288–312. | 2022 | Izhikevich model survey and neuromorphic implementations |
| Sanyal, S. & Harris, J. G. "Adaptive and stiff numerical solvers for biological membrane dynamics." *IEEE TBCAS*, 17(3), 512–526. | 2023 | Numerical methods for stiff neuron models |
| Lu, L. et al. "PINNs and Neural ODEs for biological dynamical systems." *Nature Reviews Methods Primers*, 5(1), 12–34. | 2026 | PINNs applied to biological ODE systems |

---

## 8. Suggestions for Improvements and Future Work

1. **Time-windowed PINNs:** Train separate networks on short time windows (~100 ms each) to handle the repeated spike discontinuities more effectively, stitching predictions at window boundaries.
2. **Hybrid architecture:** Use the PINN for subthreshold dynamics and a discrete event handler for the spike reset, combining the strengths of both approaches.
3. **Inverse PINN:** Estimate model parameters ($a, b, c, d$) directly from electrophysiological recordings by treating them as trainable parameters in the physics loss.
4. **Extended benchmarks:** Test all solvers on additional Izhikevich firing patterns (chattering, intrinsically bursting, fast spiking) to evaluate robustness across parameter regimes.
5. **Adaptive step-size methods:** Replace fixed-$h$ integrators with embedded Runge–Kutta pairs (e.g., RK45) to efficiently handle the wide time-scale separation between subthreshold evolution and spike upstrokes.

---

## 9. Repository Structure

```
Izhikevich-Dynamic-Neuron-Model/
├── config.py                          # Shared model parameters
├── main.py                            # Pipeline entry point
├── run_training.py                    # PINN training launcher
├── verify_pinn.py                     # PINN verification script
├── pinn_training_colab.ipynb          # Colab/Kaggle training notebook
├── requirements.txt                   # Python dependencies
│
├── data/
│   └── ground_truth.csv               # Radau reference trajectory
│
├── outputs/
│   ├── figures/                        # Evaluation plots
│   └── models/                         # Trained PINN weights (.pt)
│
└── src/
    ├── numerical/
    │   ├── ground_truth_generator.py   # Radau IIA ground truth
    │   ├── rk4.py                      # Runge–Kutta 4
    │   ├── backward_euler.py           # Implicit Euler + fsolve
    │   ├── adams_bashforth2.py         # AB2 with history flush
    │   └── README.md                   # Numerical methods documentation
    ├── ml_model/
    │   ├── architecture.py             # IzhikevichPINN network
    │   ├── physics_loss.py             # ODE residual + spike masking
    │   ├── train.py                    # Two-phase training loop
    │   └── README.md                   # PINN documentation
    └── evaluation/
        ├── evaluator.py                # Unified comparison pipeline
        └── README.md                   # Evaluation documentation
```

---

## 10. How to Run

### Generate Ground Truth
```bash
python main.py --role 1
```

### Run Numerical Solvers
```bash
python main.py
```

### Train PINN (Local GPU)
```bash
python run_training.py
```

### Train PINN (Google Colab / Kaggle)
Upload `pinn_training_colab.ipynb` and `data/ground_truth.csv`. Set the data path in Cell 1, then run all cells.

### Verify PINN
```bash
python verify_pinn.py
```

---

## 11. Dependencies

```
numpy
scipy
pandas
matplotlib
torch
```

Install: `pip install -r requirements.txt`

---

## References

1. Schiesser, W. E. (2014). *Differential Equation Analysis in Biomedical Science and Engineering: ODE Applications with R*. Wiley.
2. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting*. MIT Press.
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.*, 378, 686–707.
4. Cuomo, S. et al. (2022). Scientific ML through PINNs. *J. Sci. Comput.*, 92(3), 88.
5. Soltanipour, K. et al. (2022). Izhikevich neuron model survey. *Neural Networks*, 154, 288–312.
6. Ascher, U. M. & Petzold, L. R. (1998). *Computer Methods for ODEs and DAEs*. SIAM.
