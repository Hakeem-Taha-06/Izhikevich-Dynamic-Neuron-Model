# Physics-Informed Neural Network (PINN) — `src/ml_model/`

This module implements a Physics-Informed Neural Network to predict the membrane potential `v(t)` and recovery variable `w(t)` of the Izhikevich (2007) biophysical neuron model. The PINN is trained on sparse samples of the ground truth (10% of data points) while enforcing the governing ODE on the full time domain via automatic differentiation.

---

## 1. Network Architecture (`architecture.py`)

### Topology

| Component | Choice | Justification |
|---|---|---|
| Input | `t` — 1 feature (time in ms) | Single step-current protocol; `I_ext(t)` is computed internally by the physics loss |
| Output | `[v, w]` — 2 features in physical units (mV, pA) | Directly consumed by physics loss and evaluation |
| Hidden layers | 6 layers × 128 neurons | Sufficient depth to capture the sharp spike transitions |
| Activation | `Tanh` | Smooth, bounded, infinitely differentiable — required for `torch.autograd.grad` to compute dv/dt and dw/dt |
| Output activation | None (linear) | The network must predict values across the full range (−60 to 35 mV) |
| Weight init | Xavier normal (gain=1.0) | Improved gradient flow in deeper networks |
| Total parameters | 83,074 | |

### Input Normalization

Time is normalized to [-1, 1] via:

$$t_{norm} = \frac{t - T_{max}/2}{T_{max}/2}$$

This centers the input around zero and matches the Tanh activation's sensitive region.

### Output Denormalization

A fixed affine transform inside `forward()` maps the raw network output (near zero at initialization) to physical units:

$$v = \text{raw}_v \times V_{SCALE} + V_{SHIFT}$$
$$w = \text{raw}_w \times W_{SCALE} + W_{SHIFT}$$

| Constant | Value | Derivation |
|---|---|---|
| V_SHIFT | −60.0 mV | Resting potential `v_r` |
| V_SCALE | 95.0 mV | `v_peak − v_r` (full operating range) |
| W_SHIFT | 0.0 pA | Recovery at rest |
| W_SCALE | 100.0 pA | `|d|` (spike-reset increment) |

**Key benefit:** At initialization (raw ≈ 0), the network predicts the resting state (v ≈ −60 mV, w ≈ 0 pA), which already satisfies the ODE equilibrium. Training starts from a physically reasonable baseline.

The constants are stored as `register_buffer` (non-trainable, tracked by `state_dict`, moved with `.to(device)`).

---

## 2. Loss Function Formulation (`physics_loss.py`)

The total loss is a weighted sum of three terms:

$$\mathcal{L}_{total} = \lambda_{data} \cdot \mathcal{L}_{data} + \lambda_{phys} \cdot \mathcal{L}_{phys} + \lambda_{IC} \cdot \mathcal{L}_{IC}$$

### 2.1 Physics Loss (ODE Residual)

For each collocation point `t`, the physics loss penalizes violations of the governing ODEs:

1. **Forward pass:** Get `v_pred, w_pred` from model(t)
2. **Autograd:** Compute `dv/dt, dw/dt` via `torch.autograd.grad` with `create_graph=True`
3. **Step-current:** Compute `I_ext(t)` internally (0 for t < 100 ms, 70 pA for t ≥ 100 ms)
4. **Residual:**
   - $R_v = \frac{dv}{dt} - \frac{k(v - v_r)(v - v_t) - w + I_{ext}}{C_m}$
   - $R_w = \frac{dw}{dt} - a[b(v - v_r) - w]$
5. **Normalized loss:** $\mathcal{L}_{phys} = \text{mean}\left(\frac{R_v}{DV_{RATE}}\right)^2 + \text{mean}\left(\frac{R_w}{DW_{RATE}}\right)^2$

**Residual normalization** ensures both terms contribute equally:
| Constant | Value | Derivation |
|---|---|---|
| DV_RATE | 0.7 mV/ms | `I_EXT / C_m` — current-driven voltage rate |
| DW_RATE | 3.0 pA/ms | `a × |d|` — post-spike recovery rate |

### 2.2 Spike Masking

The discrete reset (`v ← c, w ← w + d` when `v ≥ v_peak`) creates a discontinuity that the continuous network cannot represent. Near the spike peak, the ODE residual becomes explosive.

**Solution:** Points where `v_pred > v_peak − peak_margin` are excluded from the physics loss via boolean indexing.

**Curriculum schedule (`get_margin`):** The margin starts wide (20 mV, masking v > 15 mV) and tightens linearly to 5 mV (masking v > 30 mV) over the course of training. This keeps training stable early while progressively enforcing physics closer to the spike peak.

| Training stage | Margin | Physics enforced up to | Coverage |
|---|---|---|---|
| Start (epoch 0) | 20.0 mV | 15 mV | 79% |
| Mid-training | 12.5 mV | 22.5 mV | 87% |
| Final | 5.0 mV | 30 mV | 95% |

### 2.3 Data Loss

Standard MSE between the network prediction and ground truth at sparse supervision points:

$$\mathcal{L}_{data} = \text{MSE}(\text{model}(t_{sparse}), \; [v_{gt}, w_{gt}]_{sparse})$$

With a data subsample factor of 10, this uses 10,001 out of 100,001 ground truth points (10%).

### 2.4 Initial-Condition Loss

Anchors the prediction at t = 0 to the prescribed initial state:

$$\mathcal{L}_{IC} = \left(\frac{v_{pred}(0) - V_0}{V_{SCALE}}\right)^2 + \left(\frac{w_{pred}(0) - W_0}{W_{SCALE}}\right)^2$$

### 2.5 Loss Weights

| Weight | Value | Role |
|---|---|---|
| λ_data | 200.0 | Strong data anchoring — the supervised data teaches the spike resets that physics alone cannot handle |
| λ_phys | 1.0 | Physics residual is dimensionless after normalization |
| λ_IC | 200.0 | Strong IC anchoring to prevent drift from the initial state |

---

## 3. Training Protocol (`train.py`)

### Two-Phase Optimization

| Phase | Optimizer | Epochs | Batch strategy | Purpose |
|---|---|---|---|---|
| 1 | Adam (lr=1e-3) | 8,000 | Mini-batched physics (16,384 pts), full-batch data | Fast initial convergence; SGD noise helps escape local minima |
| 2 | L-BFGS (lr=0.1) | 100 | Fixed random 16K-point subsample | Precise refinement using quasi-Newton curvature information |

### Data Strategy

The ground truth (100,001 points) is split into two roles:
- **Physics collocation:** All 100,001 points are used for ODE residual evaluation (dense coverage of the time domain)
- **Data supervision:** Every 10th point (10,001 points, 10%) provides sparse [v, w] targets

The network must learn the dynamics from the ODE residual, using only 10% of the data for supervision. This demonstrates the PINN's ability to leverage physics knowledge for data-efficient learning.

### Phase 1 — Adam

Each epoch:
1. Shuffle the 100K collocation points (`torch.randperm`)
2. For each mini-batch of 16,384 points:
   - Compute physics loss with autograd (`create_graph=True`)
   - Compute data loss on the full sparse set (only 10K points)
   - Compute IC loss at t=0
   - Backward pass with gradient clipping (`max_norm=1.0`)

### Phase 2 — L-BFGS

A fixed random subsample of 16,384 collocation points is drawn and reused for all 100 L-BFGS epochs. Configuration: `history_size=50`, `line_search_fn="strong_wolfe"`.

### ODE Residual Computation

The ODE residual is computed via **automatic differentiation** (not finite differences):

```python
dv_dt = torch.autograd.grad(
    outputs=v_pred, inputs=t,
    grad_outputs=torch.ones_like(v_pred),
    create_graph=True, retain_graph=True,
)[0]
```

`create_graph=True` builds a second-order computational graph so that gradients of the physics loss can flow back through the derivative computation to update the network weights.

---

## 4. Results

### PINN vs Ground Truth

The trained PINN successfully captures:
- ✅ Subthreshold membrane dynamics — learned from physics with minimal data supervision
- ✅ Spike timing and frequency — the ODE residual teaches *when* spikes should occur
- ✅ Recovery variable w(t) trajectory — accurately reproduced
- ✅ Correct initial conditions and quiescent pre-stimulus behavior

### Known Limitation: Spike Peak Amplitudes

Spike peak amplitudes are slightly underestimated after the first spike. This is a fundamental limitation: the spike reset (`v → c, w → w + d`) is a **discontinuity** that PINNs cannot represent because they assume smooth, differentiable solutions. The ODE residual must be masked near v_peak, leaving the network without physics guidance at the exact moment it needs it most.

This is not a failure of the implementation — it reveals a genuine limitation of applying continuous function approximators to discontinuous dynamical systems.

---

## 5. How to Run

### Training (Google Colab / Kaggle)
Upload the notebook `pinn_training_colab.ipynb` and set the data path in Cell 1.

### Training (Local)
```bash
python run_training.py
```

### Verification
```bash
python verify_pinn.py
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *J. Comput. Phys.*, 378, 686–707.
2. Cuomo, S. et al. (2022). Scientific machine learning through physics-informed neural networks: Where we are and what's next. *J. Sci. Comput.*, 92(3), 88.
3. Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.
4. Schiesser, W. E. (2014). *Differential Equation Analysis in Biomedical Science and Engineering*. Wiley.
