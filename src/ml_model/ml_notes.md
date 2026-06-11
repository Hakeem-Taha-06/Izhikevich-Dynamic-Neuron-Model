# ML Module — Engineering Decisions & Rationale

**Roles covered:** 8 (Architecture), 9 (Physics Loss)  
**Author:** Hakeem  
**Last updated:** 2026-06-11

---

## 1. Architecture Design (`architecture.py`)

### 1.1 Network Topology

| Component | Choice | Justification |
|---|---|---|
| Input features | `[t, I_ext, V_0, U_0]` — 4 inputs | A single model must generalise across the 4,000-simulation dataset (Role 4), which sweeps external current and initial conditions. Encoding `I_ext`, `V_0`, `U_0` as inputs allows the network to learn the full family of trajectories. |
| Output features | `[v, u]` — 2 outputs in physical units (mV, pA) | Directly consumed by the physics loss, IC loss, and evaluation pipeline. No downstream denormalization required. |
| Hidden layers | 3 layers × 64 neurons (default) | Sufficient capacity for a 2-variable ODE system with moderate nonlinearity. Configurable via constructor for tuning. |
| Activation | `Tanh` | Smooth, bounded in [-1, 1], infinitely differentiable. Required by the PINN approach — `torch.autograd.grad` computes dv/dt and du/dt through the activation, so non-differentiable activations (ReLU) would produce piecewise-constant derivatives. |
| Output activation | None (linear) | The network must predict physical values across the full voltage range (−60 to 35 mV) and unbounded recovery variable. A bounded activation would clip the output. |

### 1.2 Output Denormalization

**Problem:** The raw network output is centered near zero (Tanh hidden layers feed a final Linear layer with small initial weights). But the physical targets span very different ranges:
- `v` lives in [−60, 35] mV
- `u` can reach hundreds of pA

Without denormalization, the final Linear layer must learn large biases and weights to shift its output to the correct range, wasting early training epochs.

**Decision:** Apply a fixed affine transform inside `forward()`:

```
v = raw_v × V_SCALE + V_SHIFT
u = raw_u × U_SCALE + U_SHIFT
```

| Constant | Value | Derivation |
|---|---|---|
| `V_SHIFT` | −60.0 mV | `v_r` — resting potential |
| `V_SCALE` | 95.0 mV | `v_peak − v_r` — full operating range |
| `U_SHIFT` | 0.0 pA | Steady-state recovery at rest |
| `U_SCALE` | 100.0 pA | `|d|` — the spike-reset increment |

**Consequence:** At initialization (raw ≈ 0), the network predicts the resting state (`v ≈ −60 mV`, `u ≈ 0 pA`), which already satisfies the ODE at equilibrium. Training starts from a physically reasonable baseline rather than from random noise.

**Autograd compatibility:** The affine transform is inside `forward()`, so `torch.autograd.grad` chains through it automatically: `dv_physical/dt = dv_raw/dt × V_SCALE`. No special handling is needed in the physics loss.

The constants are stored as `register_buffer` (non-trainable, tracked by `state_dict`, moved with `.to(device)`).

---

## 2. Physics Loss Design (`physics_loss.py`)

### 2.1 Core Approach — ODE Residual

The physics loss penalises violations of the governing ODEs. For each collocation point `(t, I_ext, V_0, U_0)`:

1. **Forward pass** → get `v_pred`, `u_pred`
2. **Autograd** → compute `dv_pred/dt`, `du_pred/dt` via `torch.autograd.grad`
3. **Residual** → the difference between the autograd derivative and the ODE right-hand side:
   - `R_v = dv/dt − [k(v−v_r)(v−v_t) − u + I_ext] / C_m`
   - `R_u = du/dt − a·[b(v−v_r) − u]`
4. **Loss** → mean squared normalised residual

### 2.2 Spike Masking (Peak-Only)

**Problem:** The discrete reset (`v ← c`, `u ← u + d` when `v ≥ v_peak`) creates a discontinuity that the continuous network cannot represent. Near the spike, `dv_pred/dt` approaches infinity, producing explosive residuals that crash training with NaN.

**Decision: Mask the fast-upstroke zone only.** Points where `v_pred > v_peak − peak_margin` are excluded from the physics loss.

**Why not mask near the reset voltage (`c = −50 mV`) as well?** Because `c = −50` is only 10 mV above the resting potential `v_r = −60`. A near-reset mask would exclude the entire subthreshold region where the neuron spends most of its time and the ODE is smooth and perfectly valid. This was identified as a critical flaw in an earlier version and intentionally removed.

**Margin values were chosen from the actual dynamics:**

| v (mV) | k(v−v_r)(v−v_t) | dv/dt (mV/ms) | Role |
|---|---|---|---|
| −40 | 0 | 5.0 | Threshold (v_t) |
| −20 | 560 | 10.6 | |
| 0 | 1680 | 21.8 | |
| **15** | **2888** | **33.9** | **Start mask (margin=20)** |
| 25 | 3868 | 43.7 | |
| **30** | **4410** | **49.1** | **End mask (margin=5)** |
| 35 | 4988 | 54.9 | v_peak |

### 2.3 Curriculum Margin Schedule (`get_margin`)

**Decision:** The margin starts wide and tightens linearly over training epochs.

| Epoch | Margin | Masks v > | Physics-enforced range | Coverage |
|---|---|---|---|---|
| 0 (start) | 20.0 mV | 15 mV | −60 to 15 mV | 79% |
| mid | 12.5 mV | 22.5 mV | −60 to 22.5 mV | 87% |
| final | 5.0 mV | 30 mV | −60 to 30 mV | 95% |

**Rationale:** Early in training the network is untrained and the spike upstroke would produce catastrophic gradients. Starting with a wide margin (excluding dv/dt > 34 mV/ms) keeps training stable. As the network learns the smooth dynamics, the margin tightens to enforce physics closer to the spike peak.

**Role 10 dependency:** The `get_margin(epoch, max_epochs)` function must be called inside the training loop because it requires the current epoch number, which only exists there. This is the minimum coordination needed — one line per epoch.

### 2.4 Residual Scale Normalization

**Problem:** The two residuals have different physical units and magnitudes:
- `R_v` is in mV/ms — typical magnitude ~5–10
- `R_u` is in pA/ms — typical magnitude ~1–3
- Without normalization, `mean(R_v²)` dominates `mean(R_u²)` by 10–25×

**Decision:** Divide each residual by a characteristic ODE rate before squaring:

| Constant | Value | Derivation |
|---|---|---|
| `DV_RATE` | 5.0 mV/ms | `I_EXT_DEFAULT / C_m` — current-driven voltage rate at rest |
| `DU_RATE` | 3.0 pA/ms | `a × |d|` — post-spike recovery rate |

Both terms become dimensionless and contribute roughly equally. The ratio `(DV_RATE/DU_RATE)² = 2.8`, meaning without this normalization the voltage residual would dominate by nearly 3×.

### 2.5 Mean Computation — Boolean Indexing

**Problem:** An earlier implementation used `torch.where(mask, zeros, R)` to zero out masked points. The zeroed entries still appeared in the denominator of `torch.mean`, diluting the loss by the fraction of masked points.

**Decision:** Use boolean indexing on the valid (unmasked) subset:

```python
valid = ~(v_pred > (v_peak - peak_margin)).squeeze()
R_v_valid = R_v[valid]
loss = torch.mean((R_v_valid / DV_RATE) ** 2) + ...
```

An edge-case guard returns `0.0` if every point in the batch is masked.

---

## 3. Initial-Condition Loss (`compute_ic_loss`)

**Purpose:** Anchor the network's prediction at `t = 0` to the prescribed initial state. Without this, the network could learn the correct dynamics but offset from the true starting point.

**Normalization:** The v and u error terms are divided by their characteristic scales (`V_SCALE = 95 mV`, `U_SCALE = 100 pA`) so that both contribute equally regardless of physical units.

**Note:** This is optional — the data loss from ground truth already contains points at `t = 0`, which implicitly enforces the initial condition. An explicit IC loss can improve early convergence.

---

## 4. Total Loss Structure (for Role 10)

```
L_total = L_data + λ_phys · L_physics + λ_ic · L_ic
```

| Term | Computed by | Description |
|---|---|---|
| `L_data` | Role 10 | MSE between network predictions and ground truth CSV |
| `L_physics` | `compute_physics_loss()` | Normalised ODE residual with spike masking |
| `L_ic` | `compute_ic_loss()` | Normalised initial-condition penalty |

**Recommended starting weights:**

| Hyperparameter | Starting value | Rationale |
|---|---|---|
| `λ_phys` | 0.01 | Physics residual is in (dimensionless)² after normalization; data loss is in (mV)². Scale difference requires a small initial weight. |
| `λ_ic` | 1.0 | IC loss is already normalised and cheap to compute. |

---

## 5. Interface Contract Summary

| Function | Input | Output | Consumer |
|---|---|---|---|
| `IzhikevichPINN.forward()` | `(N, 4)` tensor: `[t, I_ext, V_0, U_0]` | `(N, 2)` tensor: `[v, u]` in physical units | Physics loss, IC loss, training loop |
| `predict_trajectory()` | `t, I_ext, V_0, U_0` tensors | `(N, 3)` tensor: `[Time, v, u]` | Evaluation (Role 11) |
| `compute_physics_loss()` | model, `t` (requires_grad), `I_ext, V_0, U_0`, `peak_margin` | scalar tensor | Training loop (Role 10) |
| `compute_ic_loss()` | model, `I_ext_0, V_0, U_0` | scalar tensor | Training loop (Role 10) |
| `get_margin()` | `epoch, max_epochs` | float | Training loop (Role 10) |