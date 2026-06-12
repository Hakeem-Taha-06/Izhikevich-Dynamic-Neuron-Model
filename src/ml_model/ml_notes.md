# ML Module — Engineering Decisions & Rationale

**Roles covered:** 8 (Architecture), 9 (Physics Loss), 10 (Training Loop)  
**Author:** Hakeem  
**Last updated:** 2026-06-12

---

## 1. Architecture Design (`architecture.py`)

### 1.1 Network Topology

| Component | Choice | Justification |
|---|---|---|
| Input features | `[t]` — 1 input (time in ms) | The project now uses a single step-current protocol. The step-current `I_ext(t)` is handled inside the physics loss, not as a network input. |
| Output features | `[v, w]` — 2 outputs in physical units (mV, pA) | Directly consumed by the physics loss, IC loss, and evaluation pipeline. No downstream denormalization required. |
| Hidden layers | 6 layers × 128 neurons (default) | Deeper and wider than the original 3×64 to better capture the sharp spike transitions. Matches the comparison repo's architecture. |
| Activation | `Tanh` | Smooth, bounded in [-1, 1], infinitely differentiable. Required by the PINN approach — `torch.autograd.grad` computes dv/dt and dw/dt through the activation, so non-differentiable activations (ReLU) would produce piecewise-constant derivatives. |
| Output activation | None (linear) | The network must predict physical values across the full voltage range (−60 to 35 mV) and unbounded recovery variable. |
| Weight init | Xavier normal (gain=1.0) | Better gradient flow in deeper networks compared to default init. |
| Time normalization | `t_norm = (t - T_max/2) / (T_max/2)` → [-1, 1] | Centres the input around zero and bounds it to [-1, 1], matching the Tanh activation's sensitive region. |

### 1.2 Output Denormalization

**Problem:** The raw network output is centered near zero (Tanh hidden layers feed a final Linear layer with small initial weights). But the physical targets span very different ranges:
- `v` lives in [−60, 35] mV
- `w` can reach hundreds of pA

Without denormalization, the final Linear layer must learn large biases and weights to shift its output to the correct range, wasting early training epochs.

**Decision:** Apply a fixed affine transform inside `forward()`:

```
v = raw_v × V_SCALE + V_SHIFT
w = raw_w × W_SCALE + W_SHIFT
```

| Constant | Value | Derivation |
|---|---|---|
| `V_SHIFT` | −60.0 mV | `v_r` — resting potential |
| `V_SCALE` | 95.0 mV | `v_peak − v_r` — full operating range |
| `W_SHIFT` | 0.0 pA | Steady-state recovery at rest |
| `W_SCALE` | 100.0 pA | `|d|` — the spike-reset increment |

**Consequence:** At initialization (raw ≈ 0), the network predicts the resting state (`v ≈ −60 mV`, `w ≈ 0 pA`), which already satisfies the ODE at equilibrium. Training starts from a physically reasonable baseline rather than from random noise.

**Autograd compatibility:** The affine transform is inside `forward()`, so `torch.autograd.grad` chains through it automatically: `dv_physical/dt = dv_raw/dt × V_SCALE`. No special handling is needed in the physics loss.

The constants are stored as `register_buffer` (non-trainable, tracked by `state_dict`, moved with `.to(device)`).

### 1.3 Why Input Is Just `t` (Not `[t, I_ext, V_0, W_0]`)

The original architecture conditioned on `I_ext`, `V_0`, and `W_0` because it was designed for a 4,000-simulation sweep across 40 current levels × 20 initial voltages × 5 initial recovery values. After switching to the comparison repo's protocol (single step-current at 70 pA, fixed ICs), all three extra inputs became constants — conditioning on them adds unnecessary parameters without benefit.

The step-current `I_ext(t)` is now computed inside `physics_loss.py` via `I_ext_torch(t)`, which returns 0 for `t < 100 ms` and 70 pA for `t ≥ 100 ms`. This is a known function of `t`, not a learnable quantity.

---

## 2. Physics Loss Design (`physics_loss.py`)

### 2.1 Core Approach — ODE Residual

The physics loss penalises violations of the governing ODEs. For each collocation point `t`:

1. **Forward pass** → get `v_pred`, `w_pred` from model(t)
2. **Autograd** → compute `dv_pred/dt`, `dw_pred/dt` via `torch.autograd.grad`
3. **Step-current** → compute `I_ext(t)` internally via `I_ext_torch(t)`
4. **Residual** → the difference between the autograd derivative and the ODE right-hand side:
   - `R_v = dv/dt − [k(v−v_r)(v−v_t) − w + I_ext(t)] / C_m`
   - `R_w = dw/dt − a·[b(v−v_r) − w]`
5. **Loss** → mean squared normalised residual

### 2.2 Step-Current Function (`I_ext_torch`)

The external current follows the step-current protocol from `config.py`:

```python
def I_ext_torch(t):
    return I_EXT_DEFAULT * (t >= T_STIM_ONSET).float()
    # Returns 0.0 for t < 100 ms, 70.0 for t >= 100 ms
```

The result is detached from the computation graph (via `t.detach()`) since `I_ext` is an external forcing term, not a learnable quantity. Gradients should not flow through it.

### 2.3 Spike Masking (Peak-Only)

**Problem:** The discrete reset (`v ← c`, `w ← w + d` when `v ≥ v_peak`) creates a discontinuity that the continuous network cannot represent. Near the spike, `dv_pred/dt` approaches infinity, producing explosive residuals that crash training with NaN.

**Decision: Mask the fast-upstroke zone only.** Points where `v_pred > v_peak − peak_margin` are excluded from the physics loss.

**Why not mask near the reset voltage (`c = −50 mV`) as well?** Because `c = −50` is only 10 mV above the resting potential `v_r = −60`. A near-reset mask would exclude the entire subthreshold region where the neuron spends most of its time and the ODE is smooth and perfectly valid.

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

### 2.4 Curriculum Margin Schedule (`get_margin`)

**Decision:** The margin starts wide and tightens linearly over training epochs.

| Epoch | Margin | Masks v > | Physics-enforced range | Coverage |
|---|---|---|---|---|
| 0 (start) | 20.0 mV | 15 mV | −60 to 15 mV | 79% |
| mid | 12.5 mV | 22.5 mV | −60 to 22.5 mV | 87% |
| final | 5.0 mV | 30 mV | −60 to 30 mV | 95% |

**Rationale:** Early in training the network is untrained and the spike upstroke would produce catastrophic gradients. Starting with a wide margin keeps training stable. As the network learns the smooth dynamics, the margin tightens to enforce physics closer to the spike peak.

### 2.5 Residual Scale Normalization

**Problem:** The two residuals have different physical units and magnitudes:
- `R_v` is in mV/ms — typical magnitude ~5–10
- `R_w` is in pA/ms — typical magnitude ~1–3
- Without normalization, `mean(R_v²)` dominates `mean(R_w²)` by 10–25×

**Decision:** Divide each residual by a characteristic ODE rate before squaring:

| Constant | Value | Derivation |
|---|---|---|
| `DV_RATE` | 0.7 mV/ms | `I_EXT_DEFAULT / C_m = 70 / 100` — current-driven voltage rate at rest |
| `DW_RATE` | 3.0 pA/ms | `a × |d| = 0.03 × 100` — post-spike recovery rate |

Both terms become dimensionless and contribute roughly equally.

### 2.6 Mean Computation — Boolean Indexing

**Problem:** An earlier implementation used `torch.where(mask, zeros, R)` to zero out masked points. The zeroed entries still appeared in the denominator of `torch.mean`, diluting the loss.

**Decision:** Use boolean indexing on the valid (unmasked) subset:

```python
valid = ~(v_pred > (v_peak - peak_margin)).squeeze()
R_v_valid = R_v[valid]
loss = torch.mean((R_v_valid / DV_RATE) ** 2) + ...
```

An edge-case guard returns `0.0` if every point in the batch is masked.

---

## 3. Initial-Condition Loss (`compute_ic_loss`)

**Purpose:** Anchor the network's prediction at `t = 0` to the prescribed initial state from `config.INITIAL_STATE`. Without this, the network could learn the correct dynamics but offset from the true starting point.

**Simplified interface:** `compute_ic_loss(model)` — no parameters needed. The function reads the known ICs from config (`V_0 = -60.0`, `W_0 = 0.0`) and evaluates the network at `t = 0`.

**Normalization:** The v and w error terms are divided by their characteristic scales (`V_SCALE = 95 mV`, `W_SCALE = 100 pA`) so that both contribute equally regardless of physical units.

---

## 4. Total Loss Structure

```
L_total = λ_data · L_data + λ_phys · L_physics + λ_ic · L_ic
```

| Term | Computed by | Description |
|---|---|---|
| `L_data` | Role 10 | MSE between model(t) predictions and ground truth [v, w] |
| `L_physics` | `compute_physics_loss(model, t)` | Normalised ODE residual with spike masking |
| `L_ic` | `compute_ic_loss(model)` | Normalised initial-condition penalty |

**Loss weights (aligned with comparison repo):**

| Hyperparameter | Value | Rationale |
|---|---|---|
| `λ_data` | 50.0 | Strong data anchoring — the supervised data teaches the spike resets that physics alone cannot handle |
| `λ_phys` | 1.0 | Physics residual is dimensionless after normalization |
| `λ_ic` | 200.0 | Very strong IC anchoring to prevent drift (comparison repo uses 200) |

---

## 5. Interface Contract Summary

| Function | Input | Output | Consumer |
|---|---|---|---|
| `IzhikevichPINN.forward()` | `(N, 1)` tensor: `[t]` | `(N, 2)` tensor: `[v, w]` in physical units | Physics loss, IC loss, training loop |
| `predict_trajectory()` | `t_start, t_end, dt` (optional, defaults from config) | `(N, 3)` numpy array: `[Time, v, w]` | Evaluation (Role 11) |
| `compute_physics_loss()` | `model`, `t` (requires_grad), `peak_margin` | scalar tensor | Training loop (Role 10) |
| `compute_ic_loss()` | `model` | scalar tensor | Training loop (Role 10) |
| `get_margin()` | `epoch, max_epochs` | float | Training loop (Role 10) |
| `I_ext_torch()` | `t` (any shape tensor) | same-shape tensor (pA) | Internal to physics loss |

---

## 6. Training Loop Design (`train.py`)

### 6.1 Data Ingestion — `IzhikevichDataset`

**Simplified structure:** The dataset now returns `(t, v_gt, w_gt)` per sample — just time and ground truth values. No `I_ext`, `V_0`, `W_0` columns are needed since the network takes only `t` as input.

The ground truth CSV schema remains: `Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)`. The dataset reads `Time`, `v`, and `w` columns and ignores `I_ext` and `Sim_ID`.

### 6.2 Two-Phase Optimisation

| Phase | Optimiser | Epochs | Batch strategy | Purpose |
|---|---|---|---|---|
| 1 | Adam | 8,000 | Mini-batch (`DataLoader`, 4096/batch) | Fast initial convergence — SGD noise helps escape local minima |
| 2 | L-BFGS | 100 | Full-batch (random 16K-point sample) | Precise refinement — quasi-Newton method exploits second-order curvature |

**Why this split?** Adam converges quickly to a rough solution but plateaus. L-BFGS uses Hessian approximations to polish the solution. The comparison repo uses the same Adam → L-BFGS pattern.

### 6.3 Mini-Batching (Phase 1)

Uses `torch.utils.data.DataLoader` with `batch_size=4096` and `shuffle=True`. Each batch independently computes:

```
L_batch = λ_data · L_data + λ_phys · L_physics + λ_ic · L_ic
```

A critical detail: `t` must be cloned and have `requires_grad_(True)` set **per batch**, because the DataLoader's collated tensors are leaf tensors without gradient tracking.

**Gradient clipping:** `clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents gradient explosions from the spike dynamics.

### 6.4 L-BFGS Mechanics (Phase 2)

Draws a fixed random sample of 16,384 points from the dataset and reuses it for all L-BFGS epochs. The closure function computes the full loss (data + physics + IC) and backpropagates.

`line_search_fn="strong_wolfe"` ensures sufficient decrease and curvature conditions.

### 6.5 Hyperparameter Defaults

| Parameter | Default | Rationale |
|---|---|---|
| `adam_epochs` | 8,000 | Matches comparison repo |
| `lbfgs_epochs` | 100 | Matches comparison repo |
| `batch_size` | 4,096 | Balances GPU utilisation vs gradient noise |
| `lr_adam` | 1e-3 | Standard Adam learning rate |
| `lr_lbfgs` | 0.1 | Comparison repo default |
| `lbfgs_sample_size` | 16,384 | Representative coverage, memory-safe |
| `λ_data` | 50.0 | Comparison repo uses 50 |
| `λ_phys` | 1.0 | Physics residual is dimensionless |
| `λ_ic` | 200.0 | Comparison repo uses 200 |

### 6.6 Final Inference

After training, `predict_trajectory(model)` generates the trajectory over the full time range using defaults from `config.py`. Returns `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, w]`.