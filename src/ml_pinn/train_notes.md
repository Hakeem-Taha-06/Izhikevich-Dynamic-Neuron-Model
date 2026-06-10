# Role 8: ML Training Loop Operator — Documentation

## What You Must Document Here

### 1. Training Configuration Table
| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimizer      | Adam / ? | ... |
| Learning Rate  | 1e-3 / ? | ... |
| Epochs         | 5000 / ? | ... |
| Batch Size     | Full / ? | ... |
| Loss Weighting | `w_data * L_data + w_physics * L_physics` | Document weights used |

### 2. Training Data Pipeline
- Describe how the ground truth CSV (from Role 9) is loaded and converted to `torch.Tensor`.
- Describe how collocation points are sampled (uniformly, randomly, etc.).
- Note the initial condition enforcement strategy.

### 3. Convergence Report
- Document the final training loss value and how many epochs it took to converge.
- Include a training loss curve (save the plot to `outputs/figures/`).

| Epoch | Total Loss | Data Loss | Physics Loss |
|-------|-----------|-----------|--------------|
| 0     | ...       | ...       | ...          |
| 1000  | ...       | ...       | ...          |
| ...   | ...       | ...       | ...          |
| Final | ...       | ...       | ...          |

### 4. Model Export
- The trained model weights are saved to `outputs/models/pinn_model.pt`.
- See `outputs/models/MODELS_README.md` for the weight-sharing protocol.

---

> **Deliver to:** Roles 9 & 10 — The PINN's predicted trajectory array (shape `(N, 5)`).  
> **Save to:** `outputs/models/pinn_model.pt` — The trained model weights.
