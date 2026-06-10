# Role 7: ML Loss Function Designer — Documentation

## What You Must Document Here

### 1. Physics Loss Formulation
Write the exact residual equations encoded in the loss function:

- **Residual_V** = ... 
- **Residual_m** = ...
- **Residual_h** = ...
- **Residual_n** = ...

**Loss** = ...

### 2. Autograd Differentiation
- Explain how `torch.autograd.grad` is used to compute `dV/dt`, `dm/dt`, `dh/dt`, `dn/dt` from the model output with respect to the input time `t`.
- Emphasize that `t_collocation` must have `requires_grad=True`.

### 3. Config Constants Verification
Confirm that the following constants match `config.py` **exactly** (mirror Role 3):

| Constant Used in Loss | Value | Matches config.py? |
|------------------------|-------|---------------------|
| `C_M`  | 1.0   | ✅ / ❌ |
| `G_NA` | 120.0 | ✅ / ❌ |
| `G_K`  | 36.0  | ✅ / ❌ |
| ...    | ...   | ... |

### 4. NumPy → PyTorch Conversion Notes 
`(دا لو عملت كدا، لو عملت حاجة مختلفة اكتبها، دا مجرد placeholder)`
- Document how you re-implemented the `alpha_*` and `beta_*` gating functions using `torch.exp()` instead of `np.exp()` to maintain the computational graph.

---

> **Deliver to:** Role 8 (Training Operator) — The `compute_physics_loss()` function.  
> **Critical:** Must strictly mirror Role 3's constants from `config.py`.
