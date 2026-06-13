"""
Faculty of Engineering, Cairo University
Systems and Biomedical Engineering Department
Course: SBEG108 - Numerical Methods in Biomedical Engineering

Role 10: ML Training Loop (Fully Operational Implementation)

Training pipeline for the Izhikevich PINN model:
  Phase 1 — Adam optimiser with full-batch gradient descent.
  Phase 2 — L-BFGS quasi-Newton refinement on the same data.

Ground truth CSV schema (from Role 4 / ROLES.md):
    Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)

The network takes only time (t) as input and predicts [v(t), w(t)].
The step-current protocol I_ext(t) is handled internally by the
physics loss (Role 9).

Sparse Data / Dense Physics strategy:
  - Physics loss is computed on ALL ~100K collocation points.
  - Data loss is computed on a SPARSE subset (every Nth point).
  This forces the network to learn the ODE dynamics from physics,
  with sparse data anchors teaching the spike resets.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add root folder and src folder to sys path for cross-module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Global Dependencies & Shared Constants from config.py
try:
    from config import (
        INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
        T_STIM_ONSET, I_ext_fn,
        C_m, k, v_r, v_t, v_peak, a, b, c, d
    )
except ImportError:
    # Safe fallback configurations
    INITIAL_STATE = np.array([-60.0, 0.0])
    T_START, T_END, DT_EVAL = 0.0, 1000.0, 0.01
    I_EXT_DEFAULT = 70.0
    T_STIM_ONSET = 100.0
    C_m, k, v_r, v_t, v_peak = 100.0, 0.7, -60.0, -40.0, 35.0
    a, b, c, d = 0.03, -2.0, -50.0, 100.0

# Import teammates' specialized modules
try:
    from ml_model.architecture import IzhikevichPINN, predict_trajectory
    from ml_model.physics_loss import compute_physics_loss, compute_ic_loss, get_margin
except ImportError:
    from architecture import IzhikevichPINN, predict_trajectory
    from physics_loss import compute_physics_loss, compute_ic_loss, get_margin


# =====================================================================
# 1. CSV LOADER
# =====================================================================

def load_ground_truth(csv_path, device):
    """Load ground truth CSV and return tensors on the target device.

    Parameters
    ----------
    csv_path : str
        Path to the CSV with columns: Sim_ID, Time (ms), I_ext (pA),
        v (mV), w (pA).
    device : torch.device
        Target device (cpu or cuda).

    Returns
    -------
    t : torch.Tensor, shape (N, 1)
    v_gt : torch.Tensor, shape (N, 1)
    w_gt : torch.Tensor, shape (N, 1)
    """
    df = pd.read_csv(csv_path)
    t    = torch.tensor(df["Time (ms)"].values, dtype=torch.float32).unsqueeze(1).to(device)
    v_gt = torch.tensor(df["v (mV)"].values,    dtype=torch.float32).unsqueeze(1).to(device)
    w_gt = torch.tensor(df["w (pA)"].values,    dtype=torch.float32).unsqueeze(1).to(device)
    return t, v_gt, w_gt


# =====================================================================
# 2. TRAINING PIPELINE
# =====================================================================

def execute_ml_training_pipeline(
    ground_truth_csv_path: str,
    model_save_path: str = "outputs/models/pinn_model.pt",
    adam_epochs: int = 8000,
    lbfgs_epochs: int = 100,
    lr_adam: float = 1e-3,
    lr_lbfgs: float = 0.1,
    data_subsample_factor: int = 100,
) -> np.ndarray:
    """Execute the full PINN training pipeline.

    Both phases use full-batch training.  The dataset (~100K points)
    fits comfortably in GPU memory (~1 GB peak on an RTX 3050 6GB).

    Phase 1 — Adam: Full-batch gradient descent for fast convergence.
    Phase 2 — L-BFGS: Full-batch quasi-Newton refinement.

    Parameters
    ----------
    ground_truth_csv_path : str
        Path to the ground truth CSV produced by Role 4.
    model_save_path : str
        Where to save the trained .pt weights.
    adam_epochs : int
        Number of Adam training epochs (default 8000).
    lbfgs_epochs : int
        Number of L-BFGS refinement epochs (default 100).
    lr_adam : float
        Learning rate for Adam (default 1e-3).
    lr_lbfgs : float
        Learning rate for L-BFGS (default 0.1).
    data_subsample_factor : int
        Keep every Nth data point for supervised loss (default 100).
        Factor=1 uses all points (no subsampling).

    Returns
    -------
    results : np.ndarray, shape (N, 3)
        Final predicted trajectory [Time, v, w].
    """
    total_epochs = adam_epochs + lbfgs_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Data Ingestion ----
    print(f"Loading ground truth from: {ground_truth_csv_path}")
    all_t, all_v_gt, all_w_gt = load_ground_truth(ground_truth_csv_path, device)
    n_total = len(all_t)
    print(f"Total samples: {n_total:,}")

    # ---- 2. Split: sparse data vs dense collocation ----
    # Collocation points (ALL time points — for physics loss)
    # requires_grad is set per-step so autograd can compute dv/dt
    colloc_t = all_t                                        # (N, 1)

    # Sparse data points (every Nth — for supervised data loss)
    sparse_idx = torch.arange(0, n_total, data_subsample_factor)
    data_t       = all_t[sparse_idx]                         # (M, 1)
    data_targets = torch.cat([
        all_v_gt[sparse_idx], all_w_gt[sparse_idx]
    ], dim=1)                                                # (M, 2)
    n_data = len(sparse_idx)

    print(f"Physics collocation points: {n_total:,} (all)")
    print(f"Data supervision points:    {n_data:,} (every {data_subsample_factor}th)")
    print(f"Data/Physics ratio:         1:{data_subsample_factor}")

    # ---- 3. Model Setup ----
    model = IzhikevichPINN().to(device)
    mse_criterion = nn.MSELoss()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss weights
    lambda_phys = 1.0
    lambda_ic = 200.0
    lambda_data = 50.0

    # ==================================================================
    # Phase 1 — Adam (full-batch)
    # ==================================================================
    optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)
    model.train()
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Adam ({adam_epochs} epochs, full-batch)")
    print(f"  Physics: {n_total:,} collocation pts")
    print(f"  Data:    {n_data:,} sparse pts")
    print(f"{'=' * 60}")

    for epoch in range(1, adam_epochs + 1):
        # Enable grad tracking on t for autograd inside physics loss
        t_phys = colloc_t.clone().requires_grad_(True)

        # ── Physics loss (all collocation points) ─────────────
        margin = get_margin(epoch, total_epochs)
        loss_phys = compute_physics_loss(
            model, t_phys, peak_margin=margin
        )

        # ── Data loss (sparse observations) ───────────────────
        pred_data = model(data_t)                         # (M, 2)
        loss_data = mse_criterion(pred_data, data_targets)

        # ── IC loss ───────────────────────────────────────────
        loss_ic = compute_ic_loss(model)

        # ── Total ─────────────────────────────────────────────
        total_loss = (
            lambda_data * loss_data
            + lambda_phys * loss_phys
            + lambda_ic * loss_ic
        )

        optimizer_adam.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()

        if epoch % 2000 == 0 or epoch == 1:
            print(
                f"  Epoch [{epoch:4d}/{adam_epochs}]  "
                f"Total: {total_loss.item():.4e}  "
                f"Data: {loss_data.item():.4e}  "
                f"Phys: {loss_phys.item():.4e}  "
                f"IC: {loss_ic.item():.4e}  "
                f"Margin: {margin:.1f}"
            )

    # ==================================================================
    # Phase 2 — L-BFGS (full-batch, same data)
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 2: L-BFGS refinement ({lbfgs_epochs} epochs, full-batch)")
    print(f"{'=' * 60}")

    # L-BFGS uses the SAME collocation and data tensors as Adam.
    # No separate subsampling needed — 100K points fit in memory.
    lbfgs_t = colloc_t.clone().requires_grad_(True)

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), lr=lr_lbfgs,
        max_iter=20, history_size=50,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(1, lbfgs_epochs + 1):

        def closure():
            optimizer_lbfgs.zero_grad()

            margin = get_margin(adam_epochs + epoch, total_epochs)
            loss_phys = compute_physics_loss(
                model, lbfgs_t, peak_margin=margin,
            )

            pred_data = model(data_t)
            loss_data = mse_criterion(pred_data, data_targets)

            loss_ic = compute_ic_loss(model)

            total = (
                lambda_data * loss_data
                + lambda_phys * loss_phys
                + lambda_ic * loss_ic
            )
            total.backward()
            return total

        loss_val = optimizer_lbfgs.step(closure)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  L-BFGS [{epoch:4d}/{lbfgs_epochs}]  Loss: {loss_val.item():.6f}")

    # ==================================================================
    # Save & Inference
    # ==================================================================
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # Final inference: generate the default trajectory
    model.eval()
    results = predict_trajectory(model)

    print(f"Output trajectory shape: {results.shape}  (expected (N, 3))")
    return results


# =====================================================================
# 3. LOCAL TEST
# =====================================================================

if __name__ == "__main__":
    print("--- Running Local Compilation Test ---\n")
    mock_csv = "temp_ground_truth.csv"

    # Generate a small mock CSV matching the real schema:
    #   Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)
    rows = []
    test_t = np.arange(T_START, min(T_END, 200.0) + DT_EVAL, DT_EVAL)
    for t_val in test_t:
        I_val = float(I_ext_fn(t_val))
        rows.append({
            "Sim_ID": 1,
            "Time (ms)": round(t_val, 4),
            "I_ext (pA)": I_val,
            "v (mV)": np.sin(t_val / 10.0) * 15 - 55,
            "w (pA)": np.cos(t_val / 10.0) * 2 - 10,
        })
    pd.DataFrame(rows).to_csv(mock_csv, index=False)

    try:
        output = execute_ml_training_pipeline(
            ground_truth_csv_path=mock_csv,
            model_save_path="outputs/models/pinn_model.pt",
            adam_epochs=50,
            lbfgs_epochs=10,
            data_subsample_factor=100,
        )
        print(f"\n[OK] SUCCESS! Output shape: {output.shape}")
        print(f"First row [Time, v, w]: {output[0]}")
    finally:
        if os.path.exists(mock_csv):
            os.remove(mock_csv)