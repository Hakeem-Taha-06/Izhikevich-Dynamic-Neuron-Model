"""
Faculty of Engineering, Cairo University
Systems and Biomedical Engineering Department
Course: SBEG108 - Numerical Methods in Biomedical Engineering

Role 10: ML Training Loop (Fully Operational Implementation)

Training pipeline for the Izhikevich PINN model:
  Phase 1 — Adam optimiser with mini-batch SGD for fast convergence.
  Phase 2 — L-BFGS quasi-Newton refinement on a fixed sample for
            precise final fitting.

Ground truth CSV schema (from Role 4 / ROLES.md):
    Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)

The pipeline extracts per-simulation initial conditions (V_0, W_0)
automatically from the first time-step of each Sim_ID, so that the
network receives the correct [t, I_ext, V_0, W_0] inputs matching
the architecture contract (Role 8).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add root folder and src folder to sys path for cross-module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Global Dependencies & Shared Constants from config.py
try:
    from config import (
        INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
        C_m, k, v_r, v_t, v_peak, a, b, c, d
    )
except ImportError:
    # Safe fallback configurations
    INITIAL_STATE = np.array([-60.0, 0.0])
    T_START, T_END, DT_EVAL = 0.0, 1000.0, 0.01
    I_EXT_DEFAULT = 70.0
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
# 1. DATASET
# =====================================================================

class IzhikevichDataset(Dataset):
    """PyTorch Dataset wrapping the ground truth CSV from Role 4.

    Each sample is a single time-step row from the multi-simulation
    dataset, enriched with the initial conditions (V_0, W_0) of its
    parent simulation.

    Ground truth CSV schema (from ROLES.md / ground_truth_generator.py):
        Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)
    """

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Extract initial conditions for each simulation.
        # V_0 and W_0 are the v and w values at the earliest time step
        # of each Sim_ID.
        ic = (
            df.sort_values("Time (ms)")
              .groupby("Sim_ID", sort=False)
              .first()[["v (mV)", "w (pA)"]]
              .rename(columns={"v (mV)": "V_0", "w (pA)": "W_0"})
        )
        df = df.merge(ic, on="Sim_ID")

        # Pre-convert all columns to tensors for fast __getitem__
        self.t     = torch.tensor(df["Time (ms)"].values,  dtype=torch.float32).unsqueeze(1)
        self.I_ext = torch.tensor(df["I_ext (pA)"].values, dtype=torch.float32).unsqueeze(1)
        self.V_0   = torch.tensor(df["V_0"].values,        dtype=torch.float32).unsqueeze(1)
        self.W_0   = torch.tensor(df["W_0"].values,        dtype=torch.float32).unsqueeze(1)
        self.v_gt  = torch.tensor(df["v (mV)"].values,     dtype=torch.float32).unsqueeze(1)
        self.w_gt  = torch.tensor(df["w (pA)"].values,     dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return (self.t[idx], self.I_ext[idx], self.V_0[idx], self.W_0[idx],
                self.v_gt[idx], self.w_gt[idx])


# =====================================================================
# 2. TRAINING PIPELINE
# =====================================================================

def execute_ml_training_pipeline(
    ground_truth_csv_path: str,
    model_save_path: str = "outputs/models/pinn_model.pt",
    adam_epochs: int = 4000,
    lbfgs_epochs: int = 1000,
    batch_size: int = 4096,
    lr_adam: float = 1e-3,
    lr_lbfgs: float = 1.0,
    lbfgs_sample_size: int = 16384,
) -> np.ndarray:
    """Execute the full PINN training pipeline.

    Phase 1 — Adam: Mini-batch SGD for fast initial convergence.
    Phase 2 — L-BFGS: Full-batch quasi-Newton refinement on a fixed
              random sample of collocation points.

    Parameters
    ----------
    ground_truth_csv_path : str
        Path to the ground truth CSV produced by Role 4.
    model_save_path : str
        Where to save the trained .pt weights.
    adam_epochs : int
        Number of Adam training epochs (default 4000).
    lbfgs_epochs : int
        Number of L-BFGS refinement epochs (default 1000).
    batch_size : int
        Mini-batch size for the Adam phase (default 4096).
    lr_adam : float
        Learning rate for Adam (default 1e-3).
    lr_lbfgs : float
        Learning rate for L-BFGS (default 1.0).
    lbfgs_sample_size : int
        Number of points sampled for L-BFGS full-batch phase
        (default 16384).  The full dataset is too large for L-BFGS.

    Returns
    -------
    results : np.ndarray, shape (N, 3)
        Final predicted trajectory [Time, v, w] for the default
        simulation configuration (I_EXT_DEFAULT, INITIAL_STATE).
    """
    total_epochs = adam_epochs + lbfgs_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Data Ingestion ----
    print(f"Loading ground truth from: {ground_truth_csv_path}")
    dataset = IzhikevichDataset(ground_truth_csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset: {len(dataset):,} samples loaded.")

    # ---- 2. Model Setup ----
    model = IzhikevichPINN().to(device)
    mse_criterion = nn.MSELoss()

    # Loss weights (from Role 9 guidance in physics_loss.py / ml_notes.md)
    lambda_phys = 0.01
    lambda_ic = 1.0

    # ==================================================================
    # Phase 1 — Adam
    # ==================================================================
    optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)
    model.train()
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Adam optimiser ({adam_epochs} epochs, batch_size={batch_size})")
    print(f"{'=' * 60}")

    for epoch in range(1, adam_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            t, I_ext, V_0, W_0, v_gt, w_gt = [x.to(device) for x in batch]

            # t must have requires_grad for autograd inside physics loss
            t = t.clone().requires_grad_(True)

            # Forward pass
            inputs = torch.cat([t, I_ext, V_0, W_0], dim=1)
            predictions = model(inputs)                           # (B, 2)
            targets = torch.cat([v_gt, w_gt], dim=1)              # (B, 2)

            # Data loss
            loss_data = mse_criterion(predictions, targets)

            # Physics loss with curriculum margin
            margin = get_margin(epoch, total_epochs)
            loss_phys = compute_physics_loss(
                model, t, I_ext, V_0, W_0, peak_margin=margin
            )

            # IC loss
            loss_ic = compute_ic_loss(model, I_ext, V_0, W_0)

            # Total
            total_loss = loss_data + lambda_phys * loss_phys + lambda_ic * loss_ic

            optimizer_adam.zero_grad()
            total_loss.backward()
            optimizer_adam.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        if epoch % 500 == 0 or epoch == 1:
            avg = epoch_loss / max(n_batches, 1)
            print(f"  Epoch [{epoch:4d}/{adam_epochs}]  Avg Loss: {avg:.6f}  "
                  f"Margin: {margin:.1f} mV")

    # ==================================================================
    # Phase 2 — L-BFGS
    # ==================================================================
    # L-BFGS is a full-batch optimiser.  The full dataset (~40 M rows)
    # is too large, so we draw a fixed random sample.
    print(f"\n{'=' * 60}")
    print(f"Phase 2: L-BFGS refinement ({lbfgs_epochs} epochs, "
          f"sample={lbfgs_sample_size})")
    print(f"{'=' * 60}")

    n_sample = min(lbfgs_sample_size, len(dataset))
    indices = torch.randperm(len(dataset))[:n_sample]

    # Build fixed L-BFGS tensors by indexing into the pre-computed
    # dataset tensors (fast — no per-item __getitem__ overhead).
    lbfgs_t     = dataset.t[indices].clone().to(device).requires_grad_(True)
    lbfgs_I_ext = dataset.I_ext[indices].to(device)
    lbfgs_V_0   = dataset.V_0[indices].to(device)
    lbfgs_W_0   = dataset.W_0[indices].to(device)
    lbfgs_v_gt  = dataset.v_gt[indices].to(device)
    lbfgs_w_gt  = dataset.w_gt[indices].to(device)

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), lr=lr_lbfgs,
        max_iter=20, history_size=50,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(1, lbfgs_epochs + 1):

        def closure():
            """L-BFGS requires a closure that recomputes the loss."""
            optimizer_lbfgs.zero_grad()

            inputs = torch.cat([lbfgs_t, lbfgs_I_ext,
                                lbfgs_V_0, lbfgs_W_0], dim=1)
            predictions = model(inputs)
            targets = torch.cat([lbfgs_v_gt, lbfgs_w_gt], dim=1)

            loss_data = mse_criterion(predictions, targets)

            margin = get_margin(adam_epochs + epoch, total_epochs)
            loss_phys = compute_physics_loss(
                model, lbfgs_t, lbfgs_I_ext, lbfgs_V_0, lbfgs_W_0,
                peak_margin=margin,
            )
            loss_ic = compute_ic_loss(
                model, lbfgs_I_ext, lbfgs_V_0, lbfgs_W_0,
            )

            total = loss_data + lambda_phys * loss_phys + lambda_ic * loss_ic
            total.backward()
            return total

        loss_val = optimizer_lbfgs.step(closure)

        if epoch % 100 == 0 or epoch == 1:
            print(f"  L-BFGS [{epoch:4d}/{lbfgs_epochs}]  Loss: {loss_val.item():.6f}")

    # ==================================================================
    # Save & Inference
    # ==================================================================
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # Final inference: generate the default trajectory for evaluation
    model.eval()
    time_steps = torch.arange(
        T_START, T_END + DT_EVAL, DT_EVAL,
        dtype=torch.float32, device=device,
    ).unsqueeze(1)
    I_ext_inf = torch.full_like(time_steps, I_EXT_DEFAULT)
    V_0_inf   = torch.full_like(time_steps, INITIAL_STATE[0])
    W_0_inf   = torch.full_like(time_steps, INITIAL_STATE[1])

    with torch.no_grad():
        trajectory = predict_trajectory(model, time_steps, I_ext_inf, V_0_inf, W_0_inf)
        results = trajectory.cpu().numpy()

    # Returns the strict project interface: shape (N, 3) as [Time, v, w]
    print(f"Output trajectory shape: {results.shape}  (expected (N, 3))")
    return results


# =====================================================================
# 3. LOCAL TEST
# =====================================================================

if __name__ == "__main__":
    print("--- Running Local Compilation Test ---\n")
    mock_csv = "temp_ground_truth.csv"

    # Generate a small mock CSV matching the real schema exactly:
    #   Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA)
    rows = []
    for sim_id, I_ext_val in enumerate([250.0, 500.0], start=1):
        test_t = np.arange(T_START, T_END + DT_EVAL, DT_EVAL)
        for t_val in test_t:
            rows.append({
                "Sim_ID": sim_id,
                "Time (ms)": round(t_val, 4),
                "I_ext (pA)": I_ext_val,
                "v (mV)": np.sin(t_val / 10.0) * 15 - 55,
                "w (pA)": np.cos(t_val / 10.0) * 2 - 10,
            })
    pd.DataFrame(rows).to_csv(mock_csv, index=False)

    try:
        output = execute_ml_training_pipeline(
            ground_truth_csv_path=mock_csv,
            model_save_path="outputs/models/pinn_model.pt",
            adam_epochs=50,         # small for testing
            lbfgs_epochs=10,        # small for testing
            batch_size=2048,
        )
        print(f"\n[OK] SUCCESS! Output shape: {output.shape}")
        print(f"First row [Time, v, w]: {output[0]}")
    finally:
        if os.path.exists(mock_csv):
            os.remove(mock_csv)