"""
Faculty of Engineering, Cairo University
Systems and Biomedical Engineering Department
Course: SBEG108 - Numerical Methods in Biomedical Engineering

Role 10: ML Training Workspace (Fully Operational Implementation)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add root folder and src folder to sys path to ensure clean cross-module imports
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
    T_START, T_END, DT_EVAL = 0.0, 100.0, 0.01
    I_EXT_DEFAULT = 500.0
    C_m, k, v_r, v_t, v_peak = 100.0, 0.7, -60.0, -40.0, 35.0
    a, b, c, d = 0.03, -2.0, -50.0, 100.0

# Import your teammates' specialized modules
try:
    from ml_model.architecture import IzhikevichPINN, predict_trajectory
    from ml_model.physics_loss import compute_physics_loss, compute_ic_loss, get_margin
except ImportError:
    # If paths are flat or run from inside src/ml_model directly
    from architecture import IzhikevichPINN, predict_trajectory
    from physics_loss import compute_physics_loss, compute_ic_loss, get_margin


def execute_ml_training_pipeline(
    ground_truth_csv_path: str, 
    model_save_path: str = "izhikevich_weights.pth", 
    epochs: int = 5000, 
    lr: float = 0.001
) -> np.ndarray:
    """
    Executes the training loop by combining Data-Driven Loss, Physics-Informed 
    ODE Residual Losses, and Initial Condition Anchors.
    """
    # ---- 1. Grid Generation ----
    time_steps = np.arange(T_START, T_END + DT_EVAL, DT_EVAL)
    N = len(time_steps)
    
    # Core tracking time vector (requires_grad=True is critical for physics autograd)
    t_tensor = torch.tensor(time_steps, dtype=torch.float32).view(-1, 1)
    t_tensor.requires_grad_(True)

    # ---- 2. Data Ingestion ----
    try:
        df = pd.read_csv(ground_truth_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing ground truth data file at: {ground_truth_csv_path}")

    v_target = torch.tensor(df['v'].values, dtype=torch.float32).view(-1, 1)
    u_target = torch.tensor(df['u'].values, dtype=torch.float32).view(-1, 1)
    targets = torch.cat((v_target, u_target), dim=1)

    # ---- 3. Build the 4-Column Model Inputs ----
    # Architecture expects: [t, I_ext, V_0, U_0] across all N rows
    I_ext_tensor = torch.full_like(t_tensor, I_EXT_DEFAULT)
    V_0_tensor = torch.full_like(t_tensor, INITIAL_STATE[0])
    U_0_tensor = torch.full_like(t_tensor, INITIAL_STATE[1])
    
    # Combine them to match the exact input interface contract of Role 8
    model_inputs = torch.cat([t_tensor, I_ext_tensor, V_0_tensor, U_0_tensor], dim=1)

    # ---- 4. Optimization Setup ----
    model = IzhikevichPINN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    # Scaling weights (hyperparameters) recommended by Role 9 documentation
    lambda_phys = 0.01
    lambda_ic = 1.0

    # ---- 5. The PINN Training Loop ----
    model.train()
    print(f"Starting PINN training engine for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # 1. Forward pass (Get output predictions in physical units)
        predictions = model(model_inputs)
        
        # 2. Compute Supervised Data Loss
        loss_data = mse_criterion(predictions, targets)
        
        # 3. Compute Physics ODE Residual Loss with Curriculum Schedule Margin
        margin = get_margin(epoch, epochs)
        loss_phys = compute_physics_loss(
            model, t_tensor, I_ext_tensor, V_0_tensor, U_0_tensor, peak_margin=margin
        )
        
        # 4. Compute Initial Condition Anchor Loss
        loss_ic = compute_ic_loss(model, I_ext_tensor, V_0_tensor, U_0_tensor)
        
        # 5. Combine everything into Total Loss Balance
        total_loss = loss_data + (lambda_phys * loss_phys) + (lambda_ic * loss_ic)
        
        # 6. Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print neat updates every 1000 steps
        if epoch % 1000 == 0 or epoch == 1:
            print(f"Epoch [{epoch:4d}/{epochs}] | Loss Data: {loss_data.item():.4f} | Loss Phys: {loss_phys.item():.4f} | Total: {total_loss.item():.4f}")

    # ---- 6. Code Output & Side Effects ----
    # Save optimized brain parameters to disk
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters successfully saved to: {model_save_path}")

    # Final inference configuration to extract trajectory mapping
    model.eval()
    with torch.no_grad():
        # Call the trajectory helper built inside architecture.py by Role 8
        final_trajectory_tensor = predict_trajectory(
            model, t_tensor, I_ext_tensor, V_0_tensor, U_0_tensor
        )
        # Convert output from PyTorch matrix back to NumPy array
        results = final_trajectory_tensor.numpy()

    # Returns the strict project interface rule shape (N, 3) ordered as [Time, v, u]
    return results


if __name__ == "__main__":
    print("--- Running Local Code Compilation Test ---")
    mock_csv = "temp_ground_truth.csv"
    
    # Generate mock CSV matching our config metrics to test code execution
    test_t = np.arange(T_START, T_END + DT_EVAL, DT_EVAL)
    df_mock = pd.DataFrame({
        'time': test_t,
        'v': np.sin(test_t / 10.0) * 15 - 55,  
        'u': np.cos(test_t / 10.0) * 2 - 10
    })
    df_mock.to_csv(mock_csv, index=False)
    
    try:
        # Run pipeline for 1000 sample loops to verify autograd mechanics work completely
        output = execute_ml_training_pipeline(
            ground_truth_csv_path=mock_csv,
            model_save_path="outputs/models/pinn_model.pt",
            epochs=1000,
            lr=0.001
        )
        print("\n🟢 SUCCESS! Training Loop executed cleanly.")
        print(f"Output Matrix Shape: {output.shape} -> Expected (N, 3)")
        print(f"First data sample row [Time, v, u]: {output[0]}")
    finally:
        if os.path.exists(mock_csv):
            os.remove(mock_csv)