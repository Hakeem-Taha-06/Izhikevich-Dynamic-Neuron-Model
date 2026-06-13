import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.ml_model.architecture import IzhikevichPINN, predict_trajectory

# Configuration
MODEL_PATH = "outputs/models/pinn_model_dsf100.pt"
GT_PATH = "data/ground_truth.csv"

def verify():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load Ground Truth
    df = pd.read_csv(GT_PATH)
    t_gt = df["Time (ms)"].values
    v_gt = df["v (mV)"].values

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IzhikevichPINN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Predict
    print("Generating prediction...")
    results = predict_trajectory(model)
    t_pred = results[:, 0]
    v_pred = results[:, 1]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t_gt, v_gt, 'k--', lw=2, label="Ground Truth (Sparse Anchor Source)")
    plt.plot(t_pred, v_pred, 'r-', lw=1.5, alpha=0.8, label="PINN Prediction")
    
    plt.title("PINN Verification (dsf=100)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.grid(True)
    
    save_path = "outputs/figures/verify_pinn.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Verification plot saved to: {save_path}")
    
    # Show plot interactively if running locally
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    verify()
