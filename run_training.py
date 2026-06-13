"""Launch PINN training with the specified data_subsample_factor.

Saves the model with the factor appended to the filename:
    outputs/models/pinn_model_dsf{factor}.pt
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.ml_model.train import execute_ml_training_pipeline

DATA_SUBSAMPLE_FACTOR = 100

results = execute_ml_training_pipeline(
    ground_truth_csv_path="data/ground_truth.csv",
    model_save_path=f"outputs/models/pinn_model_dsf{DATA_SUBSAMPLE_FACTOR}.pt",
    adam_epochs=8000,
    lbfgs_epochs=100,
    lr_adam=1e-3,
    lr_lbfgs=0.1,
    data_subsample_factor=DATA_SUBSAMPLE_FACTOR,
)

print(f"\nTraining complete. Model saved as: pinn_model_dsf{DATA_SUBSAMPLE_FACTOR}.pt")
print(f"Result shape: {results.shape}")
