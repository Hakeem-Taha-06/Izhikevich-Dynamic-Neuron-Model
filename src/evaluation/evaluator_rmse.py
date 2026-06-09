import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports from config.py:
# from config import INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT

# Imports from other team modules:
# from src.numerical.baseline_lsoda import solve_baseline_lsoda

def generate_ground_truth(csv_save_path="data/ground_truth.csv", seed=42):
    """
    Role 9: Ground Truth Generation
    
    Objective:
    ----------
    Generates a reference trajectory dataset by executing the high-accuracy baseline 
    LSODA solver. Can introduce minor randomized initial condition variations if needed 
    for robust training validation, and saves the output in CSV format.

    Config Variables to Import:
    --------------------------
    - INITIAL_STATE: np.ndarray of shape (4,), containing default [V_0, M_0, H_0, N_0]
    - T_START: float, start time of simulation (e.g., 0.0 ms)
    - T_END: float, end time of simulation (e.g., 100.0 ms)
    - DT_EVAL: float, output time resolution (e.g., 0.01 ms)
    - I_EXT_DEFAULT: float, default external current injection (e.g., 10.0 uA/cm^2)

    Input Parameters:
    -----------------
    - csv_save_path : str
        The file system path where the ground truth CSV must be saved. Default: "data/ground_truth.csv".
    - seed : int, optional
        Random seed for reproducibility of any randomized initial conditions. Default: 42.

    Output:
    -------
    - ground_truth : np.ndarray
        NumPy array of shape (N, 5) representing the generated trajectory.
        Columns must be: [Time, Voltage, m, h, n]
        
    Side Effects:
    -------------
    - Saves the trajectory array as a CSV file with headers:
      `Time,Voltage,m,h,n` at the target `csv_save_path`.
    """
    pass

def evaluate_models(baseline_traj, rk4_traj, be_traj, pinn_traj, execution_times):
    """
    Role 9: Master Evaluator
    
    Objective:
    ----------
    Compares all numerical integration and machine learning solvers (RK4, Backward Euler, PINN) 
    against the baseline LSODA solver. Computes the Root Mean Squared Error (RMSE) for the 
    membrane potential (V) and all gating variables (m, h, n), compiles a performance benchmarking 
    table including execution times, and generates comparison plots of the trajectories.

    Input Parameters:
    -----------------
    - baseline_traj : np.ndarray
        Ground truth reference trajectory array of shape (N, 5).
    - rk4_traj : np.ndarray
        Explicit Runge-Kutta 4 trajectory array of shape (N, 5).
    - be_traj : np.ndarray
        Implicit Backward Euler trajectory array of shape (N, 5).
    - pinn_traj : np.ndarray
        Physics-Informed Neural Network predicted trajectory array of shape (N, 5).
    - execution_times : dict
        A dictionary mapping method names ('LSODA', 'RK4', 'Backward_Euler', 'PINN') 
        to their respective runtimes in milliseconds (float).

    Output:
    -------
    - metrics_df : pd.DataFrame
        A pandas DataFrame compiling RMSEs and execution times for all solvers.
        The DataFrame columns should strictly contain:
        ['Method', 'RMSE_V', 'RMSE_m', 'RMSE_h', 'RMSE_n', 'Execution_Time_ms']
        
    Side Effects:
    -------------
    - Generates and saves a trajectory comparison plot (e.g., "data/trajectory_comparison.png") 
      visualizing the voltage V and gating variables over time for all four methods.
    """
    pass
