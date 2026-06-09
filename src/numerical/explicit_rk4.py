import numpy as np

# Imports from config.py:
# from config import (
#     C_M, G_NA, G_K, G_L, E_NA, E_K, E_L,
#     INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
#     alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
# )

def solve_explicit_rk4(y0=None, t_span=None, dt=None, I_ext=None):
    """
    Role 4: Explicit Solver Developer (RK4)
    
    Objective:
    ----------
    Solves the Hodgkin-Huxley system of ODEs using a custom implementation of the 
    explicit 4th-Order Runge-Kutta (RK4) numerical integration method.
    The module should also be used to analyze time/space complexity and find the 
    stability breaking point (where numerical oscillation leads to divergence/NaN errors)
    as the time-step dt (h) is increased.

    Config Variables to Import:
    --------------------------
    - INITIAL_STATE: np.ndarray of shape (4,), containing default [V_0, M_0, H_0, N_0]
    - T_START: float, start time of simulation (e.g., 0.0 ms)
    - T_END: float, end time of simulation (e.g., 100.0 ms)
    - DT_EVAL: float, default integration time step (e.g., 0.01 ms)
    - I_EXT_DEFAULT: float, default external current injection (e.g., 10.0 uA/cm^2)
    - C_M: float, membrane capacitance (e.g., 1.0 uF/cm^2)
    - G_NA, G_K, G_L: floats, maximal conductances for Na, K, and Leak channels
    - E_NA, E_K, E_L: floats, reversal potentials for Na, K, and Leak channels
    - alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n: functions for gating kinetics

    Input Parameters:
    -----------------
    - y0 : np.ndarray, optional
        Initial state vector [V, m, h, n]. If None, defaults to `INITIAL_STATE`.
        Shape: (4,)
    - t_span : tuple of (float, float), optional
        Simulation start and end time (t_start, t_end). If None, defaults to `(T_START, T_END)`.
    - dt : float, optional
        Fixed integration step size `h`. If None, defaults to `DT_EVAL`.
    - I_ext : float or callable, optional
        Constant external current value or a function of time I_ext(t). 
        If None, defaults to `I_EXT_DEFAULT`.

    Output:
    -------
    - results : np.ndarray
        NumPy array of shape (N, 5) representing the simulation trajectory.
        The columns MUST be strictly ordered as:
        [Time, Voltage, m, h, n]
        - Column 0: Time (ms)
        - Column 1: Membrane Voltage V (mV)
        - Column 2: Sodium activation gating variable m (dimensionless)
        - Column 3: Sodium inactivation gating variable h (dimensionless)
        - Column 4: Potassium activation gating variable n (dimensionless)
    """
    pass
