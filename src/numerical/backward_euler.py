import numpy as np
from scipy.optimize import fsolve
import sys
import os

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import (
    C_m, G_NA, G_K, G_L, E_NA, E_K, E_L,
    INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
)

def hodgkin_huxley_derivatives(y, I_ext):
    """
    Calculates the derivatives for the Hodgkin-Huxley model.
    y = [V, m, h, n]
    """
    V, m, h, n = y
    
    # Membrane potential derivative: dV/dt
    # I_ion = gNa*m^3*h*(V-ENa) + gK*n^4*(V-EK) + gL*(V-EL)
    I_na = G_NA * (m**3) * h * (V - E_NA)
    I_k = G_K * (n**4) * (V - E_K)
    I_l = G_L * (V - E_L)
    
    dVdt = (I_ext - (I_na + I_k + I_l)) / C_m
    
    # Gating variables derivatives: dx/dt = alpha_x(V)*(1-x) - beta_x(V)*x
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    
    return np.array([dVdt, dmdt, dhdt, dndt])

def solve_implicit_backward_euler(y0=None, t_span=None, dt=None, I_ext=None):
    """
    Role 5: Implicit Solver Developer (Backward Euler)
    
    Objective:
    ----------
    Solves the Hodgkin-Huxley system of ODEs using the implicit Backward Euler method.
    An internal root-finding loop is implemented at each step using `scipy.optimize.fsolve`
    to solve the implicit algebraic system of equations.
    
    The solver is expected to be unconditionally stable and show robustness at larger 
    time steps where explicit solvers (like RK4) fail.
    
    Config Variables to Import:
    --------------------------
    - INITIAL_STATE: np.ndarray of shape (4,), containing default [V_0, M_0, H_0, N_0]
    - T_START: float, start time of simulation (e.g., 0.0 ms)
    - T_END: float, end time of simulation (e.g., 100.0 ms)
    - DT_EVAL: float, default integration time step (e.g., 0.01 ms)
    - I_EXT_DEFAULT: float, default external current injection (e.g., 10.0 uA/cm^2)
    - C_m: float, membrane capacitance (e.g., 1.0 uF/cm^2)
    - G_NA, G_K, G_L: floats, maximal conductances for Na, K, and Leak channels
    - E_NA, E_K, E_L: floats, reversal potentials for Na, K, and Leak channels
    - alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n: functions for gating kinetics
    
    Input Parameters:
    -----------------
    - y0 : np.ndarray, optional
        Initial state vector [V, m, h, n]. If None, defaults to `INITIAL_STATE`.
    - t_span : tuple of (float, float), optional
        Simulation start and end time (t_start, t_end). If None, defaults to `(T_START, T_END)`.
    - dt : float, optional
        Integration time step. If None, defaults to `DT_EVAL`.
    - I_ext : float, optional
        Constant external current injection. If None, defaults to `I_EXT_DEFAULT`.
        
    Output:
    -------
    np.ndarray of shape (N, 5): [Time, Voltage, m, h, n]
    """
    # Set defaults if not provided
    if y0 is None:
        y0 = INITIAL_STATE
    if t_span is None:
        t_span = (T_START, T_END)
    if dt is None:
        dt = DT_EVAL
    if I_ext is None:
        I_ext = I_EXT_DEFAULT
        
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + dt, dt)
    num_steps = len(t_values)
    
    # Initialize results array (N, 5) -> [Time, V, m, h, n]
    results = np.zeros((num_steps, 5))
    results[0, 0] = t_values[0]
    results[0, 1:] = y0
    
    current_y = np.array(y0)
    
    for i in range(1, num_steps):
        # Backward Euler: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
        # Define the implicit function to solve: G(y_next) = y_next - y_curr - dt * f(y_next) = 0
        def implicit_func(y_next):
            return y_next - current_y - dt * hodgkin_huxley_derivatives(y_next, I_ext)
        
        # Use fsolve to find y_next. Start with current_y as initial guess.
        y_next = fsolve(implicit_func, current_y)
        
        # Update current state
        current_y = y_next
        
        # Store results
        results[i, 0] = t_values[i]
        results[i, 1:] = current_y
        
    return results

if __name__ == "__main__":
    # Test the solver
    print("Starting Backward Euler simulation...")
    res = solve_implicit_backward_euler()
    print(f"Simulation complete. Results shape: {res.shape}")
    print("First 5 steps:")
    print(res[:5])
