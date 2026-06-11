import numpy as np
import sys
import os

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import (
    C_m, k, v_r, v_t, v_peak, a, b, c, d,
    INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
    dv_dt, du_dt
)

def solve_adams_bashforth2(y0=None, t_span=None, dt=None, I_ext=None):
    """
    Role 7: Multi-Step Method Developer (Adams-Bashforth 2)
    
    Objective:
    ----------
    Solves the Izhikevich (2007) neuron model using the explicit Adams-Bashforth 2 method.
    Handles the discrete reset condition (v >= v_peak) by flushing history and restarting with Euler.
    
    Output:
    -------
    np.ndarray of shape (N, 3): [Time, v, u]
    """
    # Set defaults
    if y0 is None: y0 = INITIAL_STATE
    if t_span is None: t_span = (T_START, T_END)
    if dt is None: dt = DT_EVAL
    if I_ext is None: I_ext = I_EXT_DEFAULT
    
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + dt, dt)
    num_steps = len(t_values)
    
    # Results array [Time, v, u]
    results = np.zeros((num_steps, 3))
    results[0, 0] = t_values[0]
    results[0, 1:] = y0
    
    # Initial state
    v_curr, u_curr = y0
    
    # History for AB2 (requires two previous points)
    # F(y) = [dv/dt, du/dt]
    F_prev = np.array([dv_dt(v_curr, u_curr, I_ext), du_dt(v_curr, u_curr)])
    
    # Flag to indicate if a reset occurred, forcing an Euler step next
    reset_occurred = False
    
    for i in range(1, num_steps):
        # Apply discrete reset if a spike occurred in the previous step
        if reset_occurred:
            # Perform an Euler step from the reset state
            v_next_euler = v_curr + dt * dv_dt(v_curr, u_curr, I_ext)
            u_next_euler = u_curr + dt * du_dt(v_curr, u_curr)
            
            # Update current state for next iteration
            v_curr, u_curr = v_next_euler, u_next_euler
            
            # Update F_prev for the next AB2 step (this is F(y_n) for the next iteration)
            F_prev = np.array([dv_dt(v_curr, u_curr, I_ext), du_dt(v_curr, u_curr)])
            
            reset_occurred = False # Reset the flag
            
        else:
            # Calculate F_curr = F(y_n)
            F_curr = np.array([dv_dt(v_curr, u_curr, I_ext), du_dt(v_curr, u_curr)])
            
            # Adams-Bashforth 2 step
            # y_{n+1} = y_n + (dt/2) * [3 * F(y_n) - F(y_{n-1})]
            delta_y = (dt / 2.0) * (3 * F_curr - F_prev)
            
            v_next, u_next = v_curr + delta_y[0], u_curr + delta_y[1]
            
            # Update history for next iteration
            F_prev = F_curr
            v_curr, u_curr = v_next, u_next
            
        # Check for spike and apply reset
        if v_curr >= v_peak:
            # Store the spike at v_peak for visualization
            results[i, 0] = t_values[i]
            results[i, 1] = v_peak
            results[i, 2] = u_curr # u continues to evolve normally until reset
            
            # Apply the mathematical reset for the NEXT step
            v_curr = c
            u_curr = u_curr + d
            reset_occurred = True # Set flag to force Euler step next
        else:
            # Store normal step results
            results[i, 0] = t_values[i]
            results[i, 1] = v_curr
            results[i, 2] = u_curr
            
    return results

if __name__ == "__main__":
    print("Simulating Izhikevich Model with Adams-Bashforth 2...")
    res = solve_adams_bashforth2()
    print(f"Simulation complete. Shape: {res.shape}")
    print("First 5 steps:\n", res[:5])
    
    # Check for spikes
    spikes = res[res[:, 1] >= v_peak]
    print(f"Number of spikes detected: {len(spikes)}")
