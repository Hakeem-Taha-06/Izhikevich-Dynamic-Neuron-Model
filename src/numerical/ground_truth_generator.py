"""Role 4: Ground Truth Generator

Purpose
-------
Define the data-generation workspace for the Izhikevich spiking neuron project.
This module is reserved for the baseline dataset pipeline and its narrative contract.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a*{ b*(v - v_r) - u }
    if v >= v_peak:  v <- c,  u <- u + d

Required `config.py` imports
----------------------------
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
- `a`, `b`, `c`, `d`

Must achieve
------------
- Produce the baseline trajectory dataset used by the downstream numerical and ML stages.
- Save the generated dataset under `data/`.
- Preserve the project-wide output schema exactly.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No loops.
- No equation-solving logic.
- This file is documentation-only in the scaffold."""





from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# =====================================================
# Add project root to Python path
# =====================================================
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import (
    INITIAL_STATE,
    T_START,
    T_END,
    DT_EVAL,
    I_EXT_DEFAULT,
    dv_dt,
    du_dt
)


# =====================================================
# ODE system
# =====================================================
def izhikevich_system(t, y):
    """
    State vector

    y[0] = membrane potential v
    y[1] = recovery variable u
    """

    v, u = y

    dv = dv_dt(v, u, I_EXT_DEFAULT)
    du = du_dt(v, u)

    return [dv, du]


# =====================================================
# Generate Ground Truth
# =====================================================
def generate_ground_truth():

    # Time vector
    time = np.arange(
        T_START,
        T_END + DT_EVAL,
        DT_EVAL
    )

    N = len(time)

    # Arrays
    v = np.zeros(N)
    u = np.zeros(N)

    # Initial conditions
    v[0] = INITIAL_STATE[0]
    u[0] = INITIAL_STATE[1]

    # Main loop
    for i in range(N - 1):

        # One RK45 step
        sol = solve_ivp(
            fun=izhikevich_system,
            t_span=(time[i], time[i+1]),
            y0=[v[i], u[i]],
            method="RK45",
            t_eval=[time[i+1]]
        )

        # Next values
        v_next = sol.y[0, -1]
        u_next = sol.y[1, -1]

        # Spike reset
        if v_next >= 35.0:
            v[i] = 35.0      # spike peak for plotting
            v_next = -50.0   # reset voltage
            u_next += 100.0  # recovery jump

        # Store
        v[i+1] = v_next
        u[i+1] = u_next

    # Final dataset
    ground_truth = np.column_stack(
        (
            time,
            v,
            u
        )
    )

    return ground_truth  



# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    ground_truth = generate_ground_truth()

    print("Ground truth shape:")
    print(ground_truth.shape)

    print("\nFirst 10 rows:")
    print(ground_truth[:10])

    # Save dataset
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)

    np.savetxt(
    data_dir / "ground_truth.csv",
    ground_truth,
    delimiter=",",
    header="time,v,u",
    comments=""
)

    print("\nDataset saved successfully.")

    # Plot v(t)
    plt.figure(figsize=(10,5))

    plt.plot(
        ground_truth[:,0],
        ground_truth[:,1]
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Izhikevich Ground Truth using RK45")

    plt.grid()
    plt.show()