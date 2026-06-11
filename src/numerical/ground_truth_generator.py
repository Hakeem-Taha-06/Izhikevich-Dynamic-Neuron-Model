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
import pandas as pd
import sys
from pathlib import Path

# =====================================================
# Add project root to Python path
# =====================================================
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import *

# =====================================================
# ODE system
# =====================================================
def neuron_ode(t, y, I_ext):
    v, u = y

    dv = (k * (v - v_r) * (v - v_t) - u + I_ext) / C_m
    du = a * (b * (v - v_r) - u)

    return [dv, du]


def spike_event(t, y, I_ext):
    return y[0] - v_peak


spike_event.terminal = True
spike_event.direction = 1


def run_single_simulation(I_ext, V0, U0):

    all_times = []
    all_v = []
    all_u = []

    t_current = T_START
    state = [V0, U0]

    while t_current < T_END:

        sol = solve_ivp(
            lambda t, y: neuron_ode(t, y, I_ext),
            (t_current, T_END),
            state,
            method="Radau",
            events=lambda t, y: spike_event(t, y, I_ext),
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )

        t_segment = np.arange(
            t_current,
            sol.t[-1] + DT_EVAL,
            DT_EVAL
        )

        y_segment = sol.sol(t_segment)

        all_times.extend(t_segment.tolist())
        all_v.extend(y_segment[0].tolist())
        all_u.extend(y_segment[1].tolist())

        if len(sol.t_events[0]) == 0:
            break

        spike_time = sol.t_events[0][0]

        all_times.append(spike_time)
        all_v.append(v_peak)
        all_u.append(sol.y_events[0][0][1])

        v_reset = c
        u_reset = sol.y_events[0][0][1] + d

        state = [v_reset, u_reset]
        t_current = spike_time

    return pd.DataFrame({
        "Time (ms)": all_times,
        "v (mV)": all_v,
        "u (pA)": all_u
    })


def make_u0_variations(v0):

    u_ss = b * (v0 - v_r)

    return [
        u_ss - 40,
        u_ss - 20,
        u_ss,
        u_ss + 20,
        u_ss + 40
    ]






# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    I_values = np.arange(0.0, 500.0, 12.5)
    V_values = np.arange(-85.0, -45.0, 2.0)

    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "ground_truth.csv"

    if output_file.exists():
        output_file.unlink()

    header_written = False
    sim_id = 1

    for I_ext in I_values:

        print(f"Current = {I_ext}")

        for V0 in V_values:

            for U0 in make_u0_variations(V0):

                df = run_single_simulation(
                    I_ext=I_ext,
                    V0=V0,
                    U0=U0
                )

                df.insert(0, "Sim_ID", sim_id)
                df.insert(2, "I_ext (pA)", I_ext)

                df = df[
                    [
                        "Sim_ID",
                        "Time (ms)",
                        "I_ext (pA)",
                        "v (mV)",
                        "u (pA)"
                    ]
                ]

                df.to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=not header_written
                )

                header_written = True
                sim_id += 1

    print("Finished")
    print("Total simulations =", sim_id - 1)