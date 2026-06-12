"""Role 4: Ground Truth Generator

Purpose
-------
Define the data-generation workspace for the Izhikevich spiking neuron project.
This module is reserved for the baseline dataset pipeline and its narrative contract.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext
    dw/dt       = a*{ b*(v - v_r) - w }
    if v >= v_peak:  v <- c,  w <- w + d

Required `config.py` imports
----------------------------
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`, `I_ext_fn`
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
- `a`, `b`, `c`, `d`

Must achieve
------------
- Produce the baseline trajectory dataset used by the downstream numerical and ML stages.
- Save the generated dataset under `data/`.
- Preserve the project-wide output schema exactly.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, w]`.

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
import matplotlib.pyplot as plt

# =====================================================
# Add project root to Python path
# =====================================================
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import *

# =====================================================
# ODE system
# =====================================================
def neuron_ode(t, y):
    """ODE right-hand side using the step-current I_ext_fn(t)."""
    v, w = y
    I = float(I_ext_fn(t))

    dv = (k * (v - v_r) * (v - v_t) - w + I) / C_m
    dw = a * (b * (v - v_r) - w)

    return [dv, dw]


def spike_event(t, y):
    return y[0] - v_peak


spike_event.terminal = True
spike_event.direction = 1


def run_single_simulation(V0, W0):
    """Run one simulation using the step-current protocol from config."""
    all_times = []
    all_v = []
    all_w = []

    t_current = T_START
    state = [V0, W0]

    while t_current < T_END:

        sol = solve_ivp(
            neuron_ode,
            (t_current, T_END),
            state,
            method="Radau",
            events=spike_event,
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
        all_w.extend(y_segment[1].tolist())

        if len(sol.t_events[0]) == 0:
            break

        spike_time = sol.t_events[0][0]

        all_times.append(spike_time)
        all_v.append(v_peak)
        all_w.append(sol.y_events[0][0][1])

        v_reset = c
        w_reset = sol.y_events[0][0][1] + d

        state = [v_reset, w_reset]
        t_current = spike_time

    # Compute I_ext for each time point for the CSV
    t_arr = np.array(all_times)
    I_arr = I_ext_fn(t_arr)

    return pd.DataFrame({
        "Time (ms)": all_times,
        "I_ext (pA)": I_arr,
        "v (mV)": all_v,
        "w (pA)": all_w
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

    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "ground_truth.csv"

    if output_file.exists():
        output_file.unlink()

    # Single simulation with default initial conditions and
    # step-current protocol (0 pA -> I_EXT_DEFAULT at T_STIM_ONSET)
    V0, W0 = INITIAL_STATE
    print(f"Running step-current simulation: I=0 for t<{T_STIM_ONSET}ms, "
          f"I={I_EXT_DEFAULT} for t>={T_STIM_ONSET}ms")
    print(f"V0={V0}, W0={W0}, T_END={T_END}")

    df = run_single_simulation(V0=V0, W0=W0)
    df.insert(0, "Sim_ID", 1)

    df = df[
        [
            "Sim_ID",
            "Time (ms)",
            "I_ext (pA)",
            "v (mV)",
            "w (pA)"
        ]
    ]

    df.to_csv(output_file, index=False)

    print(f"Finished. Shape: {df.shape}")
    print(f"Saved to: {output_file}")
    

    plt.figure(figsize=(12,5))
    plt.plot(df["Time (ms)"], df["v (mV)"])
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.grid()
    plt.show()