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
            min(sol.t[-1], T_END),
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

        t_current = spike_time + DT_EVAL


    all_times = np.array(all_times)
    all_v = np.array(all_v)
    all_w = np.array(all_w)

    time_uniform = np.arange(
        T_START,
        T_END + DT_EVAL,
        DT_EVAL
    )

    v_uniform = np.interp(
        time_uniform,
        all_times,
        all_v
    )

    w_uniform = np.interp(
        time_uniform,
        all_times,
        all_w
    )

    I_uniform = I_ext_fn(time_uniform)

    return pd.DataFrame({
        "Time (ms)": time_uniform,
        "I_ext (pA)": I_uniform,
        "v (mV)": v_uniform,
        "w (pA)": w_uniform
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

    V0, W0 = INITIAL_STATE

    print(
        f"Running step-current simulation: "
        f"I=0 for t<{T_STIM_ONSET}ms, "
        f"I={I_EXT_DEFAULT} for t>={T_STIM_ONSET}ms"
    )

    print(
        f"V0={V0}, W0={W0}, T_END={T_END}"
    )
    print("DT_EVAL =", DT_EVAL)
    df = run_single_simulation(
        V0=V0,
        W0=W0
    )

    df.insert(
        0,
        "Sim_ID",
        1
    )

    df = df[
        [
            "Sim_ID",
            "Time (ms)",
            "I_ext (pA)",
            "v (mV)",
            "w (pA)"
        ]
    ]

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Finished. Shape: {df.shape}"
    )

    print(
        f"Saved to: {output_file}"
    )


    plt.figure(
        figsize=(12,5)
    )

    plt.plot(
        df["Time (ms)"],
        df["v (mV)"],
        linewidth=1
    )

    plt.xlabel(
        "Time (ms)"
    )

    plt.ylabel(
        "Membrane Potential (mV)"
    )

    plt.title(
        "Izhikevich RS Neuron using Radau"
    )

    plt.grid(True)

    plt.show()