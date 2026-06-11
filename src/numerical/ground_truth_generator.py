"""Role 4: Ground Truth & Data Engineer — Izhikevich Neuron Model

Purpose
-------
Generate a mathematically flawless, high-fidelity dataset of ~4,000 unique
Izhikevich neuron simulations for use by the Machine Learning phase.

Architecture: Segmented Integration
------------------------------------
Continuous solvers cannot handle the discrete spike reset teleportation.
This module uses **Segmented Integration**:
  1. Run a stiff solver (Radau) with an event function detecting v == v_peak.
  2. When the event fires, halt the solver, apply the discrete reset
     (v <- c, u <- u + d), log the jump, and restart from that exact timestamp.
  3. Repeat until T_END is reached.

Solver constraints (per ROLES.md):
  - Method: Radau (stiff)
  - rtol = 1e-6, atol = 1e-9

Output Schema (per ROLES.md)
-----------------------------
CSV saved to /data/ground_truth.csv with columns:
    Sim_ID | Time (ms) | I_ext (pA) | v (mV) | u (pA)

Time is strictly interpolated to 0.01 ms resolution.
Sim_ID increments by 1 for each new (I_ext, V0, U0) combination.

Iteration Grid (~4,000 simulations)
-------------------------------------
  I_ext : 0.0 to 500.0 pA, step 12.5  -> 41 values
  V0    : -85.0 to -45.0 mV, step 2.0  -> 21 values  (but we cap at 40*5=4000)
  U0    : 5 variations around steady-state b*(V0 - v_r)

Target: exactly 4,000 distinct simulation runs.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    C_m, k, v_r, v_t, v_peak,
    a, b, c, d,
    T_START, T_END, DT_EVAL,
)

# ---------------------------------------------------------------------------
# ODE right-hand sides (local copies — self-contained module)
# ---------------------------------------------------------------------------

def _ode(t, y, I_ext):
    """Izhikevich (2007) ODE system: returns [dv/dt, du/dt]."""
    v, u = y
    dv = (k * (v - v_r) * (v - v_t) - u + I_ext) / C_m
    du = a * (b * (v - v_r) - u)
    return [dv, du]


def _spike_event(t, y, I_ext):
    """Event function: triggers when v reaches v_peak."""
    return y[0] - v_peak

_spike_event.terminal  = True   # halt solver at event
_spike_event.direction = 1      # only rising crossings


# ---------------------------------------------------------------------------
# Single-simulation segmented integrator
# ---------------------------------------------------------------------------

def _simulate_one(V0, U0, I_ext, t_start=T_START, t_end=T_END, dt_out=DT_EVAL):
    """Run one Izhikevich simulation using segmented Radau integration.

    Returns
    -------
    t_dense : np.ndarray, shape (M,)
        Time points at 0.01 ms resolution from t_start to t_end.
    v_dense : np.ndarray, shape (M,)
    u_dense : np.ndarray, shape (M,)
    """
    # Output time grid (fixed resolution)
    t_out = np.arange(t_start, t_end + dt_out * 0.5, dt_out)

    # Storage for interpolated output
    v_out = np.empty_like(t_out)
    u_out = np.empty_like(t_out)

    # Track which output indices have been filled
    out_ptr = 0

    y_curr = [float(V0), float(U0)]
    t_curr = float(t_start)

    while t_curr < t_end - 1e-12:
        # Remaining output points in [t_curr, t_end]
        mask = (t_out >= t_curr - 1e-12) & (t_out <= t_end + 1e-12)
        t_eval_seg = t_out[mask]

        sol = solve_ivp(
            fun=_ode,
            t_span=(t_curr, t_end),
            y0=y_curr,
            method='Radau',
            t_eval=t_eval_seg if len(t_eval_seg) > 0 else None,
            events=_spike_event,
            rtol=1e-6,
            atol=1e-9,
            args=(I_ext,),
            dense_output=True,
        )

        # Fill in the dense output for this segment
        if sol.t_events[0].size > 0:
            # Event fired — fill up to (not including) the spike time
            t_spike = sol.t_events[0][0]
            seg_mask = mask & (t_out < t_spike - 1e-12)
        else:
            seg_mask = mask

        # Evaluate via dense solution for precision
        t_seg_pts = t_out[seg_mask]
        if len(t_seg_pts) > 0:
            y_seg = sol.sol(t_seg_pts)
            # Find where these go in the output arrays
            idx = np.where(seg_mask)[0]
            v_out[idx] = y_seg[0]
            u_out[idx] = y_seg[1]

        if sol.t_events[0].size > 0:
            # Spike occurred: log the spike point at v_peak
            t_spike = sol.t_events[0][0]
            v_at_spike = v_peak
            u_at_spike = sol.y_events[0][0][1]  # u just before reset

            # Find output index closest to spike time (if any)
            spike_idx_candidates = np.where(
                mask & (np.abs(t_out - t_spike) < dt_out * 0.6)
            )[0]
            if len(spike_idx_candidates) > 0:
                v_out[spike_idx_candidates[0]] = v_at_spike
                u_out[spike_idx_candidates[0]] = u_at_spike

            # Apply discrete reset
            v_reset = c
            u_reset = u_at_spike + d

            y_curr = [v_reset, u_reset]
            t_curr = t_spike  # restart from exact spike time
        else:
            # No more spikes — simulation done
            break

    # Fill any remaining unfilled points (after last spike or no spikes)
    unfilled = np.where(np.isnan(v_out) | (v_out == 0.0))[0]
    # More robust: just evaluate the last sol.sol over remaining range
    remaining_mask = t_out >= t_curr - 1e-12
    remaining_idx  = np.where(remaining_mask)[0]
    if len(remaining_idx) > 0 and sol.success:
        y_rem = sol.sol(t_out[remaining_idx])
        # Only fill points that haven't been set by a segment
        for ii, global_i in enumerate(remaining_idx):
            if v_out[global_i] == 0.0 and t_out[global_i] > dt_out * 0.5:
                v_out[global_i] = y_rem[0, ii]
                u_out[global_i] = y_rem[1, ii]

    # The first point is always the initial condition
    v_out[0] = float(V0)
    u_out[0] = float(U0)

    return t_out, v_out, u_out


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_ground_truth(output_path=None):
    """Generate the full ~4,000-simulation dataset and save to CSV.

    Parameters
    ----------
    output_path : str or Path, optional
        Where to save the CSV.  Defaults to ``<project_root>/data/ground_truth.csv``.

    Returns
    -------
    df : pd.DataFrame
        The complete dataset.
    """
    if output_path is None:
        data_dir = Path(_ROOT) / 'data'
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / 'ground_truth.csv'

    # ------------------------------------------------------------------
    # Build iteration grid (target: 4,000 simulations = 40 x 20 x 5)
    # ------------------------------------------------------------------
    I_ext_values = np.arange(0.0, 500.0 + 1e-9, 12.5)          # 41 values
    V0_values    = np.arange(-85.0, -45.0 + 1e-9, 2.0)          # 21 values

    # 40 * 20 * 5 = 4000  =>  take first 40 I_ext and first 20 V0
    I_ext_values = I_ext_values[:40]   # 40 values
    V0_values    = V0_values[:20]      # 20 values

    # 5 U0 offsets around steady-state
    U0_offsets = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])

    total_sims = len(I_ext_values) * len(V0_values) * len(U0_offsets)
    print(f"Grid: {len(I_ext_values)} I_ext × {len(V0_values)} V0 × {len(U0_offsets)} U0 = {total_sims} simulations")

    all_records = []
    sim_id = 0

    for I_ext in I_ext_values:
        for V0 in V0_values:
            u_ss = b * (V0 - v_r)          # steady-state u
            for u_offset in U0_offsets:
                sim_id += 1
                U0 = u_ss + u_offset

                t_arr, v_arr, u_arr = _simulate_one(V0, U0, I_ext)

                n_pts = len(t_arr)
                sim_ids = np.full(n_pts, sim_id, dtype=int)
                i_exts  = np.full(n_pts, I_ext)

                chunk = np.column_stack([sim_ids, t_arr, i_exts, v_arr, u_arr])
                all_records.append(chunk)

                if sim_id % 100 == 0:
                    print(f"  [{sim_id}/{total_sims}] I_ext={I_ext:.1f} V0={V0:.1f} U0={U0:.2f}")

    print(f"\nAll {sim_id} simulations complete. Assembling DataFrame...")

    data = np.vstack(all_records)
    df = pd.DataFrame(
        data,
        columns=['Sim_ID', 'Time (ms)', 'I_ext (pA)', 'v (mV)', 'u (pA)']
    )
    df['Sim_ID'] = df['Sim_ID'].astype(int)

    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Total rows: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Quick self-test (single simulation)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Role 4 — Ground Truth Generator  (self-test: 1 simulation)")
    print("=" * 60)

    V0_test  = -60.0
    U0_test  = b * (V0_test - v_r)
    I_test   = 300.0

    t, v, u = _simulate_one(V0_test, U0_test, I_test)

    print(f"Output length : {len(t)} points")
    print(f"Time range    : {t[0]:.2f} ms  ->  {t[-1]:.2f} ms")
    print(f"v range       : {v.min():.2f}  to  {v.max():.2f} mV")
    print(f"u range       : {u.min():.2f}  to  {u.max():.2f} pA")
    print()
    print("First 5 rows  [Time | v | u]:")
    for i in range(5):
        print(f"  {t[i]:.4f}  {v[i]:.4f}  {u[i]:.4f}")

    spike_idx = np.where(np.diff(v) < -50)[0]
    print(f"\nDetected spikes: {len(spike_idx)}")
    print("\nSelf-test complete.")
    print("\nTo generate the full dataset, call generate_ground_truth()")