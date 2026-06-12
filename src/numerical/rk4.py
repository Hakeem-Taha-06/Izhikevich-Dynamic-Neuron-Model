"""Role 5: Explicit RK4 Solver — Izhikevich Neuron Model (2007)

Purpose
-------
Implement a classical fourth-order Runge-Kutta (RK4) integrator for the
Izhikevich (2007) generalized biophysical neuron model.

Uses the step-current protocol I_ext_fn(t) from config.py.

Model Reference
---------------
Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience:
The Geometry of Excitability and Bursting. MIT Press.

Governing equations (2007 generalized biophysical form):
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext
    dw/dt       = a*{ b*(v - v_r) - w }
    if v >= v_peak:  v <- c,  w <- w + d

Required `config.py` imports
----------------------------
- `INITIAL_STATE`
- `T_START`, `T_END`, `DT_EVAL`
- `I_ext_fn`
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
- `a`, `b`, `c`, `d`
- `dv_dt`, `dw_dt`

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, w]`.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    INITIAL_STATE,
    T_START, T_END, DT_EVAL,
    I_ext_fn,
    v_peak, c, d,
    dv_dt, dw_dt,
)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_rk4(y0=None, t_span=None, dt=None):
    """Integrate the Izhikevich model with the classical 4th-order Runge-Kutta method.

    Uses the step-current protocol I_ext_fn(t) from config.py.

    At each time step, the RK4 stages evaluate the ODE at four points
    to achieve O(dt^4) local truncation error. The discrete reset
    condition is checked after the full step and applied before logging.

    Parameters
    ----------
    y0 : array-like of shape (2,), optional
        Initial state [v0 (mV), w0 (pA)]. Defaults to INITIAL_STATE.
    t_span : tuple (t_start, t_end), optional
        Simulation window in ms. Defaults to (T_START, T_END).
    dt : float, optional
        Integration time step in ms. Defaults to DT_EVAL.

    Returns
    -------
    results : numpy.ndarray, shape (N, 3)
        Trajectory matrix ordered as [Time (ms), v (mV), w (pA)].
    """
    # ── 1. Resolve defaults ───────────────────────────────────────────────
    if y0 is None:
        y0 = INITIAL_STATE.copy()
    else:
        y0 = np.asarray(y0, dtype=float)

    if t_span is None:
        t_span = (T_START, T_END)
    t_start, t_end = float(t_span[0]), float(t_span[1])

    if dt is None:
        dt = float(DT_EVAL)
    else:
        dt = float(dt)

    # ── 2. Build uniform time grid ────────────────────────────────────────
    t_values = np.arange(t_start, t_end + dt * 0.5, dt)
    N = len(t_values)

    # ── 3. Pre-allocate output (N, 3) → [Time, v, w] ─────────────────────
    results = np.empty((N, 3), dtype=float)
    results[0, 0] = t_values[0]
    results[0, 1] = y0[0]   # v0
    results[0, 2] = y0[1]   # w0

    v_curr = float(y0[0])
    w_curr = float(y0[1])

    # ── 4. Main RK4 integration loop ─────────────────────────────────────
    for i in range(1, N):
        t_i = t_values[i - 1]

        # Current at each RK4 stage
        I_1 = float(I_ext_fn(t_i))
        I_2 = float(I_ext_fn(t_i + 0.5 * dt))
        I_4 = float(I_ext_fn(t_i + dt))

        # Stage 1
        kv1 = dv_dt(v_curr, w_curr, I_1)
        kw1 = dw_dt(v_curr, w_curr)

        # Stage 2
        v2 = v_curr + 0.5 * dt * kv1
        w2 = w_curr + 0.5 * dt * kw1
        kv2 = dv_dt(v2, w2, I_2)
        kw2 = dw_dt(v2, w2)

        # Stage 3
        v3 = v_curr + 0.5 * dt * kv2
        w3 = w_curr + 0.5 * dt * kw2
        kv3 = dv_dt(v3, w3, I_2)
        kw3 = dw_dt(v3, w3)

        # Stage 4
        v4 = v_curr + dt * kv3
        w4 = w_curr + dt * kw3
        kv4 = dv_dt(v4, w4, I_4)
        kw4 = dw_dt(v4, w4)

        # Update
        v_next = v_curr + (dt / 6.0) * (kv1 + 2*kv2 + 2*kv3 + kv4)
        w_next = w_curr + (dt / 6.0) * (kw1 + 2*kw2 + 2*kw3 + kw4)

        # ── Discrete reset AFTER step, BEFORE logging ─────────────────
        if v_next >= v_peak:
            results[i, 0] = t_values[i]
            results[i, 1] = v_peak
            results[i, 2] = w_next

            v_next = c
            w_next = w_next + d
        else:
            results[i, 0] = t_values[i]
            results[i, 1] = v_next
            results[i, 2] = w_next

        v_curr = v_next
        w_curr = w_next

    return results


# ---------------------------------------------------------------------------
# Self-test & performance benchmark
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Role 5 — RK4 Solver  (self-test)")
    print("Model: Izhikevich (2007) — Regular Spiking")
    print("Step-current protocol: I_ext_fn(t)")
    print("=" * 60)

    N_RUNS = 5
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        res = solve_rk4()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / N_RUNS
    N = res.shape[0]
    spikes = len(np.where(np.diff(res[:, 1]) < -50)[0])

    print(f"\nOutput shape  : {res.shape}   (expected (N, 3))")
    print(f"Time range    : {res[0,0]:.2f} ms  ->  {res[-1,0]:.2f} ms")
    print(f"v  range      : {res[:,1].min():.2f}  to  {res[:,1].max():.2f} mV")
    print(f"w  range      : {res[:,2].min():.2f}  to  {res[:,2].max():.2f} pA")
    print(f"Detected spikes: {spikes}")

    print(f"\nFirst 5 rows  [Time | v | w]:")
    print(res[:5])

    mem_mb = res.nbytes / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Performance Report ({N_RUNS}-run average)")
    print(f"{'='*60}")
    print(f"  Total time steps (N)           : {N:,}")
    print(f"  Total execution time (avg)     : {avg_time:.4f} s")
    print(f"  Average time per step          : {avg_time/N*1000:.4f} ms")
    print(f"  Result array size              : {mem_mb:.4f} MB")
    print(f"{'='*60}")
    print("Self-test complete.")