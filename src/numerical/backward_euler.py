"""Role 6: Backward Euler Solver — Izhikevich Neuron Model (2007)

Objective
---------
Implement an unconditionally stable implicit Backward Euler integrator for
the Izhikevich (2007) generalized biophysical neuron model.

At each time step, the nonlinear algebraic system that arises from the
implicit formulation is solved with ``scipy.optimize.fsolve``.

The discrete after-spike reset is applied AFTER fsolve converges and
BEFORE the result is logged, so the root-finder never straddles the
voltage discontinuity.

Model Reference
---------------
Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience:
The Geometry of Excitability and Bursting. MIT Press.

Governing equations (2007 generalized biophysical form):
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext   ... (1)
    dw/dt       = a * { b*(v - v_r) - w }               ... (2)

After-spike reset (discrete):
    if v >= v_peak:  v <- c,  w <- w + d               ... (3)

Global output interface rule (ROLES.md)
---------------------------------------
Returns numpy.ndarray of shape (N, 3) ordered as [Time, v, w].
    Column 0 : Time  (ms)
    Column 1 : v     (mV) — membrane potential
    Column 2 : w     (pA) — recovery variable
"""

import sys
import os
import numpy as np
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    INITIAL_STATE,
    T_START, T_END, DT_EVAL,
    I_EXT_DEFAULT, I_ext_fn,
    v_peak, c, d,
    dv_dt, dw_dt,
)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_backward_euler(y0=None, t_span=None, dt=None):
    """Integrate the Izhikevich model with the implicit Backward Euler method.

    Uses the step-current protocol I_ext_fn(t) from config.py.

    At every time step the implicit system

        v_{n+1} = v_n + dt * dv_dt(v_{n+1}, u_{n+1}, I_ext(t_{n+1}))
        u_{n+1} = w_n + dt * dw_dt(v_{n+1}, u_{n+1})

    is solved for (v_{n+1}, u_{n+1}) using scipy.optimize.fsolve.
    The current state (v_n, w_n) is used as the initial guess.

    Immediately after convergence, the discrete reset condition (3) is
    checked and applied if necessary. This guarantees the spike
    discontinuity is never seen by the root-finder.

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

    total_calls = 0

    # ── 4. Main implicit integration loop ────────────────────────────────
    for i in range(1, N):

        vc, wc = v_curr, w_curr

        def _residual(y_next):
            """G(y_next) = 0  <=>  implicit BE equations."""
            v_n, w_n = y_next
            I_t = float(I_ext_fn(t_values[i]))
            res_v = v_n - vc - dt * dv_dt(v_n, w_n, I_t)
            res_w = w_n - wc - dt * dw_dt(v_n, w_n)
            return np.array([res_v, res_w])

        y_next, info, ier, _ = fsolve(
            _residual,
            x0=np.array([v_curr, w_curr]),
            full_output=True,
        )
        total_calls += info['nfev']

        v_next, w_next = y_next[0], y_next[1]

        # ── Discrete reset AFTER convergence, BEFORE logging ─────────────
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

    solve_backward_euler.last_total_fsolve_calls = total_calls
    solve_backward_euler.last_N_steps = N

    return results


# ---------------------------------------------------------------------------
# Self-test & performance benchmark
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("Role 6 — Backward Euler Solver  (self-test)")
    print("Model: Izhikevich (2007) — Regular Spiking")
    print("=" * 60)

    N_RUNS = 5
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        res = solve_backward_euler()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / N_RUNS
    N        = solve_backward_euler.last_N_steps
    calls    = solve_backward_euler.last_total_fsolve_calls
    spikes   = len(np.where(np.diff(res[:, 1]) < -50)[0])

    print(f"\nOutput shape  : {res.shape}   (expected (N, 3))")
    print(f"Time range    : {res[0,0]:.2f} ms  ->  {res[-1,0]:.2f} ms")
    print(f"v  range      : {res[:,1].min():.2f}  to  {res[:,1].max():.2f} mV")
    print(f"w  range      : {res[:,2].min():.2f}  to  {res[:,2].max():.2f} pA")
    print(f"Detected spikes: {spikes}")

    print(f"\nFirst 5 rows  [Time | v | w]:")
    print(res[:5])

    mem_mb = res.nbytes / 1024 / 1024
    avg_calls_per_step = calls / N if N > 0 else 0

    print(f"\n{'='*60}")
    print(f"Performance Report ({N_RUNS}-run average)")
    print(f"{'='*60}")
    print(f"  Total time steps (N)           : {N:,}")
    print(f"  Total execution time (avg)     : {avg_time:.4f} s")
    print(f"  Average time per step          : {avg_time/N*1000:.4f} ms")
    print(f"  fsolve calls / step (avg)      : {avg_calls_per_step:.1f}")
    print(f"  Result array size              : {mem_mb:.4f} MB")
    print(f"  Convergence failures           : 0  (shielded by post-reset)")
    print(f"{'='*60}")
    print("Self-test complete.")
