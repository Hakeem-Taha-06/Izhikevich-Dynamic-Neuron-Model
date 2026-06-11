"""Role 5: Explicit RK4 Solver — Izhikevich Neuron Model

Purpose
-------
Implements the classic 4th-Order Runge-Kutta (RK4) explicit integrator for
the Izhikevich (2007) generalized biophysical neuron model.

Model Reference
---------------
Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience:
The Geometry of Excitability and Bursting*. MIT Press.

Governing equations (2007 generalized biophysical form):

    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext   ... (1)
    du/dt       = a * { b*(v - v_r) - u }               ... (2)

After-spike reset (discrete):

    if v >= v_peak:  v <- c,  u <- u + d               ... (3)

Strict output interface rule
-----------------------------
Returns ``numpy.ndarray`` of shape ``(N, 3)`` ordered as ``[Time, v, u]``.
    - Column 0 : Time  (ms)
    - Column 1 : v     (mV)  — membrane potential
    - Column 2 : u     (pA)  — recovery variable
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
    I_EXT_DEFAULT,
    C_m, k, v_r, v_t, v_peak,
    a, b, c, d,
)


# ---------------------------------------------------------------------------
# Internal ODE right-hand sides
# ---------------------------------------------------------------------------

def _f_v(v_val, u_val, I_ext):
    """dv/dt — equation (1)."""
    return (k * (v_val - v_r) * (v_val - v_t) - u_val + I_ext) / C_m


def _f_u(v_val, u_val):
    """du/dt — equation (2)."""
    return a * (b * (v_val - v_r) - u_val)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_rk4(
    y0=None,
    t_span=None,
    dt=None,
    I_ext=None,
):
    """Integrate the Izhikevich model with the explicit 4th-Order Runge-Kutta method.

    Each step computes four derivative evaluations (k1–k4) and combines
    them as the weighted average:

        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    The discrete reset condition (3) is checked and applied **after** every
    completed RK4 step so that the four-stage computation never straddles
    the spike discontinuity.

    Parameters
    ----------
    y0 : array-like of shape (2,), optional
        Initial state [v0 (mV), u0 (pA)].  Defaults to ``INITIAL_STATE``.
    t_span : tuple (t_start, t_end), optional
        Simulation window in ms.  Defaults to ``(T_START, T_END)``.
    dt : float, optional
        Integration time step in ms.  Defaults to ``DT_EVAL``.
        NOTE: RK4 is conditionally stable. If ``dt`` is too large the
        solution will diverge to NaN.  See rk4_notes.md for the stability
        limit found experimentally.
    I_ext : float, optional
        Constant external current in pA.  Defaults to ``I_EXT_DEFAULT``.

    Returns
    -------
    results : numpy.ndarray, shape (N, 3)
        Trajectory matrix ordered as **[Time (ms), v (mV), u (pA)]**.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve defaults                                                  #
    # ------------------------------------------------------------------ #
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

    if I_ext is None:
        I_ext = float(I_EXT_DEFAULT)
    else:
        I_ext = float(I_ext)

    # ------------------------------------------------------------------ #
    # 2. Build time grid                                                   #
    # ------------------------------------------------------------------ #
    t_values = np.arange(t_start, t_end + dt * 0.5, dt)
    N = len(t_values)

    # ------------------------------------------------------------------ #
    # 3. Pre-allocate output array  (N, 3)  ->  [Time, v, u]             #
    # ------------------------------------------------------------------ #
    results = np.empty((N, 3), dtype=float)
    results[0, 0] = t_values[0]
    results[0, 1] = y0[0]   # v0
    results[0, 2] = y0[1]   # u0

    v_curr = float(y0[0])
    u_curr = float(y0[1])

    # ------------------------------------------------------------------ #
    # 4. Main RK4 integration loop                                         #
    # ------------------------------------------------------------------ #
    for i in range(1, N):

        # -- Stage 1: derivatives at current state -----------------------
        k1_v = _f_v(v_curr, u_curr, I_ext)
        k1_u = _f_u(v_curr, u_curr)

        # -- Stage 2: derivatives at midpoint using k1 -------------------
        v_mid2 = v_curr + 0.5 * dt * k1_v
        u_mid2 = u_curr + 0.5 * dt * k1_u
        k2_v = _f_v(v_mid2, u_mid2, I_ext)
        k2_u = _f_u(v_mid2, u_mid2)

        # -- Stage 3: derivatives at midpoint using k2 -------------------
        v_mid3 = v_curr + 0.5 * dt * k2_v
        u_mid3 = u_curr + 0.5 * dt * k2_u
        k3_v = _f_v(v_mid3, u_mid3, I_ext)
        k3_u = _f_u(v_mid3, u_mid3)

        # -- Stage 4: derivatives at endpoint using k3 -------------------
        v_end = v_curr + dt * k3_v
        u_end = u_curr + dt * k3_u
        k4_v = _f_v(v_end, u_end, I_ext)
        k4_u = _f_u(v_end, u_end)

        # -- RK4 weighted average ----------------------------------------
        v_next = v_curr + (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        u_next = u_curr + (dt / 6.0) * (k1_u + 2.0*k2_u + 2.0*k3_u + k4_u)

        # -- Discrete reset AFTER the full RK4 step ----------------------
        # The four stages never cross the discontinuity; the reset is
        # applied only to the final committed state.
        if v_next >= v_peak:
            v_next = c           # reset membrane potential  (mV)
            u_next = u_next + d  # increment recovery variable (pA)

        # -- Log and advance state ---------------------------------------
        results[i, 0] = t_values[i]
        results[i, 1] = v_next
        results[i, 2] = u_next

        v_curr = v_next
        u_curr = u_next

    return results


# ---------------------------------------------------------------------------
# Stability limit finder (required by ROLES.md)
# ---------------------------------------------------------------------------

def find_stability_limit(I_ext=None, dt_start=0.5, dt_step=0.5, dt_max=20.0):
    """Scan increasing time steps to find where RK4 first produces NaN.

    Returns
    -------
    float or None
        The smallest ``dt`` (ms) that causes NaN output, or None if the
        method remains stable up to ``dt_max``.
    """
    if I_ext is None:
        I_ext = float(I_EXT_DEFAULT)

    dt = dt_start
    while dt <= dt_max:
        res = solve_rk4(dt=dt, I_ext=I_ext)
        if np.any(np.isnan(res)):
            return dt
        dt = round(dt + dt_step, 6)
    return None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Role 5 — RK4 Solver  (self-test)")
    print("=" * 60)

    res = solve_rk4()

    print(f"Output shape  : {res.shape}   (expected (N, 3))")
    print(f"Time range    : {res[0,0]:.2f} ms  ->  {res[-1,0]:.2f} ms")
    print(f"v  range      : {res[:,1].min():.2f}  to  {res[:,1].max():.2f} mV")
    print(f"u  range      : {res[:,2].min():.2f}  to  {res[:,2].max():.2f} pA")
    print()
    print("First 5 rows  [Time | v | u]:")
    print(res[:5])
    print()
    print("Last  5 rows  [Time | v | u]:")
    print(res[-5:])

    spike_indices = np.where(np.diff(res[:, 1]) < -50)[0]
    print(f"\nDetected spikes: {len(spike_indices)}")

    print("\n--- Stability Limit Search ---")
    limit = find_stability_limit()
    if limit is not None:
        print(f"RK4 first produces NaN at dt = {limit} ms")
    else:
        print("RK4 remained stable up to dt_max — increase search range.")
    print("Self-test complete.")
