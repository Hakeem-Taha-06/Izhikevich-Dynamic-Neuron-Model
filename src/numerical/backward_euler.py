"""Role 6: Backward Euler Solver — Izhikevich Neuron Model

Purpose
-------
Implements an unconditionally stable implicit Backward Euler integrator for
the Izhikevich (2007) generalized biophysical neuron model.  At each time
step the non-linear algebraic system that arises from the implicit
formulation is solved with ``scipy.optimize.fsolve``.  The discrete
after-spike reset is applied *after* convergence and *before* logging, so
the voltage discontinuity is never straddled by the root-finder.

Model Reference
---------------
Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience:
The Geometry of Excitability and Bursting*. MIT Press.

Governing equations (2007 generalized biophysical form):

    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext   ... (1)
    du/dt       = a * { b*(v - v_r) - u }               ... (2)

After-spike reset (discrete):

    if v >= v_peak:  v <- c,  u <- u + d               ... (3)

Required ``config.py`` imports
------------------------------
- INITIAL_STATE   : np.ndarray shape (2,) — [v0, u0]
- T_START         : float — simulation start time (ms)
- T_END           : float — simulation end time (ms)
- DT_EVAL         : float — default time step (ms)
- I_EXT_DEFAULT   : float — default external current (pA)
- C_m, k, v_r, v_t, v_peak : biophysical constants
- a, b, c, d      : recovery / reset parameters

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
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# Project-root path resolution (works whether called from root or from src/)
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
# (kept local so the solver is entirely self-contained)
# ---------------------------------------------------------------------------

def _f_v(v_val, u_val, I_ext):
    """dv/dt  —  equation (1) rearranged."""
    return (k * (v_val - v_r) * (v_val - v_t) - u_val + I_ext) / C_m


def _f_u(v_val, u_val):
    """du/dt  —  equation (2)."""
    return a * (b * (v_val - v_r) - u_val)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_backward_euler(
    y0=None,
    t_span=None,
    dt=None,
    I_ext=None,
):
    """Integrate the Izhikevich model with the implicit Backward Euler method.

    At every time step the system

        v_{n+1} = v_n + dt * f_v(v_{n+1}, u_{n+1})
        u_{n+1} = u_n + dt * f_u(v_{n+1}, u_{n+1})

    is solved for (v_{n+1}, u_{n+1}) using ``scipy.optimize.fsolve``.
    The current state (v_n, u_n) is used as the initial guess, which is
    close to the true solution for small dt and keeps iteration counts low.

    Immediately after the root-finder converges, the discrete reset
    condition (3) is checked and applied if necessary.  This guarantees
    that the reset discontinuity is never seen by the root-finder, so
    convergence is never compromised by the spike.

    Parameters
    ----------
    y0 : array-like of shape (2,), optional
        Initial state [v0 (mV), u0 (pA)].  Defaults to ``INITIAL_STATE``.
    t_span : tuple (t_start, t_end), optional
        Simulation window in ms.  Defaults to ``(T_START, T_END)``.
    dt : float, optional
        Integration time step in ms.  Defaults to ``DT_EVAL``.
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
    # 4. Main integration loop                                             #
    # ------------------------------------------------------------------ #
    for i in range(1, N):

        # -- 4a. Define implicit residual ----------------------------------
        # G(v_next, u_next) = 0  <==>
        #   v_next - v_curr - dt * f_v(v_next, u_next) = 0
        #   u_next - u_curr - dt * f_u(v_next, u_next) = 0
        def _residual(y_next):
            v_n, u_n = y_next
            res_v = v_n - v_curr - dt * _f_v(v_n, u_n, I_ext)
            res_u = u_n - u_curr - dt * _f_u(v_n, u_n)
            return np.array([res_v, res_u])

        # -- 4b. Solve with fsolve (initial guess = current state) ---------
        y_next, _, ier, _ = fsolve(
            _residual,
            x0=np.array([v_curr, u_curr]),
            full_output=True,
        )
        # ier == 1 means a solution was found; we proceed regardless
        # (fsolve already raises a warning if it cannot converge).
        v_next, u_next = y_next[0], y_next[1]

        # -- 4c. Discrete reset AFTER convergence, BEFORE logging ----------
        # This ensures the root-finder never straddles the discontinuity.
        if v_next >= v_peak:
            v_next = c          # reset membrane potential  (mV)
            u_next = u_next + d  # increment recovery variable (pA)

        # -- 4d. Log and advance state -------------------------------------
        results[i, 0] = t_values[i]
        results[i, 1] = v_next
        results[i, 2] = u_next

        v_curr = v_next
        u_curr = u_next

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Role 6 — Backward Euler Solver  (self-test)")
    print("=" * 60)

    res = solve_backward_euler()

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

    # Count spikes (v resets back to c = -50 mV after hitting v_peak)
    spike_indices = np.where(np.diff(res[:, 1]) < -50)[0]
    print(f"\nDetected spikes: {len(spike_indices)}")
    print("Self-test complete.")