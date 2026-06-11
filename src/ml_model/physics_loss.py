"""Role 9: Physics Loss Workspace

Purpose
-------
Reserve this module for the physics-informed loss contract used in the Izhikevich project.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a*{ b*(v - v_r) - u }
    if v >= v_peak:  v <- c,  u <- u + d

Required `config.py` imports
----------------------------
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
- `a`, `b`, `c`, `d`
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`

Must achieve
------------
- Describe the loss-function workspace for the ML stage.
- State how the physics loss must align with the project's state ordering.
- Preserve the shared output interface for downstream tools.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No autograd logic.
- No residual equations.
- Documentation-only scaffold."""