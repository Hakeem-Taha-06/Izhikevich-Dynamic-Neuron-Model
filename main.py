"""Role 1: Master Execution Script

Purpose
-------
Define the top-level execution contract for the Izhikevich spiking neuron project.

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
- Coordinate the project entry point.
- Reference the numerical, ML, and evaluation stages.
- Preserve the shared output contract for every stage.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No orchestration logic.
- No imports beyond documentation-only placeholders in this scaffold."""
