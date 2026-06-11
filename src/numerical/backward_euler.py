"""Role 6: Backward Euler Solver Workspace

Purpose
-------
Reserve this module for the implicit Backward Euler solver documentation and interface contract.

Required `config.py` imports
----------------------------
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `a`, `b`, `c`, `d`

Must achieve
------------
- Describe the implicit solver’s role in the Izhikevich project.
- Preserve the baseline-compatible output contract.
- Support the solver-comparison and evaluation pipeline.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No loops.
- No equation-solving logic.
- Documentation-only scaffold."""