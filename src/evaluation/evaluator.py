"""Role 11: Evaluation Workspace

Purpose
-------
Reserve this module for evaluation, benchmarking, and figure-generation contracts.

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
- Document the evaluator workspace for the Izhikevich project.
- Specify the comparison and reporting responsibilities.
- Preserve the final project interface used by the team.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No benchmarking loops.
- No plotting logic.
- Documentation-only scaffold."""