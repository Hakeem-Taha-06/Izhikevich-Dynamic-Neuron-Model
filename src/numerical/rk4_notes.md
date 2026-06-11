# Role 5: RK4 Notes

## Purpose
Document the explicit RK4 solver contract for the Izhikevich neuron model.

## `config.py` imports
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `a`, `b`, `c`, `d`

## Must achieve
- Explain the solver’s role in the project.
- Record the expected baseline-compatible state format.
- Provide the documentation hook for future implementation work.

## Output interface rule
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

## Constraints
- Documentation only.
- No implementation code.
- No formulas or loops.
- No solver logic."""