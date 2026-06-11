# Role 6: Backward Euler Notes

## Purpose
Document the implicit Backward Euler solver contract for the Izhikevich neuron model.

## `config.py` imports
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `a`, `b`, `c`, `d`

## Must achieve
- Explain the solver’s function in the project.
- Record the expected I/O shape and ordering.
- Provide the placeholder for future implementation notes.

## Output interface rule
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

## Constraints
- Documentation only.
- No implementation code.
- No formulas or loops.
- No solver logic."""