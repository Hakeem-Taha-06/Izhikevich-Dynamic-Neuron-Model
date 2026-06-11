# Role 7: AB2 Notes

## Purpose
Document the Adams-Bashforth 2 solver contract for the Izhikevich neuron model.

## `config.py` imports
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `a`, `b`, `c`, `d`

## Must achieve
- Explain the solver’s role in the project.
- Keep the output interface aligned with the rest of the pipeline.
- Provide a future-facing notes workspace.

## Output interface rule
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

## Constraints
- Documentation only.
- No implementation code.
- No formulas or loops.
- No solver logic."""