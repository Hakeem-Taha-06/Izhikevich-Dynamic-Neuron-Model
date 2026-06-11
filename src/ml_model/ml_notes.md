# Role 10: ML Notes

## Purpose
Document the ML training workspace for the Izhikevich spiking neuron project.

## `config.py` imports
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `a`, `b`, `c`, `d`

## Must achieve
- Explain the purpose of the training stage.
- Clarify where outputs are expected to be stored.
- Preserve the shared `(N, 3)` output contract.

## Output interface rule
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

## Constraints
- Documentation only.
- No implementation code.
- No training logic.
- No equations or loops.