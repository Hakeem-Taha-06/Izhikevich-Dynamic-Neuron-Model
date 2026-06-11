# Role 10: ML Notes

## Purpose
Document the ML training workspace for the Izhikevich spiking neuron project (2007 generalized form).

## Model Reference
Izhikevich (2007) generalized biophysical model:
- `C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext`
- `du/dt       = a*{ b*(v - v_r) - u }`
- `if v >= v_peak:  v <- c,  u <- u + d`

## `config.py` imports
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
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