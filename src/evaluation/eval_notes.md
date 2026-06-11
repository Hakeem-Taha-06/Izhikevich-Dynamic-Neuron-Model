# Role 11: Evaluation Notes

## Purpose
Document the evaluation workspace for the Izhikevich spiking neuron project (2007 generalized form).

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
- Explain how solver comparison is organized.
- Define how figures and metrics are reported.
- Preserve the shared output interface contract.

## Output interface rule
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

## Constraints
- Documentation only.
- No implementation code.
- No analysis loops.
- No plotting code.