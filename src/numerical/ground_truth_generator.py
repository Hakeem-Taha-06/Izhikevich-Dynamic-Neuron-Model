"""Role 4: Ground Truth Generator

Purpose
-------
Define the data-generation workspace for the Izhikevich spiking neuron project.
This module is reserved for the baseline dataset pipeline and its narrative contract.

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
- Produce the baseline trajectory dataset used by the downstream numerical and ML stages.
- Save the generated dataset under `data/`.
- Preserve the project-wide output schema exactly.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

Constraints
-----------
- No implementation code.
- No loops.
- No equation-solving logic.
- This file is documentation-only in the scaffold."""