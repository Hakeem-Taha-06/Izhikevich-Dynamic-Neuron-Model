"""Machine learning package for the Izhikevich spiking neuron project.

Role mapping:
- Role 8: architecture.py
- Role 9: physics_loss.py
- Role 10: train.py

Expected `config.py` imports:
- `a`, `b`, `c`, `d`
- `INITIAL_STATE`
- `T_START`, `T_END`, `DT_EVAL`, `I_EXT_DEFAULT`

Output interface rule:
All numerical and ML predictive outputs must return `numpy.ndarray` of shape `(N, 3)` ordered as `[Time, v, u]`.

This package contains documentation-only stubs in this scaffold."""