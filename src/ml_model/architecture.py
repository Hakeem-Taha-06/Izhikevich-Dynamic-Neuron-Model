"""Role 8: ML Architecture Workspace

Purpose
-------
Neural-network architecture implementation for the Izhikevich PINN model.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a*{ b*(v - v_r) - u }
    if v >= v_peak:  v <- c,  u <- u + d

Required `config.py` imports
----------------------------
- `INITIAL_STATE`
- `T_START`
- `T_END`
- `DT_EVAL`
- `I_EXT_DEFAULT`
- `C_m`, `k`, `v_r`, `v_t`, `v_peak`
- `a`, `b`, `c`, `d`

Model interface contract (matches src/ml_model/physics_loss.py)
-----------------------------------------------------------------
The network's `forward()` accepts a tensor of shape ``(N, 4)`` ordered as
``[t, I_ext, V_0, U_0]`` and returns a tensor of shape ``(N, 2)`` ordered
as ``[v_pred, u_pred]`` **in physical units** (mV and pA respectively).

Conditioning the network on `I_ext`, `V_0`, and `U_0` (in addition to
`t`) is required so that a single trained model can represent the full
family of trajectories in the ~4,000-simulation ground-truth dataset
(Role 4), which sweeps these three quantities.

Output denormalization
-----------------------
The raw output of the hidden network is centered near zero (typical for
networks with Tanh activations).  A fixed affine denormalization layer
maps these raw outputs to physical units:

    v = raw_v * V_SCALE + V_SHIFT      (mV)
    u = raw_u * U_SCALE + U_SHIFT      (pA)

The constants are chosen so that at initialisation (raw ≈ 0), the
network predicts the resting state (v ≈ v_r, u ≈ 0).  This gives
training a head start — the network already satisfies the ODE at rest
before any weight updates.

Because denormalization happens inside ``forward()``, autograd
automatically chains through it when ``compute_physics_loss()``
computes dv/dt and du/dt.  No special handling is needed in the loss.

Note on the project-wide `(N, 3)` "[Time, v, u]" output rule
---------------------------------------------------------------
That convention applies to *exported trajectory arrays* used for
comparison against the ground truth / numerical solvers (Role 11), not
to this module's `forward()` pass. Use `predict_trajectory()` below to
produce that `(N, 3)` array from a trained model.
"""

import torch
import torch.nn as nn

# Required config.py imports as specified by the Team Leader
try:
    from config import (
        INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
        C_m, k, v_r, v_t, v_peak, a, b, c, d
    )
except ImportError:
    # Exception handling to allow local testing without the config file
    pass


# =====================================================================
# Denormalization constants (derived from config)
# =====================================================================
# v: shift to resting potential, scale by full operating range
#    raw=0 → v_r = -60 mV  (resting state at init)
#    raw=1 → v_r + 95 = 35 mV = v_peak
V_SHIFT = v_r                   # -60.0 mV
V_SCALE = v_peak - v_r          #  95.0 mV

# u: shift to steady-state recovery, scale by spike-reset increment
#    raw=0 → 0 pA (steady state when v = v_r)
#    raw=1 → |d| = 100 pA (one spike's worth of recovery current)
U_SHIFT = 0.0                   #   0.0 pA
U_SCALE = abs(d)                # 100.0 pA


class IzhikevichPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) architecture for the Izhikevich model.

    The raw network outputs are denormalized in ``forward()`` so that
    all downstream consumers (physics loss, IC loss, evaluation) receive
    values in physical units (mV for v, pA for u) without needing to
    know about the internal scaling.
    """

    def __init__(self, hidden_size=64, num_layers=3):
        super().__init__()

        # Store denormalization constants as buffers (non-trainable,
        # move with device, saved in state_dict).
        self.register_buffer('v_scale', torch.tensor(V_SCALE, dtype=torch.float32))
        self.register_buffer('v_shift', torch.tensor(V_SHIFT, dtype=torch.float32))
        self.register_buffer('u_scale', torch.tensor(U_SCALE, dtype=torch.float32))
        self.register_buffer('u_shift', torch.tensor(U_SHIFT, dtype=torch.float32))

        layers = []
        # Input layer: receives [t, I_ext, V_0, U_0]
        layers.append(nn.Linear(4, hidden_size))
        layers.append(nn.Tanh())
        # Hidden layers: processes dynamic patterns
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # Output layer: raw [v_raw, u_raw] (no activation function)
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the PINN.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 4), columns
                ordered as [t, I_ext, V_0, U_0].

        Returns:
            torch.Tensor: Output tensor of shape (N, 2), columns
                ordered as [v_pred, u_pred] **in physical units**
                (mV and pA).  This is the format consumed directly
                by compute_physics_loss() and compute_ic_loss() in
                physics_loss.py.
        """
        raw = self.network(x)                               # (N, 2)

        # Denormalize: map raw outputs to physical units.
        # Autograd chains through this affine transform automatically,
        # so dv_physical/dt = dv_raw/dt * v_scale.
        v = raw[:, 0:1] * self.v_scale + self.v_shift       # (N, 1) mV
        u = raw[:, 1:2] * self.u_scale + self.u_shift       # (N, 1) pA

        return torch.cat([v, u], dim=1)                     # (N, 2)


def predict_trajectory(model, t, I_ext, V_0, U_0):
    """
    Run the model over a batch of inputs and assemble the project-wide
    `(N, 3)` "[Time, v, u]" export array.

    Args:
        model (IzhikevichPINN): Trained network.
        t (torch.Tensor): shape (N, 1), time values (ms).
        I_ext (torch.Tensor): shape (N, 1), external current (pA).
        V_0 (torch.Tensor): shape (N, 1), initial membrane potential (mV).
        U_0 (torch.Tensor): shape (N, 1), initial recovery variable (pA).

    Returns:
        torch.Tensor: shape (N, 3), columns ordered as [Time, v, u].
    """
    inputs = torch.cat([t, I_ext, V_0, U_0], dim=1)  # (N, 4)
    outputs = model(inputs)                          # (N, 2) -> [v, u]
    return torch.cat((t, outputs), dim=1)            # (N, 3) -> [Time, v, u]
