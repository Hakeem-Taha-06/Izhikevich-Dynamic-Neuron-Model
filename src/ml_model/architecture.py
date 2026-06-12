"""Role 8: ML Architecture — Izhikevich PINN

Purpose
-------
Neural-network architecture for the Izhikevich PINN model.
Takes only time ``t`` as input and predicts ``[v(t), w(t)]`` in physical
units.  The step-current protocol is handled internally by the physics
loss (Role 9), not by the network.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext(t)
    dw/dt       = a*{ b*(v - v_r) - w }
    if v >= v_peak:  v <- c,  w <- w + d

Network contract
----------------
The network's ``forward()`` accepts a tensor of shape ``(N, 1)``
containing time values (ms) and returns a tensor of shape ``(N, 2)``
ordered as ``[v_pred, w_pred]`` **in physical units** (mV and pA).

Output denormalization
-----------------------
The raw output of the hidden network is centered near zero (typical for
networks with Tanh activations).  A fixed affine denormalization layer
maps these raw outputs to physical units:

    v = raw_v * V_SCALE + V_SHIFT      (mV)
    w = raw_w * W_SCALE + W_SHIFT      (pA)

The constants are chosen so that at initialisation (raw ≈ 0), the
network predicts the resting state (v ≈ v_r, w ≈ 0).

Note on the project-wide `(N, 3)` "[Time, v, w]" output rule
---------------------------------------------------------------
That convention applies to *exported trajectory arrays* used for
comparison against the ground truth / numerical solvers (Role 11), not
to this module's `forward()` pass. Use `predict_trajectory()` below to
produce that `(N, 3)` array from a trained model.
"""

import numpy as np
import torch
import torch.nn as nn

# Required config.py imports
try:
    from config import (
        INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
        T_STIM_ONSET,
        C_m, k, v_r, v_t, v_peak, a, b, c, d
    )
except ImportError:
    # Fallback for isolated testing
    v_r, v_peak, d = -60.0, 35.0, 100.0
    T_END = 1000.0


# =====================================================================
# Denormalization constants (derived from config)
# =====================================================================
# v: shift to resting potential, scale by full operating range
#    raw=0 → v_r = -60 mV  (resting state at init)
#    raw=1 → v_r + 95 = 35 mV = v_peak
V_SHIFT = v_r                   # -60.0 mV
V_SCALE = v_peak - v_r          #  95.0 mV

# w: shift to steady-state recovery, scale by spike-reset increment
#    raw=0 → 0 pA (steady state when v = v_r)
#    raw=1 → |d| = 100 pA (one spike's worth of recovery current)
W_SHIFT = 0.0                   #   0.0 pA
W_SCALE = abs(d)                # 100.0 pA


class IzhikevichPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Izhikevich model.

    Input:  t  — shape (N, 1), time in ms
    Output: [v, w] — shape (N, 2), physical units (mV, pA)

    The step-current I_ext(t) is NOT an input to the network.
    It is computed inside the physics loss from t, so the network
    only needs to learn the mapping  t → (v, w).
    """

    def __init__(self, hidden_size=128, num_layers=6):
        super().__init__()

        # Store denormalization constants as buffers (non-trainable,
        # move with device, saved in state_dict).
        self.register_buffer('v_scale', torch.tensor(V_SCALE, dtype=torch.float32))
        self.register_buffer('v_shift', torch.tensor(V_SHIFT, dtype=torch.float32))
        self.register_buffer('w_scale', torch.tensor(W_SCALE, dtype=torch.float32))
        self.register_buffer('w_shift', torch.tensor(W_SHIFT, dtype=torch.float32))

        # Time normalization constant
        self.register_buffer('t_max', torch.tensor(T_END, dtype=torch.float32))

        layers = []
        # Input layer: receives t only → (N, 1)
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # Output layer: raw [v_raw, w_raw]
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

        # Xavier initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        """
        Forward pass.

        Args:
            t (torch.Tensor): Time tensor of shape (N, 1) in ms.

        Returns:
            torch.Tensor: shape (N, 2), columns [v_pred, w_pred]
                in physical units (mV and pA).
        """
        # Normalize time to [-1, 1] for better training stability
        t_norm = (t - self.t_max / 2) / (self.t_max / 2)

        raw = self.network(t_norm)                              # (N, 2)

        # Denormalize: map raw outputs to physical units.
        v = raw[:, 0:1] * self.v_scale + self.v_shift           # (N, 1) mV
        w = raw[:, 1:2] * self.w_scale + self.w_shift           # (N, 1) pA

        return torch.cat([v, w], dim=1)                         # (N, 2)


def predict_trajectory(model, t_start=None, t_end=None, dt=None):
    """
    Run the model over a time range and produce the project-wide
    `(N, 3)` "[Time, v, w]" export array.

    Args:
        model (IzhikevichPINN): Trained network.
        t_start (float): Start time (ms). Defaults to T_START.
        t_end (float): End time (ms). Defaults to T_END.
        dt (float): Time step (ms). Defaults to DT_EVAL.

    Returns:
        numpy.ndarray: shape (N, 3), columns [Time, v, w].
    """
    if t_start is None:
        t_start = T_START
    if t_end is None:
        t_end = T_END
    if dt is None:
        dt = DT_EVAL

    time_np = np.arange(t_start, t_end + dt * 0.5, dt)
    t_tensor = torch.tensor(time_np, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        t_tensor = t_tensor.to(device)
        outputs = model(t_tensor).cpu().numpy()      # (N, 2) → [v, w]

    # Assemble [Time, v, w]
    return np.column_stack([time_np, outputs])        # (N, 3)
