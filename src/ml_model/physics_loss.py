"""Role 9: Physics-Informed Loss Function

Purpose
-------
Compute the ODE-residual (physics) loss for the Izhikevich 2007 model.
Forces the neural network to respect the governing differential equations
during training by penalising violations of the continuous dynamics.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext(t)
    dw/dt       = a*{ b*(v - v_r) - w }
    if v >= v_peak:  v <- c,  w <- w + d

Step-current protocol:
    I_ext(t) = 0       for t < T_STIM_ONSET
    I_ext(t) = 70 pA   for t >= T_STIM_ONSET

Spike Masking Strategy
----------------------
The discrete reset at v_peak creates a discontinuity that produces
infinite gradients and crashes training.  This module uses a **peak-
margin masking** approach that excludes collocation points whose
predicted voltage is in the fast-upstroke zone near the spike peak.

Why peak-only masking?
    The quadratic term k*(v - v_r)*(v - v_t) grows rapidly above the
    threshold v_t = -40 mV.  By v = 0 mV the dynamics reach ~22 mV/ms
    and by v_peak = 35 mV they exceed 55 mV/ms.  These extreme rates
    make the ODE residual unreliable during the spike upstroke.

    A separate "near-reset" mask (v < c + margin) is intentionally NOT
    used because c = -50 mV is close to v_r = -60 mV — the neuron's
    resting state.  Masking that region would exclude the bulk of the
    subthreshold dynamics where the ODE is smooth and perfectly valid,
    destroying the network's ability to learn the resting behaviour.

The peak margin can be tightened over training (curriculum schedule)
via the ``get_margin`` helper.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return ``numpy.ndarray``
of shape ``(N, 3)`` ordered as ``[Time, v, w]``.
"""

import torch
from config import (
    C_m, k, v_r, v_t, v_peak, a, b, c, d,
    I_EXT_DEFAULT, T_STIM_ONSET,
    INITIAL_STATE,
)


# =====================================================================
# STEP-CURRENT FUNCTION (PyTorch-compatible)
# =====================================================================

def I_ext_torch(t):
    """Step-current stimulus for PyTorch tensors.

    Returns 0.0 for t < T_STIM_ONSET, I_EXT_DEFAULT for t >= T_STIM_ONSET.
    The result is detached from the computation graph since I_ext is an
    external forcing term (not a learnable quantity).

    Parameters
    ----------
    t : torch.Tensor, any shape
        Time values in ms.

    Returns
    -------
    torch.Tensor, same shape as t
        External current in pA.
    """
    return I_EXT_DEFAULT * (t >= T_STIM_ONSET).float()


# =====================================================================
# CHARACTERISTIC ODE RATES (for residual normalization)
# =====================================================================
# Without normalization, mean(R_v²) and mean(R_w²) have different
# physical units ((mV/ms)² vs (pA/ms)²) and different magnitudes.
# Dividing each residual by a characteristic rate makes both terms
# dimensionless and roughly equal in scale, so neither dominates the
# total physics loss.
#
# DV_RATE: the voltage rate driven by external current alone at rest
#          = I_EXT_DEFAULT / C_m = 70 / 100 = 0.7 mV/ms
# DW_RATE: the recovery rate after a spike (w jumps by d, then decays)
#          = a * |d| = 0.03 * 100 = 3.0 pA/ms
DV_RATE = I_EXT_DEFAULT / C_m       # 0.7  mV/ms
DW_RATE = a * abs(d)                # 3.0  pA/ms


# =====================================================================
# 1. PHYSICS LOSS
# =====================================================================

def compute_physics_loss(model, t, peak_margin=10.0):
    """Compute the physics-informed ODE-residual loss.

    The step-current I_ext(t) is computed internally from the time
    values — it is not a network input.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network (Role 8).  Must accept a tensor of shape
        ``(N, 1)`` (time in ms) and return a tensor of shape
        ``(N, 2)`` ordered as ``[v_pred, w_pred]``.
    t : torch.Tensor, shape (N, 1), **requires_grad=True**
        Time collocation points (ms).  Gradient tracking is mandatory
        so that ``torch.autograd.grad`` can compute dv/dt and dw/dt.
    peak_margin : float, optional
        How far below v_peak (mV) to start masking.  Points with
        ``v_pred > v_peak - peak_margin`` are excluded from the
        residual.  Default is 10.0 mV.
        Use ``get_margin()`` for curriculum scheduling.

    Returns
    -------
    loss_physics : torch.Tensor (scalar)
        Mean-squared ODE residual, masked in the spike upstroke zone.
    """
    # ── Step 1: Forward pass ──────────────────────────────────────────
    outputs = model(t)                                      # (N, 2)
    v_pred = outputs[:, 0:1]                                # (N, 1)
    w_pred = outputs[:, 1:2]                                # (N, 1)

    # ── Step 2: Time derivatives via autograd ─────────────────────────
    dv_dt = torch.autograd.grad(
        outputs=v_pred,
        inputs=t,
        grad_outputs=torch.ones_like(v_pred),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 1)

    dw_dt = torch.autograd.grad(
        outputs=w_pred,
        inputs=t,
        grad_outputs=torch.ones_like(w_pred),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 1)

    # ── Step 3: Compute I_ext(t) from step-current protocol ───────────
    I_ext = I_ext_torch(t.detach())                         # (N, 1)

    # ── Step 4: ODE right-hand sides ──────────────────────────────────
    #   dv/dt = [k*(v - v_r)*(v - v_t) - w + I_ext(t)] / C_m
    #   dw/dt = a * [b*(v - v_r) - w]
    rhs_v = (k * (v_pred - v_r) * (v_pred - v_t) - w_pred + I_ext) / C_m
    rhs_w = a * (b * (v_pred - v_r) - w_pred)

    # ── Step 5: Residuals ─────────────────────────────────────────────
    R_v = dv_dt - rhs_v
    R_w = dw_dt - rhs_w

    # ── Step 6: Peak-only spike masking ───────────────────────────────
    valid = ~(v_pred > (v_peak - peak_margin)).squeeze()    # True → keep

    R_v_valid = R_v[valid]
    R_w_valid = R_w[valid]

    # ── Step 7: Normalise and compute mean squared residual ───────────
    if R_v_valid.numel() == 0:
        loss_physics = torch.tensor(0.0, device=t.device, requires_grad=True)
    else:
        loss_physics = (
            torch.mean((R_v_valid / DV_RATE) ** 2)
            + torch.mean((R_w_valid / DW_RATE) ** 2)
        )

    return loss_physics


# =====================================================================
# 2. INITIAL-CONDITION LOSS
# =====================================================================

def compute_ic_loss(model):
    """Compute the initial-condition loss at t = 0.

    Uses the fixed initial conditions from config.py (INITIAL_STATE).

    Parameters
    ----------
    model : torch.nn.Module
        The neural network (Role 8).

    Returns
    -------
    loss_ic : torch.Tensor (scalar)
        Normalised MSE between the network's t=0 prediction and the
        true ICs.
    """
    # Characteristic scales for normalisation
    V_SCALE = v_peak - v_r          # 95.0 mV
    W_SCALE = abs(d)                # 100.0 pA

    # Known initial conditions from config
    V_0 = float(INITIAL_STATE[0])   # -60.0 mV
    W_0 = float(INITIAL_STATE[1])   #   0.0 pA

    device = next(model.parameters()).device
    t_zero = torch.zeros(1, 1, device=device)
    outputs = model(t_zero)
    v_pred_0 = outputs[:, 0:1]
    w_pred_0 = outputs[:, 1:2]

    loss_ic = (
        ((v_pred_0 - V_0) / V_SCALE) ** 2
        + ((w_pred_0 - W_0) / W_SCALE) ** 2
    ).mean()

    return loss_ic


# =====================================================================
# 3. CURRICULUM MARGIN SCHEDULE
# =====================================================================

def get_margin(epoch, max_epochs, start=20.0, end=5.0):
    """Compute the spike-masking margin for the current epoch.

    Implements a linear curriculum that begins with a wide margin
    (stable, forgiving) and tightens to a narrow margin (precise)
    as training progresses.

    Parameters
    ----------
    epoch : int
        Current training epoch (0-indexed).
    max_epochs : int
        Total number of planned training epochs.
    start : float, optional
        Margin at epoch 0 (mV).  Default 20.0.
    end : float, optional
        Margin at the final epoch (mV).  Default 5.0.

    Returns
    -------
    margin : float
        The margin to pass to ``compute_physics_loss(..., peak_margin=margin)``.

    Examples
    --------
    >>> get_margin(0, 1000)
    20.0
    >>> get_margin(500, 1000)
    12.5
    >>> get_margin(1000, 1000)
    5.0
    """
    progress = min(epoch / max(max_epochs, 1), 1.0)
    return start + (end - start) * progress


# =====================================================================
# INSTRUCTIONS FOR ROLE 10 (Training Loop Operator)
# =====================================================================
#
# This section explains how to integrate the physics loss and the
# curriculum margin schedule into your training loop.
#
# ── Imports ──────────────────────────────────────────────────────────
#
#   from src.ml_model.physics_loss import (
#       compute_physics_loss,
#       compute_ic_loss,
#       get_margin,
#   )
#
# ── Inside your training loop ────────────────────────────────────────
#
#   IMPORTANT: `t` must have requires_grad=True for autograd to work.
#
#   max_epochs = 8000
#
#   for epoch in range(max_epochs):
#       t_batch = ...   # shape (N, 1), time samples
#       t_batch.requires_grad_(True)
#
#       # 1. Data loss (supervised, if using hybrid approach)
#       outputs = model(t_data)                 # (N, 2) → [v, w]
#       loss_data = mse(outputs, targets)
#
#       # 2. Physics loss with curriculum margin
#       margin = get_margin(epoch, max_epochs)
#       loss_phys = compute_physics_loss(model, t_batch, peak_margin=margin)
#
#       # 3. IC loss (anchors trajectory at t=0)
#       loss_ic = compute_ic_loss(model)
#
#       # 4. Combine
#       loss_total = loss_data + λ_phys * loss_phys + λ_ic * loss_ic
#
#       optimizer.zero_grad()
#       loss_total.backward()
#       optimizer.step()
#
# ── Hyperparameter guidance ──────────────────────────────────────────
#
#   lambda_phys : Start around 1.0.  Physics residual is now
#                 dimensionless (normalized by DV_RATE and DW_RATE).
#
#   lambda_ic   : Start around 200.0.  Strong IC anchoring helps
#                 convergence (per comparison repo's approach).
#
#   get_margin(start, end) :
#       Defaults (20 → 5 mV) are calibrated for the Regular Spiking
#       regime.  If training is unstable early on, increase `start`.
# =====================================================================