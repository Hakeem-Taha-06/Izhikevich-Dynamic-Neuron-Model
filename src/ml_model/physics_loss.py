"""Role 9: Physics-Informed Loss Function

Purpose
-------
Compute the ODE-residual (physics) loss for the Izhikevich 2007 model.
Forces the neural network to respect the governing differential equations
during training by penalising violations of the continuous dynamics.

Model Reference
---------------
Izhikevich (2007) generalized biophysical model:
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - w + I_ext
    dw/dt       = a*{ b*(v - v_r) - w }
    if v >= v_peak:  v <- c,  w <- w + d

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
via the ``get_margin`` helper.  See the "Instructions for Role 10"
section at the bottom of this file.

Strict output interface rule
-----------------------------
All numerical and ML predictive outputs must return ``numpy.ndarray``
of shape ``(N, 3)`` ordered as ``[Time, v, w]``.
"""

import torch
from config import C_m, k, v_r, v_t, v_peak, a, b, c, d, I_EXT_DEFAULT


# =====================================================================
# CHARACTERISTIC ODE RATES (for residual normalization)
# =====================================================================
# Without normalization, mean(R_v²) and mean(R_u²) have different
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

def compute_physics_loss(model, t, I_ext, V_0, W_0, peak_margin=10.0):
    """Compute the physics-informed ODE-residual loss.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network (Role 8).  Must accept a tensor of shape
        ``(N, 4)`` ordered as ``[t, I_ext, V_0, W_0]`` and return a
        tensor of shape ``(N, 2)`` ordered as ``[v_pred, w_pred]``.
    t : torch.Tensor, shape (N, 1), **requires_grad=True**
        Time collocation points (ms).  Gradient tracking is mandatory
        so that ``torch.autograd.grad`` can compute dv/dt and dw/dt.
    I_ext : torch.Tensor, shape (N, 1)
        External current for each sample (pA).
    V_0 : torch.Tensor, shape (N, 1)
        Initial membrane potential for each sample (mV).
    W_0 : torch.Tensor, shape (N, 1)
        Initial recovery variable for each sample (pA).
    peak_margin : float, optional
        How far below v_peak (mV) to start masking.  Points with
        ``v_pred > v_peak - peak_margin`` are excluded from the
        residual.

        Physical justification for the default and curriculum range:

        ======  ============================  =======================
        v (mV)  k(v-v_r)(v-v_t)  (quadratic)  approx dv/dt (mV/ms)
        ======  ============================  =======================
          -40       0                              5.0  (threshold)
          -20     560                             10.6
            0    1680                             21.8
           15    2888                             33.9  ← start mask
           25    3868                             43.7
           30    4410                             49.1  ← end mask
           35    4988                             54.9  (v_peak)
        ======  ============================  =======================

        * ``start=20`` → masks v > 15 mV  (dv/dt > 34 mV/ms)
          Conservative early training; covers the entire fast upstroke.
        * ``end=5``   → masks v > 30 mV  (dv/dt > 49 mV/ms)
          Tight late training; only the spike tip is excluded.

        Default is 10.0 mV (masks v > 25 mV, dv/dt > 44 mV/ms).
        Use ``get_margin()`` for curriculum scheduling.

    Returns
    -------
    loss_physics : torch.Tensor (scalar)
        Mean-squared ODE residual, masked in the spike upstroke zone.

    Notes
    -----
    No near-reset mask is applied.  The reset voltage c = -50 mV is
    only 10 mV above the resting potential v_r = -60 mV.  A margin
    mask around c would suppress the entire subthreshold region where
    the ODE is smooth and valid, crippling the physics enforcement.
    """
    # ── Step 1: Forward pass ──────────────────────────────────────────
    inputs = torch.cat([t, I_ext, V_0, W_0], dim=1)      # (N, 4)
    outputs = model(inputs)                                 # (N, 2)
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

    # ── Step 3: ODE right-hand sides ──────────────────────────────────
    #   dv/dt = [k*(v - v_r)*(v - v_t) - w + I_ext] / C_m
    #   dw/dt = a * [b*(v - v_r) - w]
    rhs_v = (k * (v_pred - v_r) * (v_pred - v_t) - w_pred + I_ext) / C_m
    rhs_u = a * (b * (v_pred - v_r) - w_pred)

    # ── Step 4: Residuals ─────────────────────────────────────────────
    R_v = dv_dt - rhs_v
    R_w = dw_dt - rhs_u

    # ── Step 5: Peak-only spike masking ───────────────────────────────
    # Mask the fast-upstroke zone where dv/dt exceeds ~34-49 mV/ms
    # and the network's continuous approximation of the spike produces
    # unreliable autograd derivatives.
    #
    # No near-reset mask is used because c = -50 mV sits inside the
    # normal subthreshold operating range (v_r = -60 to v_t = -40).
    #
    # Boolean indexing is used instead of torch.where to ensure that
    # masked (excluded) points do NOT contribute to the denominator of
    # the mean.  With torch.where, zeroed entries would dilute the
    # loss proportionally to the fraction of masked points.
    valid = ~(v_pred > (v_peak - peak_margin)).squeeze()    # True → keep

    R_v_valid = R_v[valid]
    R_w_valid = R_w[valid]

    # ── Step 6: Normalise and compute mean squared residual ───────────
    # Divide each residual by its characteristic ODE rate so that both
    # terms are dimensionless and contribute equally to the loss.
    # Without this, R_v² (voltage) dominates R_u² (recovery) by ~10–25×.
    #
    # Guard against the edge case where every point is masked (e.g. a
    # batch that consists entirely of spike peaks).
    if R_v_valid.numel() == 0:
        loss_physics = torch.tensor(0.0, device=t.device, requires_grad=True)
    else:
        loss_physics = (
            torch.mean((R_v_valid / DV_RATE) ** 2)
            + torch.mean((R_w_valid / DW_RATE) ** 2)
        )

    return loss_physics


# =====================================================================
# 2. INITIAL-CONDITION LOSS (optional helper)
# =====================================================================

def compute_ic_loss(model, I_ext_0, V_0, W_0):
    """Compute the initial-condition loss at t = 0.

    Penalises the network if its prediction at t = 0 does not match
    the prescribed initial state (V_0, W_0).

    Parameters
    ----------
    model : torch.nn.Module
        The neural network (Role 8).
    I_ext_0 : torch.Tensor, shape (N, 1)
        External current for each sample (pA).
    V_0 : torch.Tensor, shape (N, 1)
        Target initial membrane potential (mV).
    W_0 : torch.Tensor, shape (N, 1)
        Target initial recovery variable (pA).

    Returns
    -------
    loss_ic : torch.Tensor (scalar)
        Normalised MSE between the network's t=0 prediction and the
        true ICs.  Each term is divided by its characteristic scale
        so that voltage (mV) and recovery (pA) contribute equally.
    """
    # Characteristic scales for normalisation:
    #   v spans roughly v_r to v_peak  →  95 mV
    #   w jumps by d at each spike     →  100 pA
    V_SCALE = v_peak - v_r          # 35 - (-60) = 95.0 mV
    W_SCALE = abs(d)                # 100.0 pA

    t_zero = torch.zeros_like(V_0, requires_grad=False)
    inputs = torch.cat([t_zero, I_ext_0, V_0, W_0], dim=1)
    outputs = model(inputs)
    v_pred_0 = outputs[:, 0:1]
    w_pred_0 = outputs[:, 1:2]

    loss_ic = (
        torch.mean(((v_pred_0 - V_0) / V_SCALE) ** 2)
        + torch.mean(((w_pred_0 - W_0) / W_SCALE) ** 2)
    )
    return loss_ic


# =====================================================================
# 3. CURRICULUM MARGIN SCHEDULE
# =====================================================================

def get_margin(epoch, max_epochs, start=20.0, end=5.0):
    """Compute the spike-masking margin for the current epoch.

    Implements a linear curriculum that begins with a wide margin
    (stable, forgiving) and tightens to a narrow margin (precise)
    as training progresses.

    Physical basis for the defaults
    --------------------------------
    The quadratic term k*(v-v_r)*(v-v_t) in the 2007 Izhikevich model
    accelerates rapidly as v rises above the threshold v_t = -40 mV:

        v = 15 mV  →  dv/dt ≈ 34 mV/ms   (start: mask above here)
        v = 30 mV  →  dv/dt ≈ 49 mV/ms   (end:   mask above here)
        v = 35 mV  →  dv/dt ≈ 55 mV/ms   (v_peak — always masked)

    * ``start=20.0`` → masks v > 15 mV.  At epoch 0 the network is
      untrained, so we exclude the entire fast upstroke (dv/dt > 34)
      to prevent gradient blow-up.  This still leaves 79% of the
      voltage range (-60 to 15 mV) under physics enforcement.

    * ``end=5.0`` → masks v > 30 mV.  By the final epoch the network
      has learned the smooth dynamics and can tolerate steeper slopes.
      Only the spike tip (dv/dt > 49) is excluded, leaving 95% of the
      voltage range under enforcement.

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
#       compute_ic_loss,       # optional but recommended
#       get_margin,
#   )
#
# ── Inside your training loop ────────────────────────────────────────
#
#   IMPORTANT: `t` must have requires_grad=True for autograd to work.
#
#   max_epochs = 5000  # or whatever you choose
#
#   for epoch in range(max_epochs):
#       for batch in dataloader:
#           t, I_ext, V_0, W_0, v_gt, w_gt = unpack(batch)
#           t.requires_grad_(True)                      # ← mandatory
#
#           # 1. Data loss (yours to define)
#           outputs = model(torch.cat([t, I_ext, V_0, W_0], dim=1))
#           loss_data = mse(outputs, targets)
#
#           # 2. Physics loss with curriculum margin
#           margin = get_margin(epoch, max_epochs)
#           loss_phys = compute_physics_loss(
#               model, t, I_ext, V_0, W_0, peak_margin=margin
#           )
#
#           # 3. IC loss (optional, anchors trajectory to t=0)
#           loss_ic = compute_ic_loss(model, I_ext, V_0, W_0)
#
#           # 4. Combine with your chosen weights
#           loss_total = (
#               loss_data
#               + lambda_phys * loss_phys
#               + lambda_ic * loss_ic
#           )
#
#           optimizer.zero_grad()
#           loss_total.backward()
#           optimizer.step()
#
# ── Hyperparameter guidance ──────────────────────────────────────────
#
#   lambda_phys : Start around 0.01.  The physics residual is measured
#                 in (mV/ms)^2, which is a different scale to the data
#                 MSE (mV^2).  Tune up if the model ignores the ODEs;
#                 tune down if training stalls or the loss oscillates.
#
#   lambda_ic   : Start around 1.0.  The IC loss is cheap and anchors
#                 trajectories to the correct starting point.
#
#   get_margin(start, end) :
#       Defaults (20 → 5 mV) are calibrated for the Regular Spiking
#       regime.  If training is unstable early on, increase `start`.
#       If the model struggles with near-spike accuracy, decrease
#       `end` (but not below ~3 mV, or autograd derivatives near the
#       spike peak will destabilise training).
# =====================================================================