import torch

# Imports from config.py:
# from config import (
#     C_M, G_NA, G_K, G_L, E_NA, E_K, E_L,
#     alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
# )
# Note: For PyTorch autograd computations, the numpy-based gating functions 
# from config.py must be implemented using PyTorch operators (e.g., torch.exp) 
# to allow backward pass propagation.

def compute_physics_loss(model, t_collocation, I_ext=10.0):
    """
    Role 7: ML Loss Function Designer
    
    Objective:
    ----------
    Encodes the Hodgkin-Huxley system of Ordinary Differential Equations (ODEs) 
    into a custom physics-informed loss function using PyTorch Autograd. 
    It computes the residuals of the ODEs at the designated collocation time points 
    and penalizes the model for violating the physical dynamics of the neuron.

    Config Variables to Import:
    --------------------------
    - C_M: float, membrane capacitance (e.g., 1.0 uF/cm^2)
    - G_NA, G_K, G_L: floats, maximal conductances for Na, K, and Leak channels
    - E_NA, E_K, E_L: floats, reversal potentials for Na, K, and Leak channels
    - alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n: gating rate equations
      (Note: These must be evaluated using torch tensor operations inside the loss).

    Input Parameters:
    -----------------
    - model : nn.Module
        The PyTorch neural network predicting [V, m, h, n] from time input t (from Role 6).
    - t_collocation : torch.Tensor
        A tensor of shape (N, 1) containing time points at which the physics residuals 
        are evaluated. Must have `requires_grad=True` enabled.
    - I_ext : float or torch.Tensor, optional
        External current injection. Can be a constant scalar (default: 10.0 uA/cm^2) 
        or a tensor of shape (N, 1) mapping time-dependent current.

    Output:
    -------
    - loss : torch.Tensor
        A scalar PyTorch tensor (shape: ()) representing the mean squared residual of the
        Hodgkin-Huxley differential equations:
        loss = Mean( Residual_V^2 + Residual_m^2 + Residual_h^2 + Residual_n^2 )
        
        Where the residuals are defined as:
        - Residual_V = C_M * dV/dt - (I_ext - I_Na - I_K - I_L)
          where:
            I_Na = G_NA * m^3 * h * (V - E_NA)
            I_K = G_K * n^4 * (V - E_K)
            I_L = G_L * (V - E_L)
        - Residual_y = dy/dt - (alpha_y(V) * (1 - y) - beta_y(V) * y) for y in {m, h, n}
    """
    pass
