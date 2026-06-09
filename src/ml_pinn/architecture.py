import torch
import torch.nn as nn

# Config variables (if needed to import):
# No external biological constants are strictly required for the model structure itself,
# but layer dimensions can be parameterized or defined here.

class HodgkinHuxleyPINN(nn.Module):
    """
    Role 6: ML Architect
    
    Objective:
    ----------
    Defines the PyTorch neural network architecture (subclassing nn.Module) 
    used to approximate the trajectories of the Hodgkin-Huxley system. 
    The network must use differentiable activation functions (e.g., Tanh or SiLU) 
    to support PyTorch Autograd differentiation with respect to time `t`.
    
    Config Variables to Import:
    --------------------------
    - Typically no physical parameters are imported directly here (they are handled 
      in the loss function). Model parameters (input_dim, hidden_dim, layers) 
      may be imported or defined as defaults.

    Input Parameters:
    -----------------
    - t : torch.Tensor
        Collocation and/or boundary time points tensor of shape (N, 1),
        representing the independent variable (time, t).

    Output:
    -------
    - state_pred : torch.Tensor
        Predicted state variables tensor of shape (N, 4).
        The columns MUST be strictly ordered as:
        [Voltage, m, h, n]
        - Column 0: predicted membrane voltage V(t)
        - Column 1: predicted gating variable m(t)
        - Column 2: predicted gating variable h(t)
        - Column 3: predicted gating variable n(t)
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, output_dim=4):
        """
        Initializes layers for the deep neural network.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input layer (1 for Time t).
        hidden_dim : int
            Number of hidden units per layer.
        num_layers : int
            Number of hidden layers.
        output_dim : int
            Dimension of the output layer (4 for [V, m, h, n]).
        """
        super(HodgkinHuxleyPINN, self).__init__()
        pass

    def forward(self, t):
        """
        Performs the forward pass of the neural network.
        
        Parameters:
        -----------
        t : torch.Tensor
            Time points tensor of shape (N, 1).
            
        Returns:
        --------
        torch.Tensor
            Predicted states tensor of shape (N, 4).
        """
        pass
