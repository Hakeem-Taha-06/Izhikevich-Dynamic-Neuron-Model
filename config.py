"""
config.py
Master Configuration File for the Izhikevich Neuron Model.
Owned by Role 3 (Mathematical Modeler).

WARNING TO ALL DEVELOPERS: The discrete reset condition (v >= 30) MUST be 
handled inside your numerical loops. Do not attempt to calculate derivatives 
across the jump boundary.
"""
import numpy as np

# ==========================================
# 1. IZHIKEVICH PARAMETERS (Regular Spiking Default)
# ==========================================
# Role 11 will override these during biological pattern analysis testing.
A = 0.02
B = 0.2
C = -65.0
D = 8.0

# The discrete reset threshold (mV)
V_THRESH = 30.0

# ==========================================
# 2. DEFAULT INITIAL CONDITIONS
# ==========================================
# [Membrane Potential v (mV), Recovery Variable u]
V_0 = -65.0
U_0 = B * V_0  # Steady state formulation

INITIAL_STATE = np.array([V_0, U_0])

# ==========================================
# 3. SIMULATION SETTINGS
# ==========================================
T_START = 0.0
T_END = 100.0     
DT_EVAL = 0.01    # Required resolution for the Ground Truth dataset

# Default External Current
I_EXT_DEFAULT = 10.0 

# ==========================================
# 4. SHARED ODE FUNCTIONS
# ==========================================
def dv_dt(v, u, I_ext):
    """Calculates the derivative of the membrane potential."""
    return 0.04 * v**2 + 5.0 * v + 140.0 - u + I_ext

def du_dt(v, u):
    """Calculates the derivative of the recovery variable."""
    return A * (B * v - u)