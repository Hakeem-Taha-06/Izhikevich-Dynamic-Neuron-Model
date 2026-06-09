"""
config.py
Master Configuration File for the Dynamic Neuron Model (Hodgkin-Huxley).
ALL team members must import their constants from this file. DO NOT hardcode these values.
"""
import numpy as np

# ==========================================
# 1. BIOLOGICAL CONSTANTS (Hodgkin-Huxley)
# ==========================================
# Maximal conductances (mS/cm^2)
G_NA = 120.0  # Sodium
G_K = 36.0    # Potassium
G_L = 0.3     # Leak

# Reversal potentials (mV)
E_NA = 50.0   # Sodium
E_K = -77.0   # Potassium
E_L = -54.4   # Leak

# Membrane Capacitance (uF/cm^2)
C_M = 1.0

# ==========================================
# 2. DEFAULT INITIAL CONDITIONS (Resting State)
# ==========================================
# [Voltage (mV), m (dim-less), h (dim-less), n (dim-less)]
V_0 = -65.0
M_0 = 0.052
H_0 = 0.596
N_0 = 0.317

INITIAL_STATE = np.array([V_0, M_0, H_0, N_0])

# ==========================================
# 3. SIMULATION PARAMETERS
# ==========================================
T_START = 0.0
T_END = 100.0     # Simulate for 100 milliseconds
DT_EVAL = 0.01    # Output resolution (ensure all arrays match length)

# Default External Current injection (uA/cm^2)
# Note: Role 10 will override this variable during the Bifurcation sweep.
I_EXT_DEFAULT = 10.0 

# ==========================================
# 4. SHARED GATING KINETICS FUNCTIONS
# ==========================================
# Provided here so Roles 3, 4, 5, and 7 use the exact same mathematical forms.

def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)