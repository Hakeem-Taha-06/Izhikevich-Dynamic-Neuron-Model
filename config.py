"""
config.py
Master Configuration File for the Izhikevich Neuron Model.
Owned by Role 3 (Mathematical Modeler).

Model Reference
---------------
Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience:
The Geometry of Excitability and Bursting*. MIT Press.

Governing equations (2007 generalized biophysical form):

    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a*{ b*(v - v_r) - u }

After-spike reset (discrete):
    if v >= v_peak:  v <- c,  u <- u + d

WARNING TO ALL DEVELOPERS: The discrete reset condition (v >= v_peak)
MUST be handled inside your numerical loops.  Do not attempt to
calculate derivatives across the jump boundary.
"""
import numpy as np

# ==========================================
# 1. IZHIKEVICH PARAMETERS (Regular Spiking Default)
# ==========================================
# Role 11 will override these during biological pattern analysis testing.

# --- 2007 Biophysical Parameters ---
C_m = 100.0     # Membrane capacitance (pF)
k = 0.7         # Scaling constant (nS/mV)
v_r = -60.0     # Resting membrane potential (mV)
v_t = -40.0     # Instantaneous threshold potential (mV)

# --- Dimensionless / Reset Parameters ---
a = 0.03        # Recovery time-scale (1/ms)
b = -2.0        # Recovery sensitivity (nS)
c = -50.0       # After-spike reset of v (mV)
d = 100.0       # After-spike reset increment of u (pA)

# --- Spike Detection ---
v_peak = 35.0   # Spike cutoff / peak voltage (mV)

# ==========================================
# 2. DEFAULT INITIAL CONDITIONS
# ==========================================
# [Membrane Potential v (mV), Recovery Variable u (pA)]
V_0 = -60.0
U_0 = b * (V_0 - v_r)  # Steady-state formulation for 2007 model

INITIAL_STATE = np.array([V_0, U_0])

# ==========================================
# 3. SIMULATION SETTINGS
# ==========================================
T_START = 0.0
T_END = 100.0
DT_EVAL = 0.01    # Required resolution for the Ground Truth dataset

# Default External Current (pA)
I_EXT_DEFAULT = 500.0

# ==========================================
# 4. SHARED ODE FUNCTIONS
# ==========================================
def dv_dt(v, u, I_ext):
    """Calculates the derivative of the membrane potential.

    Implements:  dv/dt = [ k*(v - v_r)*(v - v_t) - u + I_ext ] / C_m
    """
    return (k * (v - v_r) * (v - v_t) - u + I_ext) / C_m


def du_dt(v, u):
    """Calculates the derivative of the recovery variable.

    Implements:  du/dt = a * { b*(v - v_r) - u }
    """
    return a * (b * (v - v_r) - u)