"""
Numerical solvers subpackage.
Provides reference solvers for the Izhikevich model (2007 generalized form):
    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a*{ b*(v - v_r) - u }

Modules:
- Ground truth generator
- Explicit 4th-Order Runge-Kutta (RK4) solver
- Backward Euler solver
- Adams-Bashforth 2 solver
"""
