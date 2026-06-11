"""Role 11: Master Evaluator & Analyst — Izhikevich Neuron Model

Purpose
-------
Prove which numerical method is most accurate and efficient, then
visualize the biological firing patterns of the Izhikevich neuron.

Inputs
------
- Output arrays from Roles 4 (Ground Truth), 5 (RK4), 6 (Backward Euler),
  7 (Adams-Bashforth 2), and 10 (ML Predictor, optional).

Deliverables (per ROLES.md)
----------------------------
1. Master comparison table: RMSE + Wall-Clock time for every method.
2. Phase portraits (v vs u) for Regular Spiking and alternative regimes.
3. Time-series plots (Time vs v) for all methods overlaid.
4. All figures saved as PNG to /outputs/figures/.
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering for file output
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    INITIAL_STATE, T_START, T_END, DT_EVAL, I_EXT_DEFAULT,
    C_m, k, v_r, v_t, v_peak, a, b, c, d,
)
from src.numerical.ground_truth_generator import _simulate_one as _gt_simulate
from src.numerical.rk4            import solve_rk4
from src.numerical.backward_euler import solve_backward_euler
from src.numerical.adams_bashforth2 import solve_adams_bashforth2

# Output directory
_FIGURES_DIR = Path(_ROOT) / 'outputs' / 'figures'
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: compute ground truth via Role 4 segmented integrator
# ---------------------------------------------------------------------------

def _ground_truth_array():
    """Return ground truth as (N, 3) array [Time, v, u]."""
    V0, U0 = INITIAL_STATE
    t, v, u = _gt_simulate(V0, U0, I_EXT_DEFAULT)
    return np.column_stack([t, v, u])


# ---------------------------------------------------------------------------
# 1. Efficiency Analysis
# ---------------------------------------------------------------------------

def run_efficiency_analysis(n_runs=3):
    """Benchmark all solvers and compute RMSE vs ground truth.

    Returns
    -------
    results : dict
        Keys are method names; values are dicts with keys:
        'rmse_v', 'rmse_u', 'wall_time_s', 'output'.
    """
    print("Generating ground truth (Radau segmented)...")
    t0 = time.perf_counter()
    gt = _ground_truth_array()
    gt_time = time.perf_counter() - t0
    print(f"  Ground truth: {gt.shape[0]} points in {gt_time:.3f}s")

    methods = {
        'Ground Truth (Radau)' : (lambda: _ground_truth_array(),    gt_time),
        'RK4'                  : (lambda: solve_rk4(),              None),
        'Backward Euler'       : (lambda: solve_backward_euler(),   None),
        'Adams-Bashforth 2'    : (lambda: solve_adams_bashforth2(), None),
    }

    results = {}
    for name, (fn, preset_time) in methods.items():
        print(f"\nBenchmarking: {name}")
        times = []
        out   = None
        for _ in range(n_runs):
            t0  = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
        wall = preset_time if preset_time is not None else np.mean(times)

        # Align arrays by interpolating onto ground-truth time axis
        if name != 'Ground Truth (Radau)':
            v_interp = np.interp(gt[:, 0], out[:, 0], out[:, 1])
            u_interp = np.interp(gt[:, 0], out[:, 0], out[:, 2])
        else:
            v_interp = out[:, 1]
            u_interp = out[:, 2]

        rmse_v = float(np.sqrt(np.mean((v_interp - gt[:, 1])**2)))
        rmse_u = float(np.sqrt(np.mean((u_interp - gt[:, 2])**2)))

        results[name] = {
            'rmse_v'      : rmse_v,
            'rmse_u'      : rmse_u,
            'wall_time_s' : wall,
            'output'      : out,
        }
        print(f"  RMSE v={rmse_v:.4f} mV | RMSE u={rmse_u:.4f} pA | time={wall:.4f}s")

    return gt, results


def print_master_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print(f"{'Method':<25} {'RMSE v (mV)':>12} {'RMSE u (pA)':>12} {'Wall-Clock (s)':>15}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<25} {r['rmse_v']:>12.4f} {r['rmse_u']:>12.4f} {r['wall_time_s']:>15.4f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 2. Plotting helpers
# ---------------------------------------------------------------------------

def plot_time_series(gt, results, label='regular_spiking'):
    """Overlay time-series (Time vs v) for all methods."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gt[:, 0], gt[:, 1], 'k-', linewidth=1.5, label='Ground Truth (Radau)', zorder=5)

    styles = [('tab:blue', '--'), ('tab:orange', '-.'), ('tab:green', ':')]
    method_names = [n for n in results if n != 'Ground Truth (Radau)']
    for (color, ls), name in zip(styles, method_names):
        out = results[name]['output']
        ax.plot(out[:, 0], out[:, 1], color=color, linestyle=ls,
                linewidth=1.0, label=name, alpha=0.85)

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Membrane Potential v (mV)', fontsize=12)
    ax.set_title(f'Izhikevich Neuron — Time Series ({label})', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = _FIGURES_DIR / f'time_series_{label}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_phase_portrait(results, label='regular_spiking'):
    """Plot phase portrait (v vs u) for all methods."""
    fig, ax = plt.subplots(figsize=(8, 6))

    styles = [('k', '-', 'Ground Truth (Radau)', 2.0),
              ('tab:blue', '--', 'RK4', 1.0),
              ('tab:orange', '-.', 'Backward Euler', 1.0),
              ('tab:green', ':', 'Adams-Bashforth 2', 1.0)]

    for color, ls, name, lw in styles:
        if name in results:
            out = results[name]['output']
            ax.plot(out[:, 1], out[:, 2], color=color, linestyle=ls,
                    linewidth=lw, label=name, alpha=0.85)

    ax.set_xlabel('Membrane Potential v (mV)', fontsize=12)
    ax.set_ylabel('Recovery Variable u (pA)', fontsize=12)
    ax.set_title(f'Phase Portrait — Izhikevich ({label})', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = _FIGURES_DIR / f'phase_portrait_{label}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Biological pattern testing
# ---------------------------------------------------------------------------

def run_pattern_analysis():
    """Test Chattering regime by overriding config parameters locally."""
    import config as cfg

    # Save originals
    orig = {attr: getattr(cfg, attr) for attr in ['a', 'b', 'c', 'd', 'C_m', 'k']}

    # Chattering (CH) parameters (Izhikevich 2007, Table 8.1)
    cfg.C_m    = 50.0
    cfg.k      = 1.5
    cfg.a      = 0.03
    cfg.b      = 1.0
    cfg.c      = -40.0
    cfg.d      = 150.0

    # Reload module-level constants used in the solvers
    for mod_name in ['src.numerical.rk4',
                     'src.numerical.backward_euler',
                     'src.numerical.adams_bashforth2']:
        mod = sys.modules.get(mod_name)
        if mod:
            for attr in ['C_m', 'k', 'a', 'b', 'c', 'd']:
                if hasattr(mod, attr):
                    setattr(mod, attr, getattr(cfg, attr))

    print("\n--- Chattering Pattern ---")
    ch_results = {
        'Ground Truth (Radau)' : {'output': _ground_truth_array()},
        'RK4'                  : {'output': solve_rk4()},
        'Backward Euler'       : {'output': solve_backward_euler()},
        'Adams-Bashforth 2'    : {'output': solve_adams_bashforth2()},
    }
    plot_time_series(ch_results['Ground Truth (Radau)']['output'],
                     ch_results, label='chattering')
    plot_phase_portrait(ch_results, label='chattering')

    # Restore originals
    for attr, val in orig.items():
        setattr(cfg, attr, val)
    for mod_name in ['src.numerical.rk4',
                     'src.numerical.backward_euler',
                     'src.numerical.adams_bashforth2']:
        mod = sys.modules.get(mod_name)
        if mod:
            for attr, val in orig.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, val)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Role 11 — Master Evaluator  (full analysis)")
    print("=" * 60)

    gt, results = run_efficiency_analysis()
    print_master_table(results)

    print("\n--- Generating plots (Regular Spiking) ---")
    plot_time_series(gt, results, label='regular_spiking')
    plot_phase_portrait(results, label='regular_spiking')

    print("\n--- Running Biological Pattern Analysis ---")
    run_pattern_analysis()

    print("\nEvaluation complete. All figures saved to outputs/figures/")