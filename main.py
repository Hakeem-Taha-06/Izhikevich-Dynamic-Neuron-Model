"""Role 1: Master Execution Script — Izhikevich Neuron Model

Purpose
-------
Top-level orchestrator that runs all phases of the project in sequence:
  Phase 2A — Ground Truth generation (Role 4)
  Phase 2B — Numerical solvers     (Roles 5, 6, 7)
  Phase 3  — ML training           (Role 10, if available)
  Phase 4  — Evaluation & plots    (Role 11)

Usage
-----
    python main.py                  # full pipeline
    python main.py --skip-gt        # skip ground truth (use existing CSV)
    python main.py --skip-ml        # skip ML training
    python main.py --eval-only      # only run the evaluator

Model Reference
---------------
Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*.  MIT Press.

    C_m * dv/dt = k*(v - v_r)*(v - v_t) - u + I_ext
    du/dt       = a * { b*(v - v_r) - u }
    if v >= v_peak:  v <- c,  u <- u + d

Global output contract
----------------------
ALL solvers return np.ndarray of shape (N, 3) ordered as [Time, v, u].
"""

import sys
import os
import argparse
import time

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pathlib import Path


def _banner(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def run_ground_truth(output_path):
    from src.numerical.ground_truth_generator import generate_ground_truth
    _banner("Phase 2A — Ground Truth Generation (Role 4)")
    t0 = time.perf_counter()
    df = generate_ground_truth(output_path=output_path)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s — {len(df):,} rows written to {output_path}")
    return df


def run_numerical_solvers():
    from src.numerical.rk4              import solve_rk4
    from src.numerical.backward_euler   import solve_backward_euler
    from src.numerical.adams_bashforth2 import solve_adams_bashforth2

    _banner("Phase 2B — Numerical Solvers (Roles 5, 6, 7)")

    results = {}
    for name, fn in [
        ('RK4',             solve_rk4),
        ('Backward Euler',  solve_backward_euler),
        ('Adams-Bashforth 2', solve_adams_bashforth2),
    ]:
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        results[name] = out
        print(f"  {name:<20}: shape={out.shape}  time={elapsed:.4f}s")
        if out.shape[1] != 3:
            raise ValueError(f"{name} output shape must be (N,3), got {out.shape}")
    return results


def run_ml(gt_csv_path):
    _banner("Phase 3 — ML Training (Role 10)")
    try:
        from src.ml_model.train import run_training
        run_training(gt_csv_path)
    except ImportError:
        print("  [SKIP] ML training module not yet implemented (Role 10).")
    except Exception as e:
        print(f"  [WARN] ML training failed: {e}")


def run_evaluation():
    from src.evaluation.evaluator import run_efficiency_analysis, print_master_table
    from src.evaluation.evaluator import plot_time_series, plot_phase_portrait, run_pattern_analysis

    _banner("Phase 4 — Evaluation & Analytics (Role 11)")
    gt, results = run_efficiency_analysis()
    print_master_table(results)
    plot_time_series(gt, results, label='regular_spiking')
    plot_phase_portrait(results, label='regular_spiking')
    run_pattern_analysis()
    print("\n  All figures saved to outputs/figures/")


def main():
    parser = argparse.ArgumentParser(description='Izhikevich Neuron Model — Master Pipeline')
    parser.add_argument('--skip-gt', action='store_true', help='Skip ground truth generation')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML training')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()

    gt_csv = Path(_ROOT) / 'data' / 'ground_truth.csv'

    if args.eval_only:
        run_evaluation()
        return

    if not args.skip_gt:
        run_ground_truth(gt_csv)
    else:
        print(f"[SKIP] Ground truth — using existing: {gt_csv}")

    run_numerical_solvers()

    if not args.skip_ml:
        run_ml(gt_csv)
    else:
        print("[SKIP] ML training.")

    run_evaluation()

    _banner("Pipeline Complete")


if __name__ == '__main__':
    main()
