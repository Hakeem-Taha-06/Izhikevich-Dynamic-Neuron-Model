# Hodgkin-Huxley Dynamic Neuron Model & ML-PINN Pipeline

Welcome to the **Hodgkin-Huxley Dynamic Neuron Model** repository. This project is a collaborative effort by a 10-person university engineering team to implement, solve, train, and evaluate the biophysical Hodgkin-Huxley neuron model using traditional numerical integration methods and Physics-Informed Neural Networks (PINNs).

---

## 📂 Project Directory Structure

```text
Izhikevich-Dynamic-Neuron-Model/
├── config.py                 # Master configuration file (constants & kinetics)
├── main.py                   # Main pipeline execution entry point
├── requirements.txt          # Python package dependencies
├── LICENSE                   # Project license
├── ROLES.md                  # Detailed team labor distribution and contracts
├── README.md                 # Project documentation (this file)
│
├── data/                     # Raw input data (ground truth CSVs)
│   └── .gitkeep
│
├── outputs/
│   ├── figures/              # All generated plots and visualizations
│   │   ├── FIGURES_README.md # Guide: what figures go here and naming rules
│   │   └── .gitkeep
│   └── models/               # Trained PyTorch model weights (.pt)
│       ├── MODELS_README.md  # Guide: how to save, load, and share weights
│       └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── numerical/
    │   ├── __init__.py
    │   ├── baseline_lsoda.py           # Role 3: Baseline LSODA Solver
    │   ├── baseline_lsoda_notes.md     # Role 3: Documentation
    │   ├── explicit_rk4.py             # Role 4: Explicit RK4 Solver
    │   ├── explicit_rk4_notes.md       # Role 4: Documentation
    │   ├── implicit_backward_euler.py  # Role 5: Implicit Backward Euler Solver
    │   └── implicit_backward_euler_notes.md  # Role 5: Documentation
    ├── ml_pinn/
    │   ├── __init__.py
    │   ├── architecture.py             # Role 6: PyTorch Neural Network
    │   ├── architecture_notes.md       # Role 6: Documentation
    │   ├── physics_loss.py             # Role 7: Physics-Informed Loss
    │   ├── physics_loss_notes.md       # Role 7: Documentation
    │   ├── train.py                    # Role 8: PINN Training Loop
    │   └── train_notes.md              # Role 8: Documentation
    └── evaluation/
        ├── __init__.py
        ├── evaluator_rmse.py           # Role 9: RMSE Evaluator
        ├── evaluator_rmse_notes.md     # Role 9: Documentation
        ├── bifurcation_physics.py      # Role 10: Bifurcation Analysis
        └── bifurcation_physics_notes.md  # Role 10: Documentation
```

---

## 🤝 Team Labor Distribution & Workspace Contracts

The project is structured across 10 distinct engineering roles. See [ROLES.md](ROLES.md) for full details.

| Phase | Role | Code File | Notes File |
|---|---|---|---|
| **Phase 0** | Role 1: Team Leader | `main.py` | — |
| | Role 2: Literature Reviewer | N/A (Paper Only) | — |
| **Phase 1** | Role 3: Math Modeler & Baseline | `src/numerical/baseline_lsoda.py` | `baseline_lsoda_notes.md` |
| **Phase 2** | Role 4: Explicit Solver (RK4) | `src/numerical/explicit_rk4.py` | `explicit_rk4_notes.md` |
| | Role 5: Implicit Solver (BE) | `src/numerical/implicit_backward_euler.py` | `implicit_backward_euler_notes.md` |
| **Phase 3** | Role 6: ML Architect | `src/ml_pinn/architecture.py` | `architecture_notes.md` |
| | Role 7: ML Loss Designer | `src/ml_pinn/physics_loss.py` | `physics_loss_notes.md` |
| | Role 8: ML Training Operator | `src/ml_pinn/train.py` | `train_notes.md` |
| **Phase 4** | Role 9: Evaluator | `src/evaluation/evaluator_rmse.py` | `evaluator_rmse_notes.md` |
| | Role 10: Crash Tester | `src/evaluation/bifurcation_physics.py` | `bifurcation_physics_notes.md` |

Each role has a `_notes.md` file alongside their `.py` file. **You must fill in your notes file** with the method documentation, tables, and analysis described inside it.

---

## 📐 Master Interface Contract

**ALL solvers must strictly adhere to the following output format:**

- **Return Type:** NumPy array of shape `(N, 5)`
- **Column Order:** `[Time, Voltage, m, h, n]`
  1. `Time` (ms)
  2. `Voltage` (mV)
  3. `m` (dimensionless sodium activation gate)
  4. `h` (dimensionless sodium inactivation gate)
  5. `n` (dimensionless potassium activation gate)

---

## 📁 Output Directories

| Directory | Purpose | Guide |
|-----------|---------|-------|
| `data/` | Ground truth CSV datasets | Produced by Role 9 |
| `outputs/figures/` | All generated plots (trajectories, bifurcation, loss curves) | See [FIGURES_README.md](outputs/figures/FIGURES_README.md) |
| `outputs/models/` | Trained PyTorch model weights (`.pt`) | See [MODELS_README.md](outputs/models/MODELS_README.md) |

---

## 🚀 Getting Started

### 1. Prerequisites
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
Once the team members have populated their respective workspace files:
```bash
python main.py
```
This will run the numerical simulations, train the PINN, compare all models, and save outputs to `data/` and `outputs/`.