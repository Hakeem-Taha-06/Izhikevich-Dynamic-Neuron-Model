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
├── data/
│   └── .gitkeep              # Keeps the folder in version control
└── src/
    ├── __init__.py           # Package initializer
    ├── numerical/
    │   ├── __init__.py       # Numerical solvers package initializer
    │   ├── baseline_lsoda.py # Role 3: Baseline LSODA Solver (SciPy solve_ivp)
    │   ├── explicit_rk4.py   # Role 4: Explicit 4th-Order Runge-Kutta Solver
    │   └── implicit_backward_euler.py # Role 5: Implicit Backward Euler Solver
    ├── ml_pinn/
    │   ├── __init__.py       # PINN package initializer
    │   ├── architecture.py   # Role 6: PyTorch Neural Network Architecture
    │   ├── physics_loss.py   # Role 7: Custom PyTorch Physics-Informed Loss
    │   └── train.py          # Role 8: PINN Training Loop & Predictor
    └── evaluation/
        ├── __init__.py       # Evaluation package initializer
        ├── evaluator_rmse.py # Role 9: Ground Truth Generator & RMSE Evaluator
        └── bifurcation_physics.py # Role 10: Bifurcation Threshold sweeping
```

---

## 🤝 Team Labor Distribution & Workspace Contracts

The project is structured across 10 distinct engineering roles:

| Phase / Role | Assigned Team Member Role | Primary Workspace File | Key Responsibility |
|---|---|---|---|
| **Phase 0** | **Role 1**: Team Leader & Editor | `main.py` | Repository management, final pipeline integration, paper compile. |
| | **Role 2**: Literature Reviewer | N/A (Paper Only) | Academic research and biological context writing. |
| **Phase 1** | **Role 3**: Math Modeler & Baseline Dev | `src/numerical/baseline_lsoda.py` | Biological constants, gating rates, LSODA baseline. |
| **Phase 2** | **Role 4**: Explicit Solver Dev | `src/numerical/explicit_rk4.py` | RK4 integration method & stability breakdown analysis. |
| | **Role 5**: Implicit Solver Dev | `src/numerical/implicit_backward_euler.py` | Backward Euler method using root-finding loops. |
| **Phase 3** | **Role 6**: ML Architect | `src/ml_pinn/architecture.py` | Deep neural network architecture using differentiable activations. |
| | **Role 7**: ML Loss Designer | `src/ml_pinn/physics_loss.py` | Custom PyTorch Autograd physics-informed loss function. |
| | **Role 8**: ML Training Operator | `src/ml_pinn/train.py` | Training loop, convergence tuning, trajectory export. |
| **Phase 4** | **Role 9**: Master Evaluator | `src/evaluation/evaluator_rmse.py` | Ground truth dataset generation, RMSE, and runtime benchmarking. |
| | **Role 10**: Physics Coherency Analyst | `src/evaluation/bifurcation_physics.py` | $I_{ext}$ bifurcation sweep and crash testing. |

---

## 📐 Master Interface Contract

To ensure seamless integration across the numerical and machine learning phases, **all solvers must strictly adhere to the following output format**:

- **Return Type:** NumPy array of shape `(N, 5)`
- **Column Order:** `[Time, Voltage, m, h, n]`
  1. `Time` (ms)
  2. `Voltage` (mV)
  3. `m` (dimensionless sodium activation gate)
  4. `h` (dimensionless sodium inactivation gate)
  5. `n` (dimensionless potassium activation gate)

---

## 🚀 Getting Started

### 1. Prerequisites
Create a virtual environment and install the required dependencies:
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
Once the team members have populated their respective workspace files with their solver logic, you can execute the entire pipeline using:
```bash
python main.py
```
This will run the numerical simulations, train the PINN, compare all models, and save generated data/graphs to the `data/` folder.