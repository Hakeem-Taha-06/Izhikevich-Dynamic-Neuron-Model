# Project: Dynamic Neuron Model (Hodgkin-Huxley)
## Team Labor Distribution & Interface Contracts

**Deadline:** Saturday, June 13th, 2026, 7:00 AM.

**Master Interface Rule:** ALL numerical methods and AI models must output their simulation results as a NumPy array of shape `(N, 5)`, where the columns are strictly ordered as `[Time, Voltage, m, h, n]`.

**Documentation Rule:** Each role (3–10) has a `_notes.md` file next to their `.py` workspace. You **must** fill in this file with the tables, analysis, and method descriptions specified inside it.

**Output Directories:**
- `data/` — Ground truth CSVs
- `outputs/figures/` — All generated plots (see `FIGURES_README.md` inside)
- `outputs/models/` — Trained model weights (see `MODELS_README.md` inside)

---

### Phase 0: The Command Layer
#### **Role 1: Team Leader & Editor**
* **Objective:** Repository management, final pipeline integration, and crafting the IEEE Overleaf paper.
* **Inputs:** Raw drafts, metrics, phase plots, and literature summaries from all team members.
* **Outputs:** The final merged GitHub repository and the unified 4-page IEEE PDF.
* **Dependencies:** Receives deliverables from everyone. Enforces the deadline.

#### **Role 2: Literature Reviewer**
* **Objective:** Conduct academic research and contextualize the biological problem based purely on recent published work.
* **Inputs:** IEEE Xplore, PubMed, arXiv (Strictly 2022-2026 publications).
* **Outputs:** A rigorously cited Introduction and Literature Review section for the final IEEE paper. 
* **Dependencies:** Delivers text to Role 1. (Does not write technical code or graph internal results).

---

### Phase 1: Mathematical Foundation
#### **Role 3: Mathematical Modeler & Baseline Developer**
* **Objective:** Define the biological constants, implement the reference ODE solver (LSODA), and author the theoretical methodology.
* **Inputs:** Schiesser Chapter 4 reference text.
* **Outputs:** 1. Verified `config.py` containing standard biological constants.
    2. The functional `baseline_lsoda.py` script.
    3. The "Mathematical Modeling" draft explaining the ODEs for the paper.
* **Documentation:** Fill in `baseline_lsoda_notes.md` with the ODE formulation, constants verification table, and solver justification.
* **Dependencies:** Delivers verified `config.py` and the LSODA script to Role 9 (Evaluator). Delivers math draft to Role 1.

---

### Phase 2: Numerical Engineering
#### **Role 4: Explicit Solver Developer (RK4)**
* **Objective:** Implement the 4th-Order Runge-Kutta method and find its stability breaking point.
* **Inputs:** `config.py`.
* **Outputs:** 1. A working RK4 Python function.
    2. A text report detailing the explicit time/space complexity and the exact time-step ($h$) where the method crashes (NaN errors).
* **Documentation:** Fill in `explicit_rk4_notes.md` with the complexity analysis table and stability breakdown report.
* **Dependencies:** Delivers function to Roles 9 & 10. Delivers complexity notes to Role 1.

#### **Role 5: Implicit Solver Developer (Backward Euler)**
* **Objective:** Implement the unconditionally stable Backward Euler method using `scipy.optimize.fsolve` for the internal root-finding loop.
* **Inputs:** `config.py`.
* **Outputs:** 1. A working Backward Euler Python function.
    2. A text report detailing the implicit time/space complexity.
* **Documentation:** Fill in `implicit_backward_euler_notes.md` with the complexity analysis and unconditional stability proof.
* **Dependencies:** Delivers function to Roles 9 & 10. Delivers complexity notes to Role 1.

---

### Phase 3: Machine Learning (PINN)
#### **Role 6: ML Architect**
* **Objective:** Design the PyTorch neural network architecture.
* **Inputs:** Expected input/output shapes (Time in $\rightarrow [V, m, h, n]$ out).
* **Outputs:** The `nn.Module` class utilizing differentiable activation functions (e.g., `Tanh`).
* **Documentation:** Fill in `architecture_notes.md` with the architecture summary table, layer diagram, and design justification.
* **Dependencies:** Delivers the model architecture to Roles 7 & 8.

#### **Role 7: ML Loss Function Designer**
* **Objective:** Encode the Hodgkin-Huxley ODEs into a custom PyTorch Autograd loss function.
* **Inputs:** Model architecture (Role 6) and `config.py` constants (Role 3).
* **Outputs:** A functional `physics_loss(model, t_collocation)` Python method that penalizes the network for violating the ODEs.
* **Documentation:** Fill in `physics_loss_notes.md` with the residual equations, autograd explanation, and config constants verification.
* **Dependencies:** Must strictly mirror Role 3's constants. Delivers loss function to Role 8.

#### **Role 8: ML Training Loop Operator**
* **Objective:** Train the PINN to convergence and export the final AI-predicted trajectories.
* **Inputs:** Ground Truth data (from Role 9), Model (Role 6), Loss Function (Role 7).
* **Outputs:** 1. The fully trained PyTorch model weights (`.pt`) saved to `outputs/models/`.
    2. The AI's predicted trajectory array.
* **Documentation:** Fill in `train_notes.md` with the hyperparameter table, convergence report, and loss curve.
* **Dependencies:** Delivers predicted array to Roles 9 & 10 for evaluation.

---

### Phase 4: Evaluation & Testing
#### **Role 9: The Evaluator (Trajectory Accuracy & Speed)**
* **Objective:** Generate the Ground Truth data, then measure and benchmark all team models.
* **Inputs:** `config.py` and `baseline_lsoda.py` (from Role 3) + predicted trajectories from Roles 4, 5, and 8.
* **Outputs:** 1. The Ground Truth dataset (`.csv`) saved to `data/`.
    2. The master Evaluation Table (RMSE and Execution Time for all methods).
    3. Final Result Graphs saved to `outputs/figures/`.
* **Documentation:** Fill in `evaluator_rmse_notes.md` with the RMSE metrics table, timing methodology, and plot descriptions.
* **Dependencies:** Delivers the CSV to Role 8 for training. Delivers final graphs and tables to Role 1.

#### **Role 10: Physics Coherency Analyst (The Crash Tester)**
* **Objective:** Calculate the exact $I_{ext}$ Bifurcation Threshold for every method to prove physical validity.
* **Inputs:** The functional solver scripts/models from Roles 3, 4, 5, and 8.
* **Outputs:** A Bifurcation Diagram saved to `outputs/figures/` showing the exact current threshold where each model transitions from a resting state to continuous spiking.
* **Documentation:** Fill in `bifurcation_physics_notes.md` with the sweep configuration, threshold results table, and physical validity conclusions.
* **Dependencies:** Imports solvers from Roles 3, 4, 5, and 8. Delivers the master bifurcation graph to Role 1.