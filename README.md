# Izhikevich Dynamic Neuron Model & ML-PINN Pipeline

Welcome to the **Izhikevich Dynamic Neuron Model** repository. This project is a collaborative effort by an 11-person university engineering team to implement, solve, train, and evaluate the biophysical Izhikevich neuron model using traditional numerical integration methods and Physics-Informed Neural Networks (PINNs).

---

## 📂 Project Directory Structure

```text
Izhikevich-Dynamic-Neuron-Model/
├── config.py                 # Master configuration file (constants & kinetics)
├── main.py                   # Role 1: Main pipeline execution entry point
├── requirements.txt          # Python package dependencies
├── LICENSE                   # Project license
├── ROLES.md                  # Detailed team labor distribution and contracts
├── README.md                 # Project documentation (this file)
│
├── data/                     # Raw input data
│   └── README.md             # Role 4: saves ground_truth.csv here
│
├── outputs/
│   ├── figures/              # Role 11: saves graphs here
│   └── models/               # Role 10: saves .pt weights here
│
└── src/
    ├── numerical/
    │   ├── __init__.py
    │   ├── ground_truth_generator.py # Role 4's workspace
    │   ├── rk4.py           # Role 5's workspace
    │   ├── rk4_notes.md
    │   ├── backward_euler.py         # Role 6's workspace
    │   ├── backward_euler_notes.md
    │   ├── adams_bashforth2.py        # Role 7's workspace
    │   └── adams_bashforth2_notes.md
    ├── ml_model/
    │   ├── __init__.py
    │   ├── architecture.py           # Role 8's workspace
    │   ├── physics_loss.py           # Role 9's workspace
    │   ├── train.py                  # Role 10's workspace
    │   └── ml_notes.md
    └── evaluation/
        ├── __init__.py
        ├── evaluator.py              # Role 11's workspace
        └── eval_notes.md
```

---

## 🤝 Team Labor Distribution & Workspace Contracts

The project is structured across 11 distinct engineering roles. See [ROLES.md](ROLES.md) for full details.

| Phase | Role | Workspace |
|---|---|---|
| **Phase 0** | Role 1: Team Leader | `main.py` |
| | Role 2: Literature Reviewer | N/A (Paper Only) |
| | Role 3: Math Modeler | `src/theory/math_theory_notes.md` |
| **Phase 1** | Role 4: Data Engineer | `src/numerical/ground_truth_generator.py` |
| | Role 5: Explicit Solver (RK4) | `src/numerical/explicit_rk4.py` |
| | Role 6: Implicit Solver (BE) | `src/numerical/backward_euler.py` |
| | Role 7: Multi-Step (AB2) | `src/numerical/adams_bashforth.py` |
| **Phase 2** | Role 8: ML Architect | `src/ml_model/architecture.py` |
| | Role 9: ML Loss Designer | `src/ml_model/physics_loss.py` |
| | Role 10: ML Training Operator | `src/ml_model/train.py` |
| **Phase 3** | Role 11: Master Evaluator | `src/evaluation/evaluator.py` |

Each role has a `.md` notes file alongside their `.py` file. **You must fill in your notes file** with the method documentation, tables, and analysis described inside it.

---

## 📐 Master Interface Contract

**ALL numerical methods and AI models must strictly adhere to the following output format:**

- **Return Type:** NumPy array of shape `(N, 3)`
- **Column Order:** `[Time, v, u]`
  1. `Time` (ms)
  2. `v` (membrane potential in mV)
  3. `u` (recovery variable)

*Note: The discrete reset (If v >= 30 mV, then v <- c, u <- u + d) must be correctly enforced in all numerical methods except Role 4, which handles segmented integration.*