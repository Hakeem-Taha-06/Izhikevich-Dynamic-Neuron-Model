# Project: Izhikevich Dynamic Neuron Model
## Master Engineering Contract & Role Distribution

**Hard Deadline:** Saturday, June 13th, 2026, 7:00 AM.

**Internal Deadline:** Friday, June 12th, 2026, 10:00 PM.

**Model Reference:** Izhikevich, E. M. (2007). *Dynamical Systems in Neuroscience*. MIT Press.

**Governing Equations (2007 Generalized Biophysical Form):**

$$C_m \frac{dv}{dt} = k(v - v_r)(v - v_t) - w + I_{ext}$$

$$\frac{dw}{dt} = a\{ b(v - v_r) - w \}$$

**Master Interface Rule:** ALL numerical methods and AI models must output their simulation results as a NumPy array of shape `(N, 3)`, specifically ordered as `[Time, v, w]`. 

---

### ⚠️ THE GLOBAL CONSTRAINT: THE DISCRETE RESET ⚠️
The Izhikevich model is NOT a continuous curve. All coders (except Role 4) must manually write logic inside their loops to handle the voltage spike:
**If $v \ge v_{peak}$, then immediately set $v \leftarrow c$ and $w \leftarrow w + d$.**

---

### Phase 1: Command & Theory

#### **Role 1: Team Leader & Editor (Project Manager)**
* **Objective:** Ensure all moving parts fit together and write the final IEEE paper.
* **Inputs:** Markdown notes, metrics, and graphs from all other members.
* **Actionable Tasks:**
    1. Act as the GitHub gatekeeper. Approve all Pull Requests.
    2. Stitch the theoretical notes (Role 3), numerical notes (Roles 5-7), and ML notes (Roles 8-10) into the Methodology section.
    3. Write the Abstract, Conclusion, and format the IEEE Overleaf document.
* **Deliverable:** The final merged GitHub repository and the 4-page IEEE PDF.

#### **Role 2: Literature Reviewer**
* **Objective:** Contextualize the project using recent academic literature.
* **Inputs:** IEEE Xplore, PubMed, arXiv.
* **Actionable Tasks:**
    1. Research papers published strictly between **2022 and 2026**.
    2. Write the theoretical background explaining *why* the Izhikevich model is used (computational efficiency combined with biological accuracy compared to Hodgkin-Huxley).
* **Deliverable:** A rigorously cited text draft delivered directly to Role 1. (No coding required).

#### **Role 3: Mathematical Modeler**
* **Objective:** Own the pure mathematics and biological parameters of the project.
* **Inputs:** Reference text (Izhikevich 2007, Schiesser 2014) and biological literature.
* **Actionable Tasks:**
    1. Define and maintain the `config.py` file. Set the $C_m, k, v_r, v_t, v_{peak}, a, b, c, d$ parameters for the default "Regular Spiking" state.
    2. Write the "Mathematical Formulation" text explaining the ODEs, the nullclines, and the reset condition.
* **Deliverable:** The `config.py` file and `/src/theory/math_theory_notes.md`.

---

### Phase 2: 

#### A) Data Pipeline

#### **Role 4: Ground Truth & Data Engineer**
* **Objective:** You are the foundation of the Machine Learning phase. Your task is to generate a mathematically flawless, high-fidelity dataset containing ~4,000 unique neuron simulations. If your data is corrupted or inaccurate, the AI model (Roles 8, 9, 10) is mathematically doomed to fail.
* **Workspace:** `/src/numerical/ground_truth_generator.py`
* **Deliverable:** `/data/ground_truth.csv`

**1. The Engineering Trap: The Discrete Reset**
The Izhikevich model dictates that when the membrane potential ($v$) hits $v_{peak}$, it instantly resets to $c$, and the recovery variable ($w$) jumps by $d$. 
* **The Problem:** Continuous industrial solvers (like `solve_ivp`) do not understand teleportation. If you just run the solver for 100ms, it will mathematically overshoot the $v_{peak}$ threshold, ruining the spike timing and corrupting the dataset.
* **Your Required Architecture:** You must implement **Segmented Integration**. You are required to use an event tracker built into your solver to detect the exact microsecond $v$ reaches $v_{peak}$. When that event triggers, you must command the solver to completely halt. You will then manually apply the mathematical reset to the state variables, log the discrete jump in your data arrays, and initialize a *new* solver run from that exact timestamp to continue the simulation until you reach the 100ms target. 
* **Solver Constraints:** Because this is a stiff differential system, you are required to use a stiff solver method (e.g., `Radau` or `LSODA`). Set your relative tolerance to $10^{-6}$ and absolute tolerance to $10^{-9}$.

**2. The Iteration Grid (Data Volume)**
A neural network needs diverse data to learn the underlying physics. You must construct a nested loop system to simulate the neuron under varying conditions. Your grid must sweep through:
* **Current ($I_{ext}$):** From 0.0 to 500.0 pA in steps of 12.5 (40 variations).
* **Initial Voltage ($V_0$):** From -85.0 mV to -45.0 mV in steps of 2.0 (20 variations).
* **Initial Recovery ($W_0$):** Use an array of 5 variations around the steady state (e.g., based on $b \cdot (V_0 - v_r)$).
* **Target:** Exactly 4,000 distinct simulation runs.

**3. The Output Schema**
Your final output must be exported via Pandas to `/data/ground_truth.csv`. The time step must be strictly interpolated to `0.01` ms. The CSV must perfectly match this structure:

| Sim_ID | Time (ms) | I_ext (pA) | v (mV) | w (pA) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.00 | 250.0 | -60.000 | 0.000 |
| 1 | 0.01 | 250.0 | -59.821 | 0.002 |
| ... | ... | ... | ... | ... |

*(Note: `Sim_ID` must increment by 1 for each new combination of initial conditions in your nested loop).*

---

#### B) Numerical Engineering

#### **Role 5: Explicit Method Developer (RK4)**
* **Objective:** Implement the 4th-Order Runge-Kutta from scratch.
* **Inputs:** `config.py`.
* **Actionable Tasks:**
    1. Code the 4-stage RK4 logic. Apply the reset check *after* every single time-step.
    2. Find the exact time-step ($h$) where the method goes unstable and crashes (NaN). 
* **Deliverable:** `/src/numerical/rk4.py` and `rk4_notes.md` (detailing the stability limit and Big-O complexity).

#### **Role 6: Implicit Method Developer (Backward Euler)**
* **Objective:** Implement an unconditionally stable implicit solver.
* **Inputs:** `config.py`.
* **Actionable Tasks:**
    1. Use `scipy.optimize.fsolve` at every time-step to solve the non-linear algebraic equation caused by the implicit formulation.
    2. Ensure the root-finding algorithm does not break when the voltage crosses the $v_{peak}$ reset threshold.
* **Deliverable:** `/src/numerical/backward_euler.py` and `backward_euler_notes.md` (detailing root-finding performance).

#### **Role 7: Multi-Step Method Developer (Adams-Bashforth 2)**
* **Objective:** Implement the 2nd-Order AB2 method.
* **Inputs:** `config.py`.
* **Actionable Tasks:**
    1. Build the history array to predict the next step using past data.
    2. **Crucial Trap:** When the neuron spikes and resets, the "history" becomes invalid. You must write logic to flush the history array and restart the method (using Euler or RK2 for one step) before resuming AB2.
* **Deliverable:** `/src/numerical/adams_bashforth2.py` and `adams_bashforth2_notes.md` (explaining the history flush logic).

---

### Phase 3: Machine Learning (PINN/Supervised)

#### **Role 8: ML Architect**
* **Objective:** Build the PyTorch Neural Network skeleton.
* **Inputs:** Expected data shape.
* **Actionable Tasks:**
    1. Design an `nn.Module` that accepts `[Time, I_ext, V_0, W_0]` as inputs and outputs `[v, w]`.
    2. Select differentiable activation functions (e.g., Tanh or SiLU).
* **Deliverable:** `/src/ml_model/architecture.py`.

#### **Role 9: ML Loss Function Designer**
* **Objective:** Force the AI to obey physics.
* **Inputs:** ODEs from `config.py` and Architecture from Role 8.
* **Actionable Tasks:**
    1. Write the Autograd logic to calculate gradients of $v$ and $w$ with respect to Time.
    2. **Crucial Trap:** The jump at $v = v_{peak}$ will cause infinite gradients and crash the training. You must design a workaround (e.g., masking the loss exactly at the spike, or using piecewise continuous training). This is the hardest task on the team.
* **Deliverable:** `/src/ml_model/physics_loss.py`.

#### **Role 10: ML Training Loop Operator**
* **Objective:** Train the model and export the results.
* **Inputs:** `ground_truth.csv` (Role 4), Model (Role 8), Loss (Role 9).
* **Actionable Tasks:**
    1. Build the PyTorch DataLoader.
    2. Execute the training loop (epochs, batching, optimizer switching from Adam to L-BFGS).
    3. Feed a test input into the trained model to generate the final predicted trajectory.
* **Deliverable:** `/src/ml_model/train.py`, the saved `.pt` weights in `/outputs/models/`, and the final array to Role 11.

---

### Phase 4: Evaluation & Analytics

#### **Role 11: Master Evaluator & Analyst**
* **Objective:** Prove which method is best and visualize the biological behaviors.
* **Inputs:** Output arrays from Roles 4, 5, 6, 7, and 10.
* **Actionable Tasks:**
    1. **Efficiency Analysis:** Calculate the RMSE against the Ground Truth and log the execution time (Wall-Clock) for every method. Format this into a master table.
    2. **Biological Pattern Testing:** Change the $C_m, k, v_r, v_t, v_{peak}, a, b, c, d$ parameters in `config.py` to trigger different states (e.g., Regular Spiking vs. Chattering). Rerun all solvers. Generate Phase Portraits (plotting $v$ vs $w$) and Time-Series graphs (Time vs $v$) to prove the methods adapted correctly.
* **Deliverable:** `/src/evaluation/evaluator.py`, `eval_notes.md`, and all PNG/PDF graphs saved to `/outputs/figures/`.