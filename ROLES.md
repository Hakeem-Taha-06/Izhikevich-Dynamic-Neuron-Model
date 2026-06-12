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

#### **Role 4: Lead Data Engineer (Training & Evaluation)**
* **Objective:** You own the entire data generation pipeline. Your task is to produce mathematically flawless simulations of the **2007 Generalized Izhikevich Model**. Thanks to the adoption of the biological step-current protocol, you are responsible for generating a highly efficient, lean dataset for the Neural Network, alongside a specific edge-case dataset for final solver evaluation.
* **Workspace:** `/src/numerical/ground_truth_generator.py`
* **Deliverables:** 1. `/data/ground_truth_train.csv` (Target size: 400,000 rows / ~10 MB)
  2. `/data/ground_truth_eval.csv` (Target size: 30,000 rows / < 1 MB)

---

### **Part 1: The Core Physics Engine (Universal Rules)**
These physical and mathematical constraints apply to **every** simulation you run, regardless of the dataset.

**1. Segmented Integration (The Discrete Reset)**
The 2007 Izhikevich model dictates that when the membrane potential ($v$) hits $v_{peak}$ (35.0 mV), it instantly resets to $c$, and the recovery variable ($w$) jumps by $d$. 
* Continuous industrial solvers cannot handle this teleportation. You must use an **event tracker** to halt the solver the exact microsecond $v$ reaches 35.0 mV. 
* Manually apply the mathematical reset to the state variables, log the discrete jump, and initialize a *new* solver run from that exact timestamp to continue.

**2. Time-Dependent Current ($I_{ext}$)**
To match the published reference literature, the neuron must start perfectly at rest. 
* You must implement a Heaviside step function for the injected current.
* **Logic:** For $t < 100.0$ ms, current is strictly 0.0 pA. For $t \ge 100.0$ ms, current becomes the active $I_{ext}$ target.

**3. Solver & Resolution Constraints**
* **Solver:** Strictly use the **Radau** method. Set relative tolerance (`rtol`) to 1e-6 and absolute tolerance (`atol`) to 1e-9.
* **Time Grid:** All simulations run from $T_{start} = 0.0$ to $T_{end} = 1000.0$ ms. 
* **Resolution:** You must step evaluate at exactly $dt = 0.1$ ms (yielding 10,000 steps per simulation).

---

### **Part 2: Deliverable A (The ML Training Dataset)**
Because the current is 0.0 for the first 100 ms, all initial conditions will naturally converge to the resting state. Therefore, you do not need to sweep initial conditions. You only sweep the current.

* **Fixed Initial Conditions:** $V_0 = -60.0$ mV, $W_0 = 0.0$ pA.
* **Fixed Biological Parameters:** $a = 0.03$, $b = -2.0$, $c = -50.0$, $d = 100.0$.
* **The Iteration Grid (40 unique runs):** Use `numpy.linspace(..., num=40)` to sweep:
  * **Current Magnitude ($I_{ext}$):** Sweep 0.0 pA to 500.0 pA.

---

### **Part 3: Deliverable B (The Physics Evaluation Dataset)**
This dataset mutates the biological parameters to test extreme edge-cases for the final IEEE paper.

* **Fixed Initial Conditions:** $V_0 = -60.0$ mV, $W_0 = 0.0$ pA.
* **The Three Test Cases:**
  1. **Regular Spiking (RS):** $a=0.03$, $b=-2.0$, $c=-50.0$, $d=100.0$ | Current: 70.0 pA
  2. **Intrinsically Bursting (IB):** $a=0.02$, $b=-2.0$, $c=-55.0$, $d=150.0$ | Current: 350.0 pA
  3. **Chattering (CH):** $a=0.02$, $b=-2.0$, $c=-50.0$, $d=20.0$ | Current: 300.0 pA

---

### **Part 4: Output Format**
Both CSV files must contain columns strictly ordered as: 
`[Time, I_ext, V0, W0, a, b, c, d, v, w]`
*(Note: Including all parameters ensures the ML team and Evaluators can programmatically filter the data during training and plotting).*

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