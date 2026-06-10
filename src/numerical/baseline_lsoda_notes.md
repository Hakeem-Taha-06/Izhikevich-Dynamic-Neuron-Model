# Role 3: Mathematical Modeler & Baseline Developer — Documentation

## What You Must Document Here

### 1. Mathematical Formulation
- Write out the full Hodgkin-Huxley ODE system you are solving:
  - `dV/dt = ...`
  - `dm/dt = ...`
  - `dh/dt = ...`
  - `dn/dt = ...`
- Explain the biological meaning of each equation in 1–2 sentences.

### 2. Solver Description
- Describe what the LSODA algorithm is and why it was chosen as the baseline.
- Note that LSODA automatically switches between stiff (BDF) and non-stiff (Adams) methods.

### 3. Config Constants Verification Table
| Constant | Value | Unit | Source |
|----------|-------|------|--------|
| `G_NA`   | 120.0 | mS/cm² | Hodgkin & Huxley, 1952 |
| `G_K`    | 36.0  | mS/cm² | Hodgkin & Huxley, 1952 |
| ...      | ...   | ...  | ... |

Fill in the complete table to confirm all values match the original 1952 paper (or Schiesser Ch. 4).

### 4. Baseline Accuracy Justification
- Explain why the LSODA output is treated as "Ground Truth."
- Report the solver tolerance settings used (e.g., `rtol`, `atol`).

### 5. Sample Output Verification
- Paste or describe a brief sanity-check: does the membrane voltage show action potentials at `I_ext = 10.0 uA/cm²`?
- Include a screenshot or description of the expected waveform shape.

---

> **Deliver to:** Role 1 (Team Leader) — The "Mathematical Modeling" methodology section draft for the IEEE paper.  
> **Deliver to:** Role 9 (Evaluator) — The verified `config.py` and working `baseline_lsoda.py` script.
