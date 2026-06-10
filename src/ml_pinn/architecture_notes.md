# Role 6: ML Architect — Documentation

## What You Must Document Here

### 1. Network Architecture Summary
Describe the neural network design in detail:

| Property | Value |
|----------|-------|
| Framework | PyTorch |
| Input Dimension | 1 (time `t`) |
| Output Dimension | 4 (`V, m, h, n`) |
| Hidden Layers | ? |
| Neurons per Layer | ? |
| Activation Function | Tanh / SiLU / ? |
| Total Trainable Parameters | ? |

### 2. Architecture Diagram
Draw or describe the layer-by-layer flow:
```
Input(t) → [Linear] → [Activation] → [Linear] → ... → [Linear] → Output(V, m, h, n)
```

### 3. Design Justification
- **Why this activation?** Explain why `Tanh` (or your choice) is used — it must be differentiable for Autograd in the physics loss (Role 7 depends on this).
- **Why this depth/width?** Justify the number of layers and neurons (e.g., based on complexity of the ODE system or literature recommendations for PINNs).

### 4. Weight Initialization
- Document the initialization scheme used (e.g., Xavier, Kaiming, default PyTorch).
- Explain why it matters for PINN convergence.

---

> **Deliver to:** Roles 7 & 8 — The `HodgkinHuxleyPINN` class (nn.Module) for loss computation and training.
