# Adams-Bashforth 2 Performance & Complexity Report (Izhikevich 2007)

## 1. Numerical Implementation Details

The **Adams-Bashforth 2 (AB2)** method is a linear multi-step method used for solving ordinary differential equations. Unlike single-step methods like Euler or Runge-Kutta, AB2 utilizes derivative information from the two most recent time steps to predict the next state. This approach generally provides higher accuracy (second-order) than the standard Euler method.

The most critical challenge in applying AB2 to the Izhikevich model is the **Discrete Reset Condition**. Since the model incorporates a discontinuous jump when the membrane potential $v$ reaches $v_{peak}$, the multi-step "history" becomes mathematically invalid at the moment of a spike. To address this, the implementation follows a **History Flush Logic**:

- When $v \geq v_{peak}$, the state variables are immediately reset to $v = c$ and $w = w + d$.
- Following the reset, the solver executes a single **Forward Euler step** to restart the integration process and generate a new valid history point.
- Once the history is re-established, the solver resumes the second-order AB2 integration.

## 2. Performance Metrics

The following table summarizes the performance of the AB2 solver across different time-step resolutions for a 100 ms simulation.

| Metric | Value (dt=0.01ms) | Value (dt=0.1ms) | Value (dt=0.001ms) |
| :--- | :--- | :--- | :--- |
| **Execution Time** | 0.0442 s | 0.0045 s | 0.4355 s |
| **Memory Usage (Inc.)** | 0.0000 MB | 0.0000 MB | 0.6250 MB |
| **Stability** | Conditionally Stable | Conditionally Stable | Highly Stable |
| **Accuracy Order** | 2nd Order | 2nd Order | 2nd Order |

## 3. Complexity Analysis

### Time Complexity: $O(N)$
The time complexity is linearly proportional to the number of simulation steps $N$, where $N = (T_{end} - T_{start}) / dt$. Unlike implicit methods (e.g., Backward Euler), AB2 does not require internal root-finding iterations, making it computationally efficient per time step.

### Space Complexity: $O(N)$
The space complexity is $O(N \times D)$, where $D$ represents the dimensions of the state vector. The primary memory consumption stems from storing the simulation results (Time, $v$, $w$) for all $N$ steps.

## 4. Observations and Conclusions

The AB2 method offers a favorable balance between computational speed and numerical accuracy. However, as an explicit method, it is **conditionally stable**. If the time step $dt$ is too large, the solver may become unstable, especially during the high-frequency dynamics of a spike. The implemented **History Flush** mechanism ensures that the solver recovers correctly after each discrete reset, maintaining the integrity of the neural simulation.
