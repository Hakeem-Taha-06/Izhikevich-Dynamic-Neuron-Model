
### Backward Euler Complexity Report

| Metric | Value |
| :--- | :--- |
| **Total Execution Time** | 2.3274 seconds |
| **Number of Steps** | 10001 |
| **Average Time per Step** | 0.232720 ms |
| **Memory Usage (Result Array)** | 0.3815 MB |
| **Peak Process Memory (RSS)** | 80.9648 MB |
| **Time Complexity** | O(N * I) where N is steps, I is fsolve iterations |
| **Space Complexity** | O(N * D) where N is steps, D is state dimensions (4+1) |

**Notes:**
- The implicit nature requires a root-finding iteration (`fsolve`) at each step.
- Stability is maintained even for large `dt`, unlike explicit methods.
