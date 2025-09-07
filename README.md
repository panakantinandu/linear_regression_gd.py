# Linear Regression: Closed-Form vs Gradient Descent

## Student Information
- **Name:** Nandu Panakanti
- **ID:** 700770346
- **Course:** Fall 2025 Machine Learning(CS5710-11438)

---

## Project Overview
This project implements **linear regression** using two approaches:
1. **Closed-Form Solution (Normal Equation)**
2. **Gradient Descent**  

The dataset was synthetically generated following the equation:
y = 3 + 4x + ε


---
---
# Project Metadata in YAML
```yaml
dataset:
  samples: 200
  x_range: [0, 5]
  equation: "y = 3 + 4x + ε"
gradient_descent:
  learning_rate: 0.05
  iterations: 1000
  initial_theta: [0, 0]
  final_theta:
    intercept: 3.105
    slope: 3.984
normal_equation:
  method: closed_form
  theta:
    intercept: 3.105
    slope: 3.984
plots:
  raw_data: plots/raw_data.png
  fitted_lines: plots/fitted_lines.png
  loss_curve: plots/loss_curve.png


**Project Overview**

This project implements linear regression using two approaches:

     1. Closed-Form Solution (Normal Equation)
     2. Gradient Descent

We generated a synthetic dataset following the equation:
y = 3 + 4x + ε  , where ε is Gaussian noise. A total of 200 samples were generated with x ∈ [0,5].


**Implementation Details**
Dataset Generation
  1. Generated 200 samples for x and y.
  2. Plotted raw data points.

Closed-Form Solution
  ->Added a bias column of 1's to X.
  ->Computed parameters using the Normal Equation:
                θ = (X^T X)^(-1) X^T y
Estimated Parameters:
           Intercept (θ₀): 3.105
           Slope (θ₁): 3.984
Fitted line plotted on raw data.

**Gradient Descent**

     Initialized θ = [0, 0]
     Learning rate η = 0.05
     Iterations = 1000
     Loss (MSE) plotted vs iterations
     Final Parameters after 1000 iterations:
        Intercept (θ₀): 3.105
        Slope (θ₁): 3.984
     Verified convergence to the Closed-Form solution.

**Observations**

Both Gradient Descent and the Closed-Form solution converge to the same parameters.
Gradient Descent required a learning rate of 0.05 and 1000 iterations to converge.
The loss curve shows a smooth decrease in MSE, indicating proper convergence.
The results confirm that Gradient Descent can approximate the Normal Equation solution accurately.

