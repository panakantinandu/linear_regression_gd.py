import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset

np.random.seed(42)  # reproducibility
m = 200
X = np.random.uniform(0, 5, m).reshape(-1, 1)
epsilon = np.random.normal(0, 1, m).reshape(-1, 1)  # Gaussian noise
y = 3 + 4 * X + epsilon   # true relation

# Add bias column (intercept term)
X_bias = np.c_[np.ones((m, 1)), X]

# Plot raw data
plt.scatter(X, y, alpha=0.6, label="Raw data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Data")
plt.legend()
plt.show()

#  Closed-form solution (Normal Eq)

theta_closed = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)

intercept_closed, slope_closed = theta_closed.flatten()
print("Closed-form solution:")
print(f"Intercept: {intercept_closed:.3f}, Slope: {slope_closed:.3f}")

y_pred_closed = X_bias @ theta_closed


# Gradient Descent Implementation

def compute_loss(X, y, theta):
    return np.mean((X @ theta - y)**2)

theta = np.zeros((2,1))  # [intercept, slope]
learning_rate = 0.05
iterations = 1000
losses = []

for i in range(iterations):
    gradients = (2/m) * X_bias.T @ (X_bias @ theta - y)
    theta -= learning_rate * gradients
    loss = compute_loss(X_bias, y, theta)
    losses.append(loss)

intercept_gd, slope_gd = theta.flatten()
print("\nGradient Descent solution:")
print(f"Intercept: {intercept_gd:.3f}, Slope: {slope_gd:.3f}")

y_pred_gd = X_bias @ theta


# Plots & Comparison

plt.scatter(X, y, alpha=0.5, label="Raw data")
plt.plot(X, y_pred_closed, color="red", label="Closed-form fit")
plt.plot(X, y_pred_gd, color="green", linestyle="--", label="Gradient Descent fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression: Closed-form vs Gradient Descent")
plt.legend()
plt.show()

# Loss curve
plt.plot(range(iterations), losses, color="purple")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss Curve")
plt.show()

# Explanation

print("\nExplanation:")
print("The closed-form solution gives the exact best-fit line. "
      "Gradient Descent converges close to the same slope and intercept after enough iterations "
      "with the correct learning rate. Both methods match well, confirming correctness.")
