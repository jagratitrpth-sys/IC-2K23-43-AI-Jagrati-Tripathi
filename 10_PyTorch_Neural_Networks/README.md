# Lab 6: PyTorch Basics, Linear Regression, and Neural Networks

## Problem Statement
Introduce deep learning fundamentals using the PyTorch framework. The lab covers basic tensor operations, training a linear regression model using gradient descent, and constructing a multi-layer feedforward neural network architecture.

## Approach & Theoretical Concepts

### 1. Linear Regression via Gradient Descent
Instead of solving linear regression analytically, we use an iterative optimization approach:
* **Hypothesis:** $\hat{y} = wx + b$
* **Loss Function (MSE):** $J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$
* **Backpropagation:** PyTorch's `loss.backward()` automatically computes the partial derivatives (gradients) $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$.
* **Weight Update (SGD):** $w := w - \alpha \frac{\partial J}{\partial w}$, where $\alpha$ is the learning rate.

### 2. Feedforward Neural Network Fundamentals
Neural networks introduce non-linearity to solve complex problems. 
The architecture implemented in this lab consists of:
1. **Input Layer:** Takes a 2D tensor vector.
2. **Hidden Layer:** A linear transformation $Z = W_1X + b_1$ followed by a Rectified Linear Unit (ReLU) activation function $A = \max(0, Z)$.
3. **Output Layer:** A final linear transformation mapped through a Sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ to squash the output between 0 and 1.

## Expected Output
```text
--- 2. Linear Regression via Gradient Descent ---
Training Model for 200 epochs...
Final Training Loss: 0.0034
Prediction for x=5 (Expected exactly 11): 10.9015

--- 3. Neural Network Fundamentals ---
Network Architecture:
SimpleNN(
  (hidden): Linear(in_features=2, out_features=4, bias=True)
  (relu): ReLU()
  (output): Linear(in_features=4, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)