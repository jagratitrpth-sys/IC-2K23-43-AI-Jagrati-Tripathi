# Lab 7: Advanced AI Concepts (XAI, Adversarial AI, PCA/SVD)

## Problem Statement
Implement and demonstrate advanced AI techniques including dimensionality reduction, adversarial attacks on neural networks, and model interpretability.

## Concepts Implemented

### 1. PCA & SVD (Dimensionality Reduction)
* **SVD (Singular Value Decomposition):** A matrix factorization technique that breaks down a matrix into three matrices ($U, \Sigma, V^T$) to isolate the most important underlying structures.
* **PCA (Principal Component Analysis):** Uses SVD under the hood to project high-dimensional data into a lower-dimensional space while maximizing the variance (preserving as much information as possible).

### 2. Adversarial AI (FGSM)
The **Fast Gradient Sign Method (FGSM)** is an attack designed to fool a neural network. It works by using the model's own gradients to calculate the exact direction that will maximize the error/loss, and then adds a tiny, often imperceptible amount of noise ($\epsilon$) to the input image in that exact direction.
* Formula: $X_{adv} = X + \epsilon \cdot sign(\nabla_X J(\theta, X, y))$

### 3. Explainable AI (Grad-CAM)
**Gradient-weighted Class Activation Mapping (Grad-CAM)** makes "black box" CNN models transparent. It uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map (a heatmap) highlighting the important regions in the image for predicting the concept.
* **Mechanism:** It performs Global Average Pooling (GAP) on the gradients to calculate "importance weights," multiplies these weights by the forward feature maps, and applies a ReLU function to show only positive influences.