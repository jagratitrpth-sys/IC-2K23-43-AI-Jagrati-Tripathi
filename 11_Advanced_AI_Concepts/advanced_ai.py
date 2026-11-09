import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

def pca_svd_demo():
    print("--- 1. PCA & SVD (Dimensionality Reduction) ---")
    # Dummy data: 5 samples, 3 features (e.g., 3D spatial coordinates)
    X = np.array([[2.5, 2.4, 0.5],
                  [0.5, 0.7, 0.1],
                  [2.2, 2.9, 0.8],
                  [1.9, 2.2, 0.3],
                  [3.1, 3.0, 0.9]])
    print(f"Original Data Shape (3D): {X.shape}")

    # Singular Value Decomposition (SVD)
    # Centering the data first
    X_centered = X - np.mean(X, axis=0)
    U, S, VT = np.linalg.svd(X_centered)
    print(f"SVD Singular Values: {S}")

    # Principal Component Analysis (PCA) via Scikit-Learn
    pca = PCA(n_components=2) # Reducing from 3D to 2D
    X_pca = pca.fit_transform(X)
    print(f"PCA Reduced Data Shape (2D): {X_pca.shape}")
    print(f"Explained Variance Ratio by the 2 components: {pca.explained_variance_ratio_}\n")


def fgsm_adversarial_demo():
    print("--- 2. Adversarial AI (FGSM Attack) ---")
    # Simple linear model acting as our "classifier"
    model = nn.Linear(3, 2)
    
    # Original Input (e.g., a normalized 3-pixel image)
    original_input = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    
    # Target label we want to evaluate against
    target_label = torch.tensor([1.0, 0.0]) 
    criterion = nn.MSELoss()
    
    # Forward pass on clean data
    output = model(original_input)
    print(f"Model Output (Clean Input): {output.detach().numpy()}")
    
    # Calculate loss and gradients
    loss = criterion(output, target_label)
    loss.backward()
    
    # FGSM Attack Formula: X_adv = X + epsilon * sign(grad(X))
    epsilon = 0.15 # Strength of the invisible noise
    data_grad = original_input.grad.data
    perturbed_input = original_input + epsilon * data_grad.sign()
    
    # Clip to maintain valid input range [0, 1]
    perturbed_input = torch.clamp(perturbed_input, 0, 1)
    
    # Forward pass on tampered data
    perturbed_output = model(perturbed_input)
    print(f"Model Output (Adversarial Input): {perturbed_output.detach().numpy()}\n")


def grad_cam_conceptual():
    print("--- 3. Explainable AI (Conceptual Grad-CAM) ---")
    print("Grad-CAM highlights which parts of an image influenced the AI's decision.")
    
    # A tiny Convolutional Neural Network
    class ToyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.fc = nn.Linear(9, 2)
            
        def forward(self, x):
            feature_map = self.conv(x)
            out = self.fc(feature_map.view(1, -1))
            return feature_map, out

    model = ToyCNN()
    # Dummy 3x3 grayscale image
    dummy_image = torch.ones(1, 1, 3, 3, requires_grad=True)
    
    # 1. Forward pass to get the feature map and final prediction
    feature_map, output = model(dummy_image)
    
    # 2. Isolate the score for the predicted class (Class 0)
    target_score = output[0, 0]
    
    # 3. Backward pass to get the gradients of the target class w.r.t the feature map
    gradients = torch.autograd.grad(target_score, feature_map)[0]
    
    # 4. Global Average Pooling (GAP) of the gradients to get importance weights (alpha)
    alpha = torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    # 5. Weighted combination of feature maps
    cam = torch.sum(alpha * feature_map, dim=1, keepdim=True)
    
    # 6. Apply ReLU to only keep features that have a positive influence on the class
    cam = torch.relu(cam)
    
    print(f"Input 3x3 Image:\n{dummy_image.squeeze().detach().numpy()}\n")
    print(f"Grad-CAM Heatmap (Values > 0 represent visual importance):\n{cam.squeeze().detach().numpy()}")


if __name__ == "__main__":
    pca_svd_demo()
    fgsm_adversarial_demo()
    grad_cam_conceptual()